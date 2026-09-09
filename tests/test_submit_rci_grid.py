r"""Tests for the RCI Slurm launcher.

Two halves. The first exercises the pure planning functions in
`scripts/submit_rci_grid.py` -- ordering, chunking, lane assignment, the cost
model and the idempotency filters. The second actually *runs*
`slurm/rci/run_chain.sh` against a fake repository whose `.venv/bin/python` is a
shell stub that records its arguments, which is what proves the two files agree
on a command line and that the marker protocol behaves.

Neither half needs a GPU, a scheduler or ssh.
"""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

import scripts.submit_rci_grid as srg
from scripts.submit_rci_grid import (
    Chunk,
    ClusterState,
    Config,
    EPISODES,
    MAX_CONCURRENT_GPUS,
    MODE_ORDER,
    assign_lanes,
    build_matrix,
    build_sbatch_command,
    episodes_for,
    estimate_hours,
    format_walltime,
    hours_per_episode_trial,
    main,
    order_modes,
    order_pairs,
    parse_markers,
    parse_queue,
    pending_chunks,
    plan_chain,
    split_episodes,
    total_hours,
    wrap_remote,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CHAIN = REPO_ROOT / "slurm" / "rci" / "run_chain.sh"

REFERENCE_MODES = {"from_scratch", "naive", "joint"}
CL_METHOD_MODES = set(MODE_ORDER) - REFERENCE_MODES


@pytest.fixture
def cfg(tmp_path):
    """A Config pointing at a scratch tree, otherwise the shipped defaults."""
    return Config(repo=str(tmp_path / "repo"), out=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def test_cost_model_reproduces_the_measured_totals():
    """A full-roster plan must cost the measured total, scaled 8/7 for the mode
    the measurement did not include. If this drifts, every printed bill lies."""
    for dataset, measured in srg.DATASET_GPU_HOURS.items():
        tasks = [t for (ds, t) in EPISODES if ds == dataset]
        estimated = sum(
            estimate_hours(dataset, mode, episodes_for(dataset, task), 5, 1.0)
            for task in tasks
            for mode in MODE_ORDER
        )
        expected = measured * len(MODE_ORDER) / srg.MEASURED_MODES
        assert estimated == pytest.approx(expected, rel=0.05), dataset


def test_mode_cost_factors_average_to_one():
    """The per-mode factors are a heuristic laid over a measured aggregate. If
    their mean drifts from 1 the aggregate is no longer the anchor."""
    factors = [srg.MODE_COST_FACTOR[m] for m in MODE_ORDER]
    assert sum(factors) / len(factors) == pytest.approx(1.0, abs=0.05)


def test_every_roster_mode_has_a_cost_factor():
    assert set(MODE_ORDER) == set(srg.MODE_COST_FACTOR)


def test_hours_per_episode_trial_rejects_unmeasured_dataset():
    with pytest.raises(ValueError, match="no measured cost"):
        hours_per_episode_trial("rel-event")


def test_episodes_for_rejects_unmeasured_pair():
    """Guessing episode counts is what produced 2.7x-high ceilings last time."""
    with pytest.raises(ValueError, match="no measured episode count"):
        episodes_for("rel-f1", "qualifying-position")


def test_estimate_scales_with_seeds_and_device():
    base = estimate_hours("rel-stack", "ewc", 4, 5, 1.0)
    assert estimate_hours("rel-stack", "ewc", 4, 10, 1.0) == pytest.approx(2 * base)
    assert estimate_hours("rel-stack", "ewc", 4, 5, 1.8) == pytest.approx(1.8 * base)


def test_estimate_rejects_zero_seeds():
    with pytest.raises(ValueError, match="num_samples"):
        estimate_hours("rel-f1", "naive", 4, 0, 1.0)


def test_startup_is_billed_per_job():
    """Chunking finer is not free: each job re-pays dataset load. A bill that
    hid that would make a 94-job plan look like a 13-job one."""
    cfg = Config(startup_hours=0.25)
    chains = [plan_chain("rel-hm", "user-churn", "naive", cfg)]
    training, startup, grand = total_hours(chains, cfg)
    assert startup == pytest.approx(len(chains[0]) * 0.25)
    assert grand == pytest.approx(training + startup)
    assert startup > 0


# ---------------------------------------------------------------------------
# Chunking: short jobs
# ---------------------------------------------------------------------------

def test_split_episodes_last_chunk_holds_the_remainder():
    assert split_episodes(11, 4) == [4, 4, 3]
    assert split_episodes(52, 4) == [4] * 13
    assert split_episodes(2, 8) == [2]


def test_split_episodes_rejects_bad_input():
    with pytest.raises(ValueError, match="per_chunk"):
        split_episodes(10, 0)
    with pytest.raises(ValueError, match="total"):
        split_episodes(-1, 4)


def test_chunks_cover_the_whole_chain(cfg):
    for (dataset, task), episodes in EPISODES.items():
        for mode in MODE_ORDER:
            chain = plan_chain(dataset, task, mode, cfg)
            assert sum(c.episodes for c in chain) == episodes, (dataset, task, mode)
            assert [c.index for c in chain] == list(range(len(chain)))


def test_no_job_exceeds_the_partition_wall_limit(cfg):
    """The whole point of chunking. A chunk that cannot finish inside its wall
    limit is killed every time and the chain never advances."""
    for (dataset, task) in EPISODES:
        for mode in MODE_ORDER:
            for chunk in plan_chain(dataset, task, mode, cfg):
                job_hours = chunk.gpu_hours + cfg.startup_hours
                assert job_hours <= cfg.time_limit_hours, (chunk.chunk_key, job_hours)


def test_the_52_episode_chain_is_not_one_job(cfg):
    """rel-hm as a single job is a multi-day allocation, which is exactly what
    the fair-share budget punishes."""
    chain = plan_chain("rel-hm", "user-churn", "joint", cfg)
    assert len(chain) > 1
    assert max(c.episodes for c in chain) < EPISODES[("rel-hm", "user-churn")]


def test_cheap_chain_stays_a_single_job(cfg):
    """Chunking is a cost, not a virtue: an 11-episode rel-f1 chain fits in one
    4 h job and should not pay startup twelve times."""
    assert len(plan_chain("rel-f1", "driver-dnf", "naive", cfg)) == 1


def test_explicit_chunk_size_overrides_the_estimate(tmp_path):
    cfg = Config(out=str(tmp_path), chunk_episodes=3)
    assert [c.episodes for c in plan_chain("rel-f1", "driver-dnf", "naive", cfg)] \
        == [3, 3, 3, 2]


def test_chunk_size_must_be_positive(tmp_path):
    cfg = Config(out=str(tmp_path), chunk_episodes=0)
    with pytest.raises(ValueError, match="chunk-episodes"):
        plan_chain("rel-f1", "driver-dnf", "naive", cfg)


def test_impossible_time_budget_is_refused(tmp_path):
    cfg = Config(out=str(tmp_path), time_limit_hours=0.2, startup_hours=0.25)
    with pytest.raises(ValueError, match="nothing can run"):
        plan_chain("rel-f1", "driver-dnf", "naive", cfg)


def test_warns_when_a_single_episode_cannot_fit(tmp_path):
    """A silent version of this burns the whole wall limit, every job, forever."""
    cfg = Config(out=str(tmp_path), time_limit_hours=1.0, num_samples=5,
                 device_factor=1.8)
    chains = [plan_chain("rel-hm", "user-churn", "joint", cfg)]
    warnings = srg.plan_warnings(chains, cfg)
    assert any("cannot advance" in w for w in warnings)


def test_oversized_manual_chunk_is_flagged(tmp_path):
    """--chunk-episodes bypasses the sizing arithmetic, so the check that the
    job still fits its wall limit has to happen separately."""
    cfg = Config(out=str(tmp_path), chunk_episodes=40)
    chains = [plan_chain("rel-hm", "user-churn", "joint", cfg)]
    assert any("will kill it" in w for w in srg.plan_warnings(chains, cfg))


def test_well_sized_chunks_are_not_flagged(cfg):
    chains = [plan_chain("rel-hm", "user-churn", "joint", cfg)]
    assert not any("will kill it" in w for w in srg.plan_warnings(chains, cfg))


def test_uncached_dataset_is_flagged(cfg):
    """rel-f1 is not in the RCI relbench cache, and downloads must not happen
    by accident on a compute node."""
    warnings = srg.plan_warnings([plan_chain("rel-f1", "driver-dnf", "naive", cfg)], cfg)
    assert any("cache" in w and "rel-f1" in w for w in warnings)
    cached = srg.plan_warnings(
        [plan_chain("rel-stack", "user-badge", "naive", cfg)], cfg)
    assert not any("cache" in w for w in cached)


# ---------------------------------------------------------------------------
# Ordering: task-major and priority
# ---------------------------------------------------------------------------

def test_matrix_is_task_major(cfg):
    """All modes of a pair before any mode of the next: a pair with 7 of 8
    modes finished compares nothing."""
    pairs = [("rel-trial", "site-success"), ("rel-f1", "driver-dnf"),
             ("rel-f1", "driver-top3")]
    chains = build_matrix(pairs, MODE_ORDER, cfg)
    seen, order = set(), []
    for chain in chains:
        pair = (chain[0].dataset, chain[0].task)
        if not order or order[-1] != pair:
            assert pair not in seen, f"{pair} resumed after another pair"
            order.append(pair)
            seen.add(pair)
    assert len(order) == len(pairs)


def test_every_pair_gets_every_mode(cfg):
    pairs = [("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3")]
    chains = build_matrix(pairs, MODE_ORDER, cfg)
    for pair in pairs:
        modes = {c[0].mode for c in chains if (c[0].dataset, c[0].task) == pair}
        assert modes == set(MODE_ORDER)


def test_cheapest_dataset_first_whatever_the_input_order():
    given = [("rel-hm", "user-churn"), ("rel-stack", "post-votes"),
             ("rel-trial", "site-success"), ("rel-f1", "driver-dnf")]
    assert [d for d, _ in order_pairs(given)] == [
        "rel-f1", "rel-trial", "rel-stack", "rel-hm"]


def test_cheapest_task_first_within_a_dataset():
    given = [("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3")]
    assert order_pairs(given)[0] == ("rel-f1", "driver-top3")


def test_unmeasured_pairs_sort_last():
    given = [("rel-f1", "qualifying-position"), ("rel-hm", "user-churn")]
    assert order_pairs(given)[-1] == ("rel-f1", "qualifying-position")


def test_duplicate_pairs_are_not_planned_twice(cfg):
    pairs = [("rel-f1", "driver-dnf"), ("rel-f1", "driver-dnf")]
    chains = build_matrix(pairs, ["naive"], cfg)
    assert len(chains) == 1


def test_reference_frame_lands_before_the_cl_methods():
    ordered = order_modes(MODE_ORDER)
    last_reference = max(ordered.index(m) for m in REFERENCE_MODES)
    first_method = min(ordered.index(m) for m in CL_METHOD_MODES)
    assert last_reference < first_method


def test_order_modes_is_input_order_independent():
    assert order_modes(["der_pp", "from_scratch"]) == ["from_scratch", "der_pp"]


def test_order_modes_rejects_reproduction_only_mode():
    """ft_upsample exists to reproduce the submitted paper; it is not on the
    roster and must not be launchable by accident."""
    with pytest.raises(ValueError, match="unknown mode"):
        order_modes(["ft_upsample"])


def test_plan_chain_rejects_shell_metacharacters(cfg):
    with pytest.raises(ValueError, match="unsafe name"):
        plan_chain("rel-f1; rm -rf /", "driver-dnf", "naive", cfg)


# ---------------------------------------------------------------------------
# Lanes: the 4-GPU cap
# ---------------------------------------------------------------------------

def test_lane_count_is_capped_at_four(cfg):
    chains = build_matrix([("rel-f1", "driver-dnf")], MODE_ORDER, cfg)
    with pytest.raises(ValueError, match="the cap is 4"):
        assign_lanes(chains, [0, 1, 2, 3, 4])


def test_assign_lanes_rejects_no_lanes(cfg):
    with pytest.raises(ValueError, match="no lanes"):
        assign_lanes(build_matrix([("rel-f1", "driver-dnf")], ["naive"], cfg), [])


def test_a_chain_never_spans_two_lanes(cfg):
    """Two chunks of one chain in different lanes could run at once, and both
    would resume from the same increment."""
    chains = build_matrix([("rel-hm", "user-churn")], MODE_ORDER, cfg)
    plan = assign_lanes(chains, [0, 1, 2, 3])
    lane_of = {}
    for lane, chunks in plan.items():
        for chunk in chunks:
            lane_of.setdefault(chunk.chain_key, lane)
            assert lane_of[chunk.chain_key] == lane, chunk.chunk_key


def test_chunks_stay_in_index_order_within_a_lane(cfg):
    chains = build_matrix([("rel-hm", "user-churn")], MODE_ORDER, cfg)
    for chunks in assign_lanes(chains, [0, 1, 2, 3]).values():
        seen = {}
        for position, chunk in enumerate(chunks):
            previous = seen.get(chunk.chain_key)
            if previous is not None:
                assert chunk.index == previous[0] + 1
                assert position == previous[1] + 1, "chain interleaved with another"
            seen[chunk.chain_key] = (chunk.index, position)


def test_every_chunk_is_assigned_exactly_once(cfg):
    chains = build_matrix([("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3")],
                          MODE_ORDER, cfg)
    plan = assign_lanes(chains, [0, 1, 2, 3])
    assigned = [c.chunk_key for chunks in plan.values() for c in chunks]
    assert sorted(assigned) == sorted(c.chunk_key for chain in chains for c in chain)


def test_lanes_are_load_balanced(cfg):
    """The packer must reach the best balance the input actually allows.

    A fixed max/min ratio is the wrong bar. Chunks of one chain are
    sequentially dependent -- each resumes the previous chunk's checkpoint --
    so a chain cannot be spread across lanes: it is one indivisible item. Once
    the measured cost of ewc made a single chain (25.2 GPU-h on rel-stack)
    larger than the ideal per-lane share (68.1 / 4 = 17.0), NO assignment can
    hit 1.5x, and a test demanding it would only be satisfied by pretending
    ewc is cheap again.

    So assert the real property: no lane exceeds the theoretical optimum, which
    is the greater of the ideal share and the largest indivisible chain.
    """
    chains = build_matrix([("rel-stack", "user-badge")], MODE_ORDER, cfg)
    lanes = assign_lanes(chains, [0, 1, 2, 3])
    loads = [sum(c.gpu_hours for c in chunks) for chunks in lanes.values()]

    chain_costs = [sum(c.gpu_hours for c in chain) for chain in chains]
    optimum = max(sum(chain_costs) / len(lanes), max(chain_costs))
    assert max(loads) <= optimum * 1.05, (
        f"loads {loads} exceed the achievable optimum {optimum:.1f}"
    )
    # And the packer must still be doing real work: no lane left empty while
    # another carries more than one chain's worth.
    assert min(loads) > 0


# ---------------------------------------------------------------------------
# Restartability and idempotency
# ---------------------------------------------------------------------------

def test_parse_markers_splits_chunk_and_chain_markers():
    listing = ("rel-f1__driver-dnf__naive__c00.done\n"
               "rel-f1__driver-dnf__naive__c01.done\n"
               "rel-f1__driver-dnf__er.chain-complete\n"
               "chain-params.txt\n")
    done, complete = parse_markers(listing)
    assert done == {"rel-f1__driver-dnf__naive__c00", "rel-f1__driver-dnf__naive__c01"}
    assert complete == {"rel-f1__driver-dnf__er"}


def test_parse_queue_reads_lane_and_chain_from_job_names():
    listing = ("clg-L2-rel-f1__driver-dnf__naive__c03\n"
               "clg-L0-rel-hm__user-churn__er__c11\n"
               "some-unrelated-job\n")
    chains, lanes = parse_queue(listing)
    assert chains == {"rel-f1__driver-dnf__naive", "rel-hm__user-churn__er"}
    assert lanes == {0, 2}


def test_pending_skips_done_chunks(cfg):
    chain = plan_chain("rel-hm", "user-churn", "naive", cfg)
    state = ClusterState(done_chunks={chain[0].chunk_key, chain[2].chunk_key})
    pending = pending_chunks(chain, state)
    assert [c.index for c in pending] == [c.index for c in chain[1:]
                                          if c.index != 2]


def test_pending_skips_a_completed_chain(cfg):
    chain = plan_chain("rel-hm", "user-churn", "naive", cfg)
    state = ClusterState(complete_chains={chain[0].chain_key})
    assert pending_chunks(chain, state) == []


def test_a_chain_with_anything_queued_is_skipped_whole(cfg):
    """Not just the queued chunk. Submitting a sibling chunk into a free lane
    would put two jobs of one chain in flight, both resuming from the same
    increment."""
    chain = plan_chain("rel-hm", "user-churn", "naive", cfg)
    state = ClusterState(busy_chains={chain[0].chain_key})
    assert pending_chunks(chain, state) == []


def test_finished_chain_yields_nothing(cfg):
    chain = plan_chain("rel-f1", "driver-dnf", "naive", cfg)
    state = ClusterState(done_chunks={c.chunk_key for c in chain})
    assert pending_chunks(chain, state) == []


# ---------------------------------------------------------------------------
# sbatch command construction
# ---------------------------------------------------------------------------

def test_sbatch_requests_exactly_one_gpu(cfg):
    chunk = plan_chain("rel-f1", "driver-dnf", "naive", cfg)[0]
    command = build_sbatch_command(chunk, 0, cfg, None)
    assert "--gres=gpu:1" in command
    assert "--gres=gpu:2" not in command


def test_first_job_of_a_lane_has_no_dependency(cfg):
    chunk = plan_chain("rel-f1", "driver-dnf", "naive", cfg)[0]
    assert "--dependency" not in build_sbatch_command(chunk, 0, cfg, None)


def test_later_jobs_depend_on_the_previous_one(cfg):
    chunk = plan_chain("rel-f1", "driver-dnf", "naive", cfg)[0]
    command = build_sbatch_command(chunk, 1, cfg, "12345")
    assert "--dependency=afterany:12345" in command


def test_job_name_encodes_lane_and_chunk_and_round_trips(cfg):
    chunk = plan_chain("rel-hm", "user-churn", "der_pp", cfg)[3]
    chains, lanes = parse_queue(chunk.job_name(2))
    assert lanes == {2}
    assert chains == {"rel-hm__user-churn__der_pp"}


def test_walltime_formats_and_rejects_zero():
    assert format_walltime(4.0) == "4:00:00"
    assert format_walltime(0.5) == "0:30:00"
    assert format_walltime(1 / 60) == "0:01:00"
    with pytest.raises(ValueError):
        format_walltime(0)


def test_wrap_remote_uses_ssh_only_when_asked():
    assert wrap_remote("ls", None) == ["bash", "-c", "ls"]
    assert wrap_remote("ls", "rci.cvut.cz") == ["ssh", "rci.cvut.cz", "ls"]


def test_mkdir_covers_the_slurm_output_directory(cfg):
    """Slurm opens --output before the job script runs and will not create the
    directory, so a missing one fails every job in the submission."""
    chains = build_matrix([("rel-f1", "driver-dnf")], ["naive"], cfg)
    made = srg.build_mkdir_command(cfg, chains).split()
    sbatch = build_sbatch_command(chains[0][0], 0, cfg, None)
    output_dir = re.search(r"--output=(\S+)/[^/\s]+", sbatch).group(1)
    assert output_dir in made
    assert cfg.marker_dir in made


# ---------------------------------------------------------------------------
# The driver end to end, with the cluster faked out
# ---------------------------------------------------------------------------

class FakeCluster:
    """Stands in for `bash -c` / `ssh`: answers ls and squeue, logs sbatch."""

    def __init__(self, markers="", queue=""):
        self.markers = markers
        self.queue = queue
        self.submitted = []
        self.next_job_id = 1000

    def __call__(self, command, ssh_host=None):
        if command.startswith("ls -1"):
            return subprocess.CompletedProcess(command, 0, self.markers, "")
        if command.startswith("squeue"):
            return subprocess.CompletedProcess(command, 0, self.queue, "")
        if command.startswith("mkdir"):
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command.startswith("sbatch"), command
        self.submitted.append(command)
        self.next_job_id += 1
        return subprocess.CompletedProcess(command, 0, f"{self.next_job_id}\n", "")


@pytest.fixture
def cluster(monkeypatch):
    fake = FakeCluster()
    monkeypatch.setattr(srg, "run_command", fake)
    return fake


def test_dry_run_prints_the_cost_and_the_commands(cluster, capsys):
    # `cluster` is not decoration: without it this reads the REAL marker
    # directory, so the plan shrinks as the operator's grid fills up. It passed
    # on an empty machine and failed on RCI, where tier A is finished, printing
    # one sbatch instead of nine. A planner test must describe the planner, not
    # the state of somebody's scratch directory.
    assert main(["--pairs", "rel-f1:driver-dnf", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "GPU-h" in out and "TOTAL" in out
    # One sbatch per CHUNK, not per mode: an expensive mode is split so that no
    # single job can exceed the partition wall. ewc on an 11-episode task is two
    # chunks, so this is 9 for 8 modes -- and the count must track the plan
    # rather than the roster, or the test silently passes when chunking breaks.
    expected_chunks = sum(
        len(chain) for chain in build_matrix([("rel-f1", "driver-dnf")], MODE_ORDER, Config())
    )
    assert expected_chunks > len(MODE_ORDER), "chunking should split at least one mode"
    assert out.count("sbatch --parsable") == expected_chunks
    assert "run_chain.sh" in out


def test_dry_run_submits_nothing(cluster):
    assert main(["--pairs", "rel-f1:driver-dnf", "--dry-run"]) == 0
    assert cluster.submitted == []


def test_submission_never_uses_more_than_four_lanes(cluster):
    main(["--tier", "A", "--yes"])
    lanes = {re.search(r"--job-name=clg-L(\d)-", c).group(1)
             for c in cluster.submitted}
    assert len(lanes) <= MAX_CONCURRENT_GPUS


def test_each_lane_is_a_serial_dependency_chain(cluster):
    main(["--pairs", "rel-f1:driver-dnf", "--yes"])
    heads = 0
    for command in cluster.submitted:
        if "--dependency" not in command:
            heads += 1
    assert heads == len({re.search(r"clg-L(\d)-", c).group(1)
                         for c in cluster.submitted})


def test_busy_lanes_still_receive_work(monkeypatch, capsys):
    """New work goes into every lane, including ones already in flight.

    Skipping busy lanes was strictly worse than queueing: with three of four
    lanes occupied, the entire remainder was funnelled into the one free lane and
    ran serially on a single GPU. Slurm enforces the concurrency limit itself, so
    extra jobs queue rather than over-running.
    """
    import scripts.submit_rci_grid as m

    def busy(cfg, ssh):
        state = m.ClusterState()
        state.busy_lanes = {0, 2, 3}
        return state

    monkeypatch.setattr(m, "read_state", busy)
    monkeypatch.setattr(m, "run_command", lambda *a, **k: subprocess.CompletedProcess(
        args="sbatch", returncode=0, stdout="Submitted batch job 1\n", stderr=""))

    rc = m.main(["--tier", "A", "--num-samples", "1", "--dry-run"])
    assert rc in (0, None)
    out = capsys.readouterr().out
    assert "lane 0" in out and "lane 3" in out, "busy lanes must still be used"


def test_all_lanes_busy_still_submits(monkeypatch, capsys):
    # Queueing behind in-flight work is the intended outcome, not a reason to
    # submit nothing and make someone re-run the command later.
    import scripts.submit_rci_grid as m

    def busy(cfg, ssh):
        state = m.ClusterState()
        state.busy_lanes = {0, 1, 2, 3}
        return state

    monkeypatch.setattr(m, "read_state", busy)
    monkeypatch.setattr(m, "run_command", lambda *a, **k: subprocess.CompletedProcess(
        args="sbatch", returncode=0, stdout="Submitted batch job 1\n", stderr=""))

    rc = m.main(["--tier", "A", "--num-samples", "1", "--dry-run"])
    assert rc in (0, None)
    out = capsys.readouterr().out
    assert "already have work in flight" in out
    assert "sbatch" in out, "work must still be planned"


def test_resubmission_skips_completed_work(monkeypatch, capsys):
    first = FakeCluster()
    monkeypatch.setattr(srg, "run_command", first)
    main(["--pairs", "rel-f1:driver-dnf", "--yes"])
    assert first.submitted

    markers = "".join(
        re.search(r"--job-name=clg-L\d-(\S+)", c).group(1) + ".done\n"
        for c in first.submitted)
    second = FakeCluster(markers=markers)
    monkeypatch.setattr(srg, "run_command", second)
    assert main(["--pairs", "rel-f1:driver-dnf", "--yes"]) == 0
    assert second.submitted == []
    assert "Nothing left to submit" in capsys.readouterr().out


def test_queued_chain_is_not_resubmitted(monkeypatch):
    fake = FakeCluster(queue="clg-L0-rel-f1__driver-dnf__naive__c00\n")
    monkeypatch.setattr(srg, "run_command", fake)
    main(["--pairs", "rel-f1:driver-dnf", "--yes"])
    assert not any("--mode naive" in c for c in fake.submitted)
    assert any("--mode ewc" in c for c in fake.submitted)


def test_unknown_scheduler_is_reported_not_assumed_empty(monkeypatch, capsys):
    def failing(command, ssh_host=None):
        return subprocess.CompletedProcess(command, 127, "", "squeue: not found")

    monkeypatch.setattr(srg, "run_command", failing)
    main(["--pairs", "rel-f1:driver-dnf", "--dry-run"])
    assert "could not read squeue" in capsys.readouterr().out


@pytest.fixture
def squeue_shim(tmp_path, monkeypatch):
    """Put a fake `squeue` first on PATH and run read_state through a real shell.

    Faking `run_command` cannot test this: the thing under test is the command
    string itself, and whether a scheduler that fails is distinguishable from a
    queue that is empty.
    """

    def install(body, code=0):
        shim = tmp_path / "bin"
        shim.mkdir(exist_ok=True)
        squeue = shim / "squeue"
        squeue.write_text(f"#!/bin/bash\n{body}\nexit {code}\n")
        squeue.chmod(squeue.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")

    return install


def test_read_state_reports_a_broken_scheduler(squeue_shim, tmp_path):
    """An empty queue and no scheduler must not look the same: one means no
    lane is busy, the other means we cannot tell."""
    squeue_shim("echo 'squeue: error' >&2", code=1)
    state = srg.read_state(Config(out=str(tmp_path)), None)
    assert state.known is False


def test_read_state_reads_a_working_queue(squeue_shim, tmp_path):
    squeue_shim("echo clg-L2-rel-f1__driver-dnf__naive__c00")
    state = srg.read_state(Config(out=str(tmp_path)), None)
    assert state.known is True
    assert state.busy_lanes == {2}
    assert state.busy_chains == {"rel-f1__driver-dnf__naive"}


def test_read_state_tolerates_a_missing_marker_directory(squeue_shim, tmp_path):
    """First run on a fresh checkout: no markers yet is normal, not an error."""
    squeue_shim("")
    state = srg.read_state(Config(out=str(tmp_path / "never-created")), None)
    assert state.known is True
    assert state.done_chunks == set()


def test_read_state_reads_markers_from_disk(squeue_shim, tmp_path):
    squeue_shim("")
    cfg = Config(out=str(tmp_path))
    markers = Path(cfg.marker_dir)
    markers.mkdir(parents=True)
    (markers / "rel-f1__driver-dnf__naive__c00.done").touch()
    (markers / "rel-f1__driver-dnf__er.chain-complete").touch()
    state = srg.read_state(cfg, None)
    assert state.done_chunks == {"rel-f1__driver-dnf__naive__c00"}
    assert state.complete_chains == {"rel-f1__driver-dnf__er"}


def test_refuses_to_spend_fair_share_non_interactively(cluster, monkeypatch):
    monkeypatch.setattr(srg.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit, match="--yes"):
        main(["--pairs", "rel-f1:driver-dnf"])
    assert cluster.submitted == []


def test_time_limit_beyond_the_partition_is_refused():
    with pytest.raises(SystemExit, match="exceeds"):
        main(["--pairs", "rel-f1:driver-dnf", "--partition", "amdgpufast",
              "--time-limit", "24", "--dry-run"])


def test_more_lanes_than_gpus_is_refused():
    with pytest.raises(SystemExit, match="between 1 and 4"):
        main(["--pairs", "rel-f1:driver-dnf", "--lanes", "8", "--dry-run"])


def test_v100_partition_costs_more_than_a100(cluster, capsys):
    # Same reason as above: with real markers this prints no "==>" summary at
    # all once the selection is already done, and the parse below IndexErrors.
    def total(partition):
        main(["--pairs", "rel-trial:site-success", "--partition", partition,
              "--dry-run"])
        line = [l for l in capsys.readouterr().out.splitlines()
                if l.startswith("==>")][0]
        return float(re.search(r"~([\d.]+) GPU-h", line).group(1))

    assert total("gpu") > total("amdgpufast")


# ---------------------------------------------------------------------------
# slurm/rci/run_chain.sh, actually executed
# ---------------------------------------------------------------------------

STUB_TEMPLATE = """#!/bin/bash
echo "ARGS: $@"
{body}
exit {code}
"""


@pytest.fixture
def fake_repo(tmp_path):
    """A repo whose interpreter is a stub that records how it was called."""

    def build(body="", code=0):
        repo = tmp_path / "repo"
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / "experiments" / "continuous_learning").mkdir(parents=True,
                                                             exist_ok=True)
        (repo / "experiments" / "continuous_learning"
         / "continuous_learning.py").write_text("")
        python = repo / ".venv" / "bin" / "python"
        python.write_text(STUB_TEMPLATE.format(body=body, code=code))
        python.chmod(python.stat().st_mode | stat.S_IEXEC)
        return repo

    return build


def run_chain(repo, out, extra_env=None, **kwargs):
    """Invoke run_chain.sh with the flags submit_rci_grid.py would pass."""
    args = ["--dataset", "rel-f1", "--task", "driver-dnf", "--mode", "er",
            "--chunk", "0", "--episodes", "4", "--num-samples", "5",
            "--seed", "42", "--out", str(out),
            "--mlflow-uri", "http://example:2222",
            "--mlflow-experiment", "test_er", "--cpus-per-trial", "4"]
    for flag, value in (kwargs or {}).items():
        flag = "--" + flag.replace("_", "-")
        args[args.index(flag) + 1] = str(value)
    env = dict(os.environ, CL_REPO=str(repo), CUDA_VISIBLE_DEVICES="3")
    env.pop("SLURM_CPUS_PER_TASK", None)
    env.update(extra_env or {})
    return subprocess.run([str(RUN_CHAIN), *args], capture_output=True, text=True,
                          env=env)


def read_log(out, chunk_key="rel-f1__driver-dnf__er__c00"):
    chain_key = chunk_key.rsplit("__c", 1)[0]
    return (Path(out) / chain_key / f"{chunk_key}.log").read_text()


def test_run_chain_passes_resume_and_max_increments(fake_repo, tmp_path):
    """Chunking only works if every job resumes and stops after K episodes."""
    out = tmp_path / "out"
    result = run_chain(fake_repo(), out)
    assert result.returncode == 0, result.stderr
    log = read_log(out)
    assert "--resume" in log
    assert "--max_increments=4" in log
    assert "--learning_mode=er" in log


def test_run_chain_does_not_pin_a_gpu(tmp_path):
    """The job must ask for a GPU and let Slurm place it.

    Slurm restricts the job to its allocation and exports ids relative to it, so
    re-pinning inside the job can only agree with the scheduler or contradict it.
    Requesting resources (--gres=gpu:1) and leaving placement alone is both
    simpler and the cluster's expectation.
    """
    script = (REPO_ROOT / "slurm" / "rci" / "run_chain.sh").read_text()
    # Strip comments: the file explains WHY it does not pin, and that prose
    # mentions the flag. Assert what the script executes, not what it says.
    code = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--gres=gpu:1" in script, "the job must still request a GPU"
    assert "--gpu_ids" not in code, "must not pin a device inside the allocation"
    assert "--nodelist" not in code, "must not pin a node"
    # An `echo` that reports what Slurm handed us is wanted; an assignment that
    # overrides it is not.
    assignments = [
        line for line in code.splitlines()
        if re.match(r"\s*(export\s+)?CUDA_VISIBLE_DEVICES=", line)
    ]
    assert not assignments, f"must not override Slurm's allocation: {assignments}"


def test_run_chain_model_save_dir_does_not_depend_on_the_chunk(fake_repo, tmp_path):
    """chain_id is dataset/task/mode/model_save_dir. If the path moved per
    chunk, every chunk would start a fresh chain at episode 1."""
    out = tmp_path / "out"
    repo = fake_repo()
    run_chain(repo, out, chunk=0)
    run_chain(repo, out, chunk=1)

    def save_dir(chunk_key):
        return re.search(r"--model_save_dir=(\S+)", read_log(out, chunk_key)).group(1)

    assert save_dir("rel-f1__driver-dnf__er__c00") == \
        save_dir("rel-f1__driver-dnf__er__c01")


def test_run_chain_writes_a_marker_on_success(fake_repo, tmp_path):
    out = tmp_path / "out"
    run_chain(fake_repo(), out)
    assert (out / "markers" / "rel-f1__driver-dnf__er__c00.done").exists()


def test_run_chain_writes_no_marker_on_failure(fake_repo, tmp_path):
    """A failed chunk must be retried, so it must not look done."""
    out = tmp_path / "out"
    result = run_chain(fake_repo(code=1), out)
    assert result.returncode == 1
    assert not (out / "markers" / "rel-f1__driver-dnf__er__c00.done").exists()


def test_run_chain_marks_the_chain_complete_on_the_sentinel(fake_repo, tmp_path):
    out = tmp_path / "out"
    repo = fake_repo(body='echo "All increments are already completed according '
                          'to MLflow. Exiting."')
    run_chain(repo, out)
    assert (out / "markers" / "rel-f1__driver-dnf__er.chain-complete").exists()


def test_run_chain_does_not_mark_complete_when_it_is_not(fake_repo, tmp_path):
    out = tmp_path / "out"
    run_chain(fake_repo(body='echo "Limiting to 4 increment(s): 1..4"'), out)
    assert not (out / "markers" / "rel-f1__driver-dnf__er.chain-complete").exists()


def test_run_chain_exits_early_once_the_chain_is_complete(fake_repo, tmp_path):
    """The queued tail of a lane must cost seconds, not a dataset load each."""
    out = tmp_path / "out"
    repo = fake_repo(body='echo RAN >> "$CL_WITNESS"')
    (out / "markers").mkdir(parents=True)
    (out / "markers" / "rel-f1__driver-dnf__er.chain-complete").touch()
    witness = tmp_path / "witness"
    result = run_chain(repo, out, extra_env={"CL_WITNESS": str(witness)})
    assert result.returncode == 0
    assert not witness.exists()


def test_run_chain_is_idempotent_for_a_done_chunk(fake_repo, tmp_path):
    out = tmp_path / "out"
    repo = fake_repo(body='echo RAN >> "$CL_WITNESS"')
    witness = tmp_path / "witness"
    run_chain(repo, out, extra_env={"CL_WITNESS": str(witness)})
    assert witness.read_text().count("RAN") == 1
    run_chain(repo, out, extra_env={"CL_WITNESS": str(witness)})
    assert witness.read_text().count("RAN") == 1


def test_run_chain_refuses_a_changed_seed_count(fake_repo, tmp_path):
    """expected_trials is len(seeds): resuming with more seeds than the earlier
    chunks ran silently restarts the chain at episode 1."""
    out = tmp_path / "out"
    repo = fake_repo()
    run_chain(repo, out, chunk=0)
    result = run_chain(repo, out, chunk=1, num_samples=3)
    assert result.returncode == 2
    assert "Refusing" in result.stderr


def test_run_chain_accepts_the_same_seed_count(fake_repo, tmp_path):
    out = tmp_path / "out"
    repo = fake_repo()
    run_chain(repo, out, chunk=0)
    assert run_chain(repo, out, chunk=1).returncode == 0


def test_run_chain_rejects_missing_arguments(fake_repo, tmp_path):
    result = subprocess.run(
        [str(RUN_CHAIN), "--dataset", "rel-f1"], capture_output=True, text=True,
        env=dict(os.environ, CL_REPO=str(fake_repo())))
    assert result.returncode == 2
    assert "--task is required" in result.stderr


def test_run_chain_rejects_unknown_arguments(fake_repo):
    result = subprocess.run(
        [str(RUN_CHAIN), "--nonsense", "1"], capture_output=True, text=True,
        env=dict(os.environ, CL_REPO=str(fake_repo())))
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_generated_sbatch_command_is_understood_by_run_chain(fake_repo, tmp_path):
    """The cross-file contract: whatever submit_rci_grid.py emits after the
    script path must parse. A `--flag=value` / `--flag value` mismatch here
    would fail only on the cluster, one job at a time."""
    out = tmp_path / "out"
    cfg = Config(repo=str(fake_repo()), out=str(out))
    chunk = plan_chain("rel-hm", "user-churn", "der_pp", cfg)[2]
    command = build_sbatch_command(chunk, 1, cfg, "999")
    tail = command.split("run_chain.sh", 1)[1].split()

    result = subprocess.run([str(RUN_CHAIN), *tail], capture_output=True, text=True,
                            env=dict(os.environ, CL_REPO=cfg.repo,
                                     CUDA_VISIBLE_DEVICES="0"))
    assert result.returncode == 0, result.stderr
    log = read_log(out, chunk.chunk_key)
    assert f"--max_increments={chunk.episodes}" in log
    assert "--learning_mode=der_pp" in log
    assert "--num_samples=5" in log
    assert (out / "markers" / f"{chunk.chunk_key}.done").exists()


# --- regression: the 4-GPU cap must not be lost to a transient squeue failure --


def test_unreadable_queue_warns_and_still_submits(monkeypatch, capsys):
    """An unreadable squeue must not stop submission.

    Lanes are how this script schedules against the 4-GPU limit, but Slurm is
    what enforces it -- the account is capped, so extra work queues rather than
    over-running. An unknown queue therefore costs latency (work dealt into a
    lane that is actually busy waits behind it), not quota, and refusing would
    block legitimate submissions on a transient ssh failure.
    """
    import scripts.submit_rci_grid as m

    def unreadable(cfg, ssh):
        state = m.ClusterState()
        state.known = False
        return state

    monkeypatch.setattr(m, "read_state", unreadable)
    submitted = []
    monkeypatch.setattr(
        m, "run_command",
        lambda *a, **k: submitted.append(a) or subprocess.CompletedProcess(
            args="sbatch", returncode=0, stdout="Submitted batch job 1\n", stderr=""
        ),
    )

    rc = m.main(["--yes", "--pairs", "rel-f1:driver-top3", "--modes", "naive"])
    assert rc in (0, None)
    assert submitted, "an unknown queue must not block submission"
    assert "could not read squeue" in capsys.readouterr().out


def test_dry_run_is_never_refused_even_when_the_queue_is_unreadable(monkeypatch, capsys):
    # A dry run submits nothing, so it cannot breach the cap. Refusing it would
    # remove the one way to inspect a plan while the cluster is unreachable.
    import scripts.submit_rci_grid as m

    def unreadable(cfg, ssh):
        state = m.ClusterState()
        state.known = False
        return state

    monkeypatch.setattr(m, "read_state", unreadable)
    submitted = []
    monkeypatch.setattr(
        m, "run_command",
        lambda *a, **k: submitted.append(a) or subprocess.CompletedProcess(
            args="sbatch", returncode=0, stdout="Submitted batch job 1\n", stderr=""
        ),
    )

    rc = m.main(["--pairs", "rel-f1:driver-top3", "--modes", "naive", "--dry-run"])
    assert rc in (0, None)
    assert not submitted, "a dry run must not submit"


def test_ignore_state_allows_submission_when_the_queue_is_unreadable(monkeypatch, capsys):
    # The escape hatch must remain for someone who has checked the queue by hand.
    import scripts.submit_rci_grid as m

    def unreadable(cfg, ssh):
        state = m.ClusterState()
        state.known = False
        return state

    monkeypatch.setattr(m, "read_state", unreadable)
    monkeypatch.setattr(
        m, "run_command",
        lambda *a, **k: subprocess.CompletedProcess(
            args="sbatch", returncode=0, stdout="Submitted batch job 12345\n", stderr=""
        ),
    )
    rc = m.main(["--yes", "--ignore-state", "--pairs", "rel-f1:driver-top3",
                 "--modes", "naive"])
    assert rc in (0, None), "--ignore-state must not refuse"


# --- multi-partition placement ----------------------------------------------


def test_a_partition_list_is_summarised_conservatively():
    """Slurm can place a job on whichever listed partition frees up first.

    That is the lever that matters when every partition is fully allocated: the
    queue wait dominates, not the device. The summary must be pessimistic --
    shortest wall limit so a job cannot outlive whichever partition takes it, and
    slowest device so the cost estimate is an upper bound.
    """
    import scripts.submit_rci_grid as m

    combined = m.resolve_partition("amdgpufast,gpuextralong")
    assert combined.name == "amdgpufast,gpuextralong"  # passed through to sbatch
    assert combined.max_hours == 4.0, "must take the SHORTEST wall limit"
    assert combined.device == "V100", "must take the SLOWEST device"
    assert m.DEVICE_FACTOR[combined.device] == 1.8


def test_single_partition_still_resolves_exactly():
    import scripts.submit_rci_grid as m

    p = m.resolve_partition("amdgpufast")
    assert (p.name, p.max_hours, p.device) == ("amdgpufast", 4.0, "A100")


def test_unknown_partition_in_a_list_is_rejected():
    import scripts.submit_rci_grid as m

    with pytest.raises(ValueError, match="unknown partition"):
        m.resolve_partition("amdgpufast,not_a_partition")


def test_empty_partition_spec_is_rejected():
    import scripts.submit_rci_grid as m

    with pytest.raises(ValueError, match="at least one"):
        m.resolve_partition(" , ")
