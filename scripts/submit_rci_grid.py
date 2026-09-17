r"""Submit the continual-learning grid to the RCI Slurm cluster.

`scripts/run_grid.py` drives the four local A100s; this drives RCI, where the
constraints are different and tighter:

* **At most 4 GPUs at once, ever.** Enforced structurally, not by hope: the work
  is dealt into at most 4 *lanes*, and every job in a lane declares
  ``--dependency=afterany`` on the previous job of that lane. One lane is one
  GPU, so 4 lanes is 4 GPUs no matter how many jobs are queued. A lane already
  occupied by an earlier submission is detected in `squeue` and left alone, so
  running this twice cannot add a fifth GPU.
* **Short jobs.** A chain of 52 episodes as one job is a three-day
  `gpuextralong` allocation, and fair-share charges for it. Instead each job
  runs `--max_increments K` and `--resume`, sized so the job fits the
  partition's wall limit; the chain continues in the next job of its lane.
* **Task-major.** All 8 modes of one (dataset, task) are planned and submitted
  before the next pair. A pair with 7 of 8 modes finished compares nothing, so a
  half-finished submission should leave whole, comparable cells behind.
* **Priority.** Cheapest datasets first (rel-f1, rel-trial, ...), and within a
  pair the reference frame (from_scratch, naive, joint) before the CL methods.

Chunking rests on the `--resume` contract in `continuous_learning.py`; see the
header of `slurm/rci/run_chain.sh` for the three preconditions it imposes and
where they are held.

Usage::

    # what would it cost and what would it run?
    .venv/bin/python scripts/submit_rci_grid.py --tier A --dry-run

    # from this machine, submitting over ssh
    .venv/bin/python scripts/submit_rci_grid.py --tier A --ssh rci.cvut.cz --yes

    # on the RCI login node
    .venv/bin/python scripts/submit_rci_grid.py --pairs rel-f1:driver-dnf

Nothing here needs a GPU, a cluster or ssh to be exercised: `--dry-run` prints
the job matrix, the fair-share estimate and every `sbatch` command it would run.
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# The hard cap from the RCI usage rules. Lanes are GPUs, so this caps lanes.
MAX_CONCURRENT_GPUS = 4

# Set by the RCI port; `$HOME` is expanded by the shell that runs the command,
# which is the remote shell under --ssh and the local one otherwise.
DEFAULT_REPO = "$HOME/git/claude-redelex"

# Tiers as in scripts/run_grid.py, plus E for rel-amazon: run_grid has no tier
# for it, but the measured cost table does, so it can be planned.
TIERS: Dict[str, List[Tuple[str, str]]] = {
    "A": [("rel-f1", "driver-position"), ("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3"),
          ("rel-trial", "study-outcome"), ("rel-trial", "study-adverse"),
          ("rel-trial", "site-success")],
    "B": [("rel-hm", "user-churn"), ("rel-hm", "item-sales")],
    "C": [("rel-stack", "user-engagement"), ("rel-stack", "post-votes"),
          ("rel-stack", "user-badge")],
    "D": [("rel-ratebeer", "beer-churn"), ("rel-ratebeer", "user-churn"),
          ("rel-ratebeer", "user-count"), ("rel-ratebeer", "brewer-dormant")],
    "E": [("rel-amazon", "user-churn"), ("rel-amazon", "user-ltv"),
          ("rel-amazon", "item-churn"), ("rel-amazon", "item-ltv")],
}

# Measured episode counts, not metadata ceilings -- the ceilings were up to 2.7x
# high. See analysis/dataset-episodes-measured.md.
EPISODES: Dict[Tuple[str, str], int] = {
    ("rel-f1", "driver-position"): 11,
    ("rel-f1", "driver-dnf"): 11,
    ("rel-f1", "driver-top3"): 2,
    ("rel-trial", "study-outcome"): 7,
    ("rel-trial", "study-adverse"): 7,
    ("rel-trial", "site-success"): 7,
    ("rel-ratebeer", "beer-churn"): 12,
    ("rel-ratebeer", "user-churn"): 12,
    ("rel-ratebeer", "user-count"): 12,
    ("rel-ratebeer", "brewer-dormant"): 9,
    ("rel-stack", "user-engagement"): 18,
    ("rel-stack", "post-votes"): 18,
    ("rel-stack", "user-badge"): 17,
    ("rel-amazon", "user-churn"): 15,
    ("rel-amazon", "user-ltv"): 15,
    ("rel-amazon", "item-churn"): 15,
    ("rel-amazon", "item-ltv"): 16,
    ("rel-hm", "user-churn"): 52,
    ("rel-hm", "item-sales"): 52,
}

# Measured on A100 after the validation fix: GPU-hours to run 7 modes x 5 seeds
# over *all* of that dataset's tasks. Divided by (modes x seeds x episodes) it
# gives the cost of one episode of one trial, which is the unit everything here
# is planned in.
# rel-f1 is MEASURED: the eight driver-position chains totalled 16.57 h over 11
# episodes, which scales to 36.15 h across the dataset's 24 episodes, or 31.6 h
# expressed per MEASURED_MODES=7. With the mode factors below, that reproduces
# every one of the eight measured chains to within 4%.
#
# The other five are still the ORIGINAL ESTIMATES, produced by the same method
# that proved 58% optimistic on rel-f1 (it said 20.0 h). Treat them as lower
# bounds and re-measure each from its first completed chains -- an optimistic
# aggregate here shows up as a chunk that TIMEOUTs at 90% and needs a manual
# resume, which is exactly how the ewc chains were lost.
DATASET_GPU_HOURS: Dict[str, float] = {
    "rel-f1": 31.6,       # measured, 2026-09-07
    "rel-trial": 41.0,
    "rel-stack": 186.0,
    "rel-hm": 389.0,
    "rel-ratebeer": 150.0,
    "rel-amazon": 224.0,
}
MEASURED_MODES = 7
MEASURED_SEEDS = 5

# Order of the roster: the three reference points first, because a CL number
# means nothing without the frame it sits in, then the CL methods cheapest
# first. Within a pair this is also roughly fastest-first, which is what the
# user asked for.
MODE_ORDER: Tuple[str, ...] = (
    "from_scratch", "naive", "joint",
    "freeze_extend", "lwf", "ewc", "er", "der_pp",
)

# A planning heuristic, NOT a measurement -- the measured aggregate averages over
# 7 modes and cannot be split per mode after the fact. Training length is capped
# by --max_training_steps and limit_train_batches, so modes differ mainly in
# per-step overhead: an extra teacher forward (lwf), a penalty term (ewc),
# replay batches on top of the increment (er, der_pp), a frozen backbone with no
# backward through it (freeze_extend), or a full-history sampler (from_scratch,
# joint). The mean over the roster is held near 1.0 so a full-roster estimate
# stays anchored to the measured total (times 8/7 for the extra mode).
# Measured on rel-f1:driver-position -- eight chains, same dataset, same task,
# same 11 episodes, so the ratios are directly comparable. Anchored on
# from_scratch=1.15 (1:40:57 elapsed => 1.463 h per unit factor).
#
# The first guesses here were badly wrong in the direction that costs jobs: ewc
# was modelled at 1.00 and measured at 3.08, so an 11-episode ewc chain needs
# 4.50 h against the 4 h wall and TIMEOUTs one episode from the end. lwf (1.51x)
# and naive (1.50x) were under-modelled too. Chunk sizing divides the wall budget
# by these numbers, so optimism here does not show up as a bad estimate -- it
# shows up as a chain that dies at 90% and has to be resumed by hand.
#
# Rounded UP deliberately: an overestimate only makes chunks smaller, while an
# underestimate loses the whole chunk. The values below are those measured
# ratios divided by their own mean, because these factors only *redistribute*
# a dataset's measured aggregate between modes -- their mean must stay 1.0 or
# DATASET_GPU_HOURS stops being the anchor. The absolute level lives there.
#
# ewc pays a diagonal-Fisher pass over the episode's data after every episode,
# and lwf a teacher forward pass -- both per-episode costs on top of training.
# That naive (increment only) matches joint (full history) says per-episode cost
# is dominated by fixed overhead -- validation, materialisation, prediction --
# rather than by training-set size.
MODE_COST_FACTOR: Dict[str, float] = {
    "from_scratch": 0.800,  # measured 0.80 of the mode mean
    "joint": 0.835,         # measured 0.83 of the mode mean
    "naive": 0.835,         # measured 0.83 of the mode mean
    "er": 0.835,            # measured 0.83 of the mode mean
    "der_pp": 0.835,        # measured 0.83 of the mode mean
    "ewc": 2.157,           # measured 2.16 of the mode mean
    "lwf": 1.113,           # measured 1.11 of the mode mean
    "freeze_extend": 0.591, # measured 0.59 of the mode mean
}


@dataclass(frozen=True)
class Partition:
    r"""An RCI GPU partition and what it costs to plan against.

    Args:
        name: Slurm partition name.
        max_hours: Wall-clock limit, used to size chunks and to reject a
            `--time-limit` the partition would refuse.
        device: GPU model, which sets the default speed factor.
    """

    name: str
    max_hours: float
    device: str


# Limits per the RCI GPU rules. `--time-limit` may go below these, never above.
def resolve_partition(spec: str) -> Partition:
    r"""Resolve one partition name, or a comma-separated list, to a Partition.

    Slurm accepts several partitions for one job and starts it on whichever can
    run it soonest. That is the useful lever when every partition is fully
    allocated: the wait dominates, not the device.

    A list is summarised conservatively -- the SHORTEST wall limit, so a job
    cannot outlive whichever partition takes it, and the SLOWEST device, so the
    cost estimate is an upper bound rather than a hopeful one.

    Args:
        spec: ``"gpufast"`` or ``"amdgpufast,gpufast,gpu"``.

    Returns:
        A Partition whose ``name`` is the original spec (passed through to
        ``sbatch --partition``) and whose limits bound every member.

    Raises:
        ValueError: If any name is unknown.
    """
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        raise ValueError("--partition must name at least one partition")
    unknown = [n for n in names if n not in PARTITIONS]
    if unknown:
        raise ValueError(
            f"unknown partition(s) {unknown}; known: {sorted(PARTITIONS)}"
        )
    members = [PARTITIONS[n] for n in names]
    slowest = max(members, key=lambda p: DEVICE_FACTOR.get(p.device, 1.0))
    return Partition(spec, min(p.max_hours for p in members), slowest.device)


# The A100 nodes (g01-g12) are reachable through FOUR partitions, not one.
# Only amdgpufast was listed here, so every plan was squeezed into a 4 h wall
# that nothing about the hardware requires: amdgpu/amdgpulong/amdgpuextralong
# serve the same nodes with AllowAccounts=ALL and AllowQos=ALL. That omission
# is what forced ewc chains to be chunked to fit 4 h, and when the estimate was
# optimistic they TIMEOUTed at ~90% and needed a manual resume -- five times in
# tier A. It also caps the prediction pass, whose largest rel-stack chain scores
# every checkpoint over the full task table and is estimated at 4-5 h.
#
# amdgpufast stays the default: a shorter wall schedules sooner under backfill,
# and most chunks genuinely fit. The longer partitions are for the work that
# does not -- pass --partition amdgpu rather than chunking around a limit that
# was never real.
PARTITIONS: Dict[str, Partition] = {
    "amdgpufast": Partition("amdgpufast", 4.0, "A100"),
    "amdgpu": Partition("amdgpu", 24.0, "A100"),
    "amdgpulong": Partition("amdgpulong", 72.0, "A100"),
    "amdgpuextralong": Partition("amdgpuextralong", 504.0, "A100"),
    "gpufast": Partition("gpufast", 4.0, "V100"),
    "gpu": Partition("gpu", 24.0, "V100"),
    "gpulong": Partition("gpulong", 72.0, "V100"),
    "gpuextralong": Partition("gpuextralong", 504.0, "V100"),
}

# The cost table was measured on A100. A V100 is roughly this much slower on
# this workload; an estimate, override with --device-factor once a real RCI
# timing exists.
DEVICE_FACTOR: Dict[str, float] = {"A100": 1.0, "V100": 1.8}

# Datasets already in the 17 GB relbench cache on RCI. Anything else has to be
# fetched first, and fetching does not belong on a login node.
# Verified 2026-09-17 against each dataset's materialize-receipt.json on RCI:
# rel-hm 3/3, rel-amazon 3/3, rel-ratebeer 13/13, all ok=true.
CACHED_ON_RCI: Set[str] = {"rel-f1", "rel-amazon", "rel-stack", "rel-trial", "rel-avito",
                           "rel-hm", "rel-ratebeer"}

# Job names carry the lane so a later submission can see which lanes are taken.
JOB_PREFIX = "clg"
_JOB_NAME_RE = re.compile(rf"^{JOB_PREFIX}-L(\d+)-(.+)$")
# Matrix values end up inside a shell command line; keep them boring.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Chunk:
    r"""One Slurm job: `episodes` episodes of one chain, wherever it stands.

    A chunk is a unit of *work*, not a fixed range of episodes. The job resumes
    from whatever increment MLflow reports as the last complete one and runs at
    most `episodes` more, so a chunk that dies half way is not re-planned -- the
    next chunk of the lane simply starts lower down the chain.

    Args:
        index: Position in the chain, used only to name the marker and the job.
        episodes: Value of `--max_increments` for this job.
        gpu_hours: Estimated cost, for the fair-share report.
    """

    dataset: str
    task: str
    mode: str
    index: int
    episodes: int
    gpu_hours: float
    # A sweep arm reruns the same (dataset, task, mode) under a different protocol
    # setting -- most of all a different --val_delta_days. Those are genuinely
    # different chains, but they share a name, so without a discriminator the
    # busy-chain check in `pending_chunks` reads arm 2 as arm 1 already queued and
    # silently skips it. Empty for the main grid, so existing keys are unchanged.
    variant: str = ""

    @property
    def chain_key(self) -> str:
        stem = f"{self.dataset}__{self.task}__{self.mode}"
        return f"{stem}__{self.variant}" if self.variant else stem

    @property
    def chunk_key(self) -> str:
        return f"{self.chain_key}__c{self.index:02d}"

    def job_name(self, lane: int) -> str:
        return f"{JOB_PREFIX}-L{lane}-{self.chunk_key}"


@dataclass
class Config:
    r"""Everything the command builders need that is not part of the matrix."""

    repo: str = DEFAULT_REPO
    out: str = ""
    partition: str = "amdgpufast"
    time_limit_hours: float = 4.0
    time_fill: float = 0.75
    startup_hours: float = 0.25
    device_factor: float = 1.0
    cpus: int = 4
    mem: str = "32G"
    cpus_per_trial: int = 4
    num_samples: int = 5
    seed: int = 42
    mlflow_uri: str = "http://potato.felk.cvut.cz:2222"
    mlflow_experiment_prefix: str = "pelesjak_cl_rci"
    dependency_type: str = "afterany"
    chunk_episodes: Optional[int] = None
    # The Delta-I ablation reruns a pair at a non-native update stride, which changes
    # how many episodes `get_splits` yields. EPISODES is keyed by (dataset, task) and
    # holds the *native* count only, so a sweep arm has to say what it measured or the
    # chain silently stops at the native episode count.
    episodes_override: Optional[int] = None
    # Discriminates one sweep arm from another; see `Chunk.variant`.
    variant: str = ""
    extra: str = ""

    def __post_init__(self) -> None:
        if not self.out:
            self.out = f"{self.repo}/logs/rci/grid"

    @property
    def marker_dir(self) -> str:
        return f"{self.out}/markers"

    @property
    def slurm_log_dir(self) -> str:
        return f"{self.out}/slurm"

    def mlflow_experiment(self, mode: str) -> str:
        return f"{self.mlflow_experiment_prefix}_{mode}"


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

def hours_per_episode_trial(dataset: str) -> float:
    r"""GPU-hours for one episode of one trial of one mode.

    Derived from the measured per-dataset totals by dividing out the 7 modes,
    5 seeds and the episodes of every task of that dataset.

    Args:
        dataset: Dataset name, e.g. ``rel-stack``.

    Returns:
        GPU-hours per (episode x trial x mode) on an A100.

    Raises:
        ValueError: if the dataset has no measured cost or no measured episodes.
    """
    if dataset not in DATASET_GPU_HOURS:
        raise ValueError(
            f"no measured cost for {dataset!r}; known: {sorted(DATASET_GPU_HOURS)}"
        )
    total_episodes = sum(n for (ds, _), n in EPISODES.items() if ds == dataset)
    if total_episodes <= 0:
        raise ValueError(f"no measured episode counts for {dataset!r}")
    return DATASET_GPU_HOURS[dataset] / (MEASURED_MODES * MEASURED_SEEDS * total_episodes)


def episodes_for(dataset: str, task: str) -> int:
    r"""Measured number of episodes for one (dataset, task).

    Raises:
        ValueError: if the pair was never measured. Guessing here is what put
            2.7x-high ceilings into the last plan.
    """
    try:
        return EPISODES[(dataset, task)]
    except KeyError:
        raise ValueError(
            f"no measured episode count for {dataset}:{task}. Measure it with "
            f"ContinuousWrapper.get_splits() before planning compute for it."
        ) from None


def mode_factor(mode: str) -> float:
    r"""Relative cost of a mode.

    Raises:
        ValueError: if the mode is not on the roster.
    """
    if mode not in MODE_COST_FACTOR:
        raise ValueError(f"unknown mode {mode!r}; roster: {list(MODE_ORDER)}")
    return MODE_COST_FACTOR[mode]


def estimate_hours(dataset: str, mode: str, episodes: int, num_samples: int,
                   device_factor: float) -> float:
    r"""GPU-hours for `episodes` episodes of one chain.

    Trials inside an episode are serialised, not parallel: the job holds one
    GPU and every trial asks for one, so wall-clock is the sum over trials and
    GPU-hours equal wall-hours.
    """
    if episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    return (episodes * num_samples * hours_per_episode_trial(dataset)
            * mode_factor(mode) * device_factor)


def episodes_per_chunk(dataset: str, mode: str, cfg: Config) -> int:
    r"""How many episodes fit in one job of the configured partition.

    Sized against a fraction of the wall limit, minus a fixed allowance for
    process start, dataset load and graph materialisation -- overhead that is
    paid once per *job*, so chunking too finely spends fair-share on startup.

    Returns:
        At least 1. One episode may still overrun the limit, which
        `plan_warnings` reports rather than silently accepting.

    Raises:
        ValueError: if the wall budget leaves no room for any training at all.
    """
    if cfg.chunk_episodes is not None:
        if cfg.chunk_episodes < 1:
            raise ValueError(f"--chunk-episodes must be >= 1, got {cfg.chunk_episodes}")
        return cfg.chunk_episodes
    budget = cfg.time_limit_hours * cfg.time_fill - cfg.startup_hours
    if budget <= 0:
        raise ValueError(
            f"time limit {cfg.time_limit_hours} h at fill {cfg.time_fill} leaves "
            f"{budget:.2f} h after {cfg.startup_hours} h of startup; nothing can run"
        )
    per_episode = estimate_hours(dataset, mode, 1, cfg.num_samples, cfg.device_factor)
    return max(1, int(budget / per_episode))


def split_episodes(total: int, per_chunk: int) -> List[int]:
    r"""Cut `total` episodes into chunks of at most `per_chunk`.

    The last chunk carries the remainder rather than a full slice, so the cost
    report does not charge for episodes that do not exist.

    Raises:
        ValueError: on a non-positive chunk size or a negative total.
    """
    if per_chunk < 1:
        raise ValueError(f"per_chunk must be >= 1, got {per_chunk}")
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    full, rest = divmod(total, per_chunk)
    sizes = [per_chunk] * full
    if rest:
        sizes.append(rest)
    return sizes


# --------------------------------------------------------------------------
# The job matrix
# --------------------------------------------------------------------------

def order_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    r"""Sort (dataset, task) pairs cheapest first.

    Datasets by their measured total GPU-hours, then tasks by episode count:
    training length is step-capped, so the episode count dominates a chain's
    cost far more than its rows per episode do. Unmeasured pairs sort last, in
    the order given, so an exploratory pair never displaces a planned one.
    """
    ordered = list(dict.fromkeys(pairs))

    def key(pair: Tuple[str, str]) -> Tuple[int, float, int, int]:
        dataset, task = pair
        known = (dataset in DATASET_GPU_HOURS) and ((dataset, task) in EPISODES)
        if not known:
            return (1, 0.0, 0, ordered.index(pair))
        return (0, DATASET_GPU_HOURS[dataset], EPISODES[(dataset, task)],
                ordered.index(pair))

    return sorted(ordered, key=key)


def order_modes(modes: Iterable[str]) -> List[str]:
    r"""Sort modes into roster order: reference frame first, then cheapest.

    Raises:
        ValueError: if a mode is not on the roster. `ft_upsample` is a
            reproduction-only alias and is deliberately not launchable here.
    """
    modes = list(modes)
    unknown = [m for m in modes if m not in MODE_ORDER]
    if unknown:
        raise ValueError(f"unknown mode(s) {unknown}; roster: {list(MODE_ORDER)}")
    return sorted(dict.fromkeys(modes), key=MODE_ORDER.index)


def plan_chain(dataset: str, task: str, mode: str, cfg: Config) -> List[Chunk]:
    r"""Chunks for one (dataset, task, mode) chain, in execution order."""
    for value in (dataset, task, mode):
        if not _SAFE_NAME_RE.match(value):
            raise ValueError(f"unsafe name for a shell command line: {value!r}")
    if cfg.variant and not _SAFE_NAME_RE.match(cfg.variant):
        raise ValueError(f"unsafe name for a shell command line: {cfg.variant!r}")
    episodes = (cfg.episodes_override
                if cfg.episodes_override is not None
                else episodes_for(dataset, task))
    sizes = split_episodes(episodes, episodes_per_chunk(dataset, mode, cfg))
    return [
        Chunk(dataset, task, mode, i, size,
              estimate_hours(dataset, mode, size, cfg.num_samples, cfg.device_factor),
              variant=cfg.variant)
        for i, size in enumerate(sizes)
    ]


def build_matrix(pairs: Iterable[Tuple[str, str]], modes: Iterable[str],
                 cfg: Config) -> List[List[Chunk]]:
    r"""The full plan as a list of chains, task-major and priority-ordered.

    Returns:
        One list of chunks per chain. Chains appear pair by pair -- all modes of
        one pair before any mode of the next -- so stopping half way leaves
        whole, comparable cells rather than a mode missing everywhere.
    """
    ordered_modes = order_modes(modes)
    return [
        plan_chain(dataset, task, mode, cfg)
        for dataset, task in order_pairs(pairs)
        for mode in ordered_modes
    ]


def plan_warnings(chains: Sequence[Sequence[Chunk]], cfg: Config) -> List[str]:
    r"""Things that will bite later, collected before anything is submitted."""
    warnings: List[str] = []
    usable = cfg.time_limit_hours * cfg.time_fill - cfg.startup_hours
    for chain in chains:
        if not chain:
            continue
        head = chain[0]
        one_episode = estimate_hours(head.dataset, head.mode, 1, cfg.num_samples,
                                     cfg.device_factor)
        if one_episode > usable:
            warnings.append(
                f"{head.chain_key}: one episode is ~{one_episode:.1f} h but a job "
                f"has ~{usable:.1f} h of usable wall time. The chain cannot "
                f"advance -- use a longer partition or fewer seeds."
            )
        # Separate from the check above because --chunk-episodes bypasses the
        # sizing arithmetic entirely: episodes can each fit while the chunk does not.
        for chunk in chain:
            job_hours = chunk.gpu_hours + cfg.startup_hours
            if job_hours > cfg.time_limit_hours:
                warnings.append(
                    f"{chunk.chunk_key}: ~{job_hours:.1f} h estimated against a "
                    f"{cfg.time_limit_hours:g} h limit. Slurm will kill it and the "
                    f"next chunk will redo the unfinished episode."
                )
                break  # one line per chain is enough to make the point
    missing = sorted({c[0].dataset for c in chains if c} - CACHED_ON_RCI)
    if missing:
        warnings.append(
            f"not in the RCI relbench cache: {', '.join(missing)}. The first job "
            f"to touch one will try to download it on a compute node. Prefetch it "
            f"from a batch job first -- never on the login node."
        )
    return warnings


# --------------------------------------------------------------------------
# Cluster state: what is already done, what is already queued
# --------------------------------------------------------------------------

@dataclass
class ClusterState:
    r"""What the cluster and the marker directory say about work in flight."""

    done_chunks: Set[str] = field(default_factory=set)
    complete_chains: Set[str] = field(default_factory=set)
    busy_chains: Set[str] = field(default_factory=set)
    busy_lanes: Set[int] = field(default_factory=set)
    known: bool = True


def parse_markers(listing: str) -> Tuple[Set[str], Set[str]]:
    r"""Split an ``ls -1`` of the marker directory into done chunks and chains.

    Returns:
        ``(done_chunk_keys, complete_chain_keys)``.
    """
    done: Set[str] = set()
    complete: Set[str] = set()
    for raw in listing.splitlines():
        name = raw.strip()
        if name.endswith(".done"):
            done.add(name[: -len(".done")])
        elif name.endswith(".chain-complete"):
            complete.add(name[: -len(".chain-complete")])
    return done, complete


def parse_queue(listing: str) -> Tuple[Set[str], Set[int]]:
    r"""Read our own job names out of `squeue` output.

    Returns:
        ``(busy_chain_keys, busy_lane_ids)``. Chains are tracked, not chunks,
        because two chunks of one chain must never run at once -- they would
        both resume from the same increment. Lanes are tracked so a second
        submission fills only lanes nobody is using, which is what keeps the
        4-GPU cap true across submissions.
    """
    chains: Set[str] = set()
    lanes: Set[int] = set()
    for raw in listing.splitlines():
        match = _JOB_NAME_RE.match(raw.strip())
        if not match:
            continue
        lanes.add(int(match.group(1)))
        chains.add(match.group(2).rsplit("__c", 1)[0])
    return chains, lanes


def pending_chunks(chain: Sequence[Chunk], state: ClusterState) -> List[Chunk]:
    r"""The chunks of one chain that still need submitting.

    A chain with anything in the queue is skipped whole: submitting another of
    its chunks now would put two jobs of the same chain in flight in different
    lanes, and both would resume from the same increment.
    """
    if not chain:
        return []
    if chain[0].chain_key in state.complete_chains:
        return []
    if chain[0].chain_key in state.busy_chains:
        return []
    return [c for c in chain if c.chunk_key not in state.done_chunks]


def assign_lanes(chains: Sequence[Sequence[Chunk]],
                 lanes: Sequence[int]) -> Dict[int, List[Chunk]]:
    r"""Deal whole chains into lanes, keeping each chain serial and in order.

    Greedy least-loaded by estimated GPU-hours, considering the chains
    **largest-first** (LPT). Every chunk of a chain lands in one lane, in index
    order, so a lane is a strictly serial dependency chain and no chain ever has
    two jobs in flight.

    Largest-first matters, and walking in priority order instead was a real bug.
    A chain is indivisible across lanes -- its chunks resume one another -- so
    this is bin-packing, where greedy-in-arbitrary-order has no useful bound but
    greedy-largest-first is within 4/3 of optimal. On rel-stack the priority walk
    placed the cheap from_scratch chain first and then dropped the 18.4 h ewc
    chain on the same lane, for a 25.2 h lane beside three 14.2 h lanes: an
    11-hour tail on one GPU that no dependency required.

    Assignment order and submission order are separate concerns, so the priority
    order is restored *within* each lane afterwards: lanes balance by size, while
    each lane still starts with its cheapest reference modes, which is what makes
    partial results useful early.

    Args:
        lanes: Lane ids to fill. At most `MAX_CONCURRENT_GPUS` of them.

    Returns:
        Lane id -> the chunks to submit in that lane, in order.

    Raises:
        ValueError: on an empty lane list or more lanes than the GPU cap.
    """
    if not lanes:
        raise ValueError("no lanes available to submit into")
    if len(set(lanes)) > MAX_CONCURRENT_GPUS:
        raise ValueError(
            f"{len(set(lanes))} lanes would run {len(set(lanes))} GPUs at once; "
            f"the cap is {MAX_CONCURRENT_GPUS}"
        )
    indexed = [(i, chain) for i, chain in enumerate(chains) if chain]
    load: Dict[int, float] = {lane: 0.0 for lane in lanes}
    assigned: Dict[int, List[Tuple[int, Sequence[Chunk]]]] = {lane: [] for lane in lanes}

    # LPT: heaviest chain first, each onto the lane that is currently lightest.
    # Ties break on the original index so the assignment stays deterministic.
    for i, chain in sorted(
        indexed, key=lambda pair: (-sum(c.gpu_hours for c in pair[1]), pair[0])
    ):
        lane = min(lanes, key=lambda k: (load[k], k))
        assigned[lane].append((i, chain))
        load[lane] += sum(c.gpu_hours for c in chain)

    # Restore priority order inside each lane -- balance is about which lane,
    # not about what that lane runs first.
    plan: Dict[int, List[Chunk]] = {lane: [] for lane in lanes}
    for lane, entries in assigned.items():
        for _, chain in sorted(entries, key=lambda pair: pair[0]):
            plan[lane].extend(chain)
    return plan


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def format_walltime(hours: float) -> str:
    r"""Slurm ``HH:MM:SS``. Rounded up to the minute, never zero."""
    if hours <= 0:
        raise ValueError(f"walltime must be positive, got {hours}")
    minutes = max(1, math.ceil(hours * 60))
    return f"{minutes // 60}:{minutes % 60:02d}:00"


def build_sbatch_command(chunk: Chunk, lane: int, cfg: Config,
                         dependency: Optional[str]) -> str:
    r"""The `sbatch` command line for one chunk.

    Args:
        dependency: Slurm job id of the previous job in this lane, or None for
            the first. Passed as ``afterany`` by default: a chunk that fails or
            hits the wall limit leaves MLflow consistent, and the next chunk
            resumes from the last complete increment, so the failure is retried
            rather than stalling the lane behind an unsatisfiable dependency.

    Returns:
        A single shell command string, to be run by a local or remote shell.
    """
    parts = [
        "sbatch", "--parsable",
        f"--job-name={chunk.job_name(lane)}",
        f"--partition={cfg.partition}",
        f"--time={format_walltime(cfg.time_limit_hours)}",
        "--gres=gpu:1",
        f"--cpus-per-task={cfg.cpus}",
        f"--mem={cfg.mem}",
        f"--output={cfg.slurm_log_dir}/{chunk.chunk_key}-%j.out",
    ]
    if dependency:
        parts.append(f"--dependency={cfg.dependency_type}:{dependency}")
    # run_chain.sh parses `--flag value`, not `--flag=value`; keep the two in step.
    parts += [
        f"{cfg.repo}/slurm/rci/run_chain.sh",
        "--dataset", chunk.dataset,
        "--task", chunk.task,
        "--mode", chunk.mode,
        "--chunk", str(chunk.index),
        "--episodes", str(chunk.episodes),
        "--num-samples", str(cfg.num_samples),
        "--seed", str(cfg.seed),
        "--out", cfg.out,
        "--mlflow-uri", cfg.mlflow_uri,
        "--mlflow-experiment", cfg.mlflow_experiment(chunk.mode),
        "--cpus-per-trial", str(cfg.cpus_per_trial),
    ]
    if cfg.extra:
        parts += ["--extra", shlex.quote(cfg.extra)]
    return " ".join(parts)


def build_mkdir_command(cfg: Config, chains: Sequence[Sequence[Chunk]]) -> str:
    r"""One `mkdir -p` for every directory Slurm needs before a job starts."""
    dirs = [cfg.marker_dir, cfg.slurm_log_dir]
    dirs += [f"{cfg.out}/{chain[0].chain_key}" for chain in chains if chain]
    return "mkdir -p " + " ".join(dirs)


def wrap_remote(command: str, ssh_host: Optional[str]) -> List[str]:
    r"""Turn a shell command string into an argv, local or over ssh.

    Both forms end up in a shell, so ``$HOME`` in a path expands on the machine
    that will actually use it.
    """
    if ssh_host:
        return ["ssh", ssh_host, command]
    return ["bash", "-c", command]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_matrix(plan: Dict[int, List[Chunk]]) -> str:
    r"""The job matrix, lane by lane, in submission order."""
    lines = []
    for lane in sorted(plan):
        chunks = plan[lane]
        hours = sum(c.gpu_hours for c in chunks)
        lines.append(f"lane {lane}: {len(chunks)} job(s), ~{hours:.1f} GPU-h "
                     f"(serial, 1 GPU)")
        for position, chunk in enumerate(chunks):
            dep = "-" if position == 0 else f"after {plan[lane][position - 1].chunk_key}"
            lines.append(f"    {chunk.chunk_key:<52} {chunk.episodes:>3} ep "
                         f"{chunk.gpu_hours:>6.2f} h  {dep}")
    return "\n".join(lines)


def total_hours(chains: Sequence[Sequence[Chunk]],
                cfg: Config) -> Tuple[float, float, float]:
    r"""The full fair-share bill of a plan.

    Returns:
        ``(training, startup, total)`` GPU-hours. Startup is charged per *job*,
        not per chain: every chunk pays for process start, dataset load and
        graph load again, so cutting a chain finer buys shorter jobs at a real
        price. Leaving it out of the bill would make fine chunking look free.
    """
    jobs = sum(len(chain) for chain in chains)
    training = sum(c.gpu_hours for chain in chains for c in chain)
    startup = jobs * cfg.startup_hours
    return training, startup, training + startup


def format_cost(chains: Sequence[Sequence[Chunk]], cfg: Config,
                heading: str) -> str:
    r"""The fair-share bill, per pair and in total.

    Printed before anything is submitted: nobody should discover what a command
    costs by watching the budget drain.
    """
    per_pair: Dict[Tuple[str, str], List[Chunk]] = {}
    for chain in chains:
        for chunk in chain:
            per_pair.setdefault((chunk.dataset, chunk.task), []).append(chunk)

    lines = [heading,
             f"{'dataset:task':<34}{'modes':>6}{'jobs':>6}{'episodes':>10}{'GPU-h':>10}"]
    for (dataset, task), chunks in per_pair.items():
        hours = sum(c.gpu_hours for c in chunks)
        lines.append(f"{dataset + ':' + task:<34}"
                     f"{len({c.mode for c in chunks}):>6}"
                     f"{len(chunks):>6}"
                     f"{sum(c.episodes for c in chunks):>10}"
                     f"{hours:>10.1f}")
    training, startup, grand = total_hours(chains, cfg)
    jobs = sum(len(chain) for chain in chains)
    lines.append(f"{'training':<34}{'':>6}{jobs:>6}"
                 f"{sum(c.episodes for chain in chains for c in chain):>10}"
                 f"{training:>10.1f}")
    lines.append(f"{'startup (' + str(jobs) + ' jobs x ' + format(cfg.startup_hours, 'g') + ' h)':<34}"
                 f"{'':>6}{'':>6}{'':>10}{startup:>10.1f}")
    lines.append(f"{'TOTAL':<34}{'':>6}{'':>6}{'':>10}{grand:>10.1f}")
    return "\n".join(lines)


def format_provenance(cfg: Config, lanes: int) -> str:
    r"""Where the numbers come from, so the estimate can be argued with."""
    return "\n".join([
        f"basis   : measured A100 GPU-h per dataset (7 modes x 5 seeds), divided "
        f"by modes x seeds x episodes",
        f"scaling : {cfg.num_samples} seed(s), device factor {cfg.device_factor:g} "
        f"({resolve_partition(cfg.partition).device} on {cfg.partition}), per-mode "
        f"factors are a heuristic, not a measurement",
        f"lanes   : {lanes} of {MAX_CONCURRENT_GPUS} GPUs, so wall-clock is "
        f"roughly GPU-h / {lanes} plus queueing",
    ])


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tier", nargs="+", choices=sorted(TIERS),
                     help="Dataset tiers to run, cheapest first.")
    src.add_argument("--pairs", nargs="+", metavar="DATASET:TASK",
                     help="Explicit dataset:task pairs.")
    p.add_argument("--modes", nargs="+", default=list(MODE_ORDER))
    p.add_argument("--lanes", type=int, default=MAX_CONCURRENT_GPUS,
                   help=f"Concurrent GPUs. Hard cap {MAX_CONCURRENT_GPUS}.")
    p.add_argument("--partition", default="amdgpufast",
                   help="One partition, or a comma-separated list. Slurm places a "
                        "job on whichever listed partition can run it soonest, "
                        "which matters when every partition is allocated and the "
                        "wait -- not the device -- is what costs wall-clock. The "
                        "wall limit and device factor are taken from the most "
                        "conservative member of the list. amdgpufast is A100 (the "
                        "device the cost model was measured on) and 4 h.")
    p.add_argument("--time-limit", type=float, default=None,
                   help="Wall limit per job in hours. Default: the partition's.")
    p.add_argument("--time-fill", type=float, default=0.75,
                   help="Fraction of the wall limit a chunk is sized to use.")
    p.add_argument("--startup-hours", type=float, default=0.25,
                   help="Per-job allowance for dataset load and materialisation.")
    p.add_argument("--device-factor", type=float, default=None,
                   help="Slowdown vs the A100 the costs were measured on. "
                        "Default: by the partition's GPU.")
    p.add_argument("--chunk-episodes", type=int, default=None,
                   help="Override the computed episodes per job.")
    p.add_argument("--variant", default="",
                   help="Tag distinguishing one sweep arm from another. Sweep arms "
                        "rerun the same (dataset, task, mode) under a different "
                        "protocol setting, so without this the busy-chain check "
                        "reads a later arm as an earlier one already queued.")
    p.add_argument("--episodes-override", type=int, default=None,
                   help="Override the measured episode count for the chain. Needed "
                        "when --extra changes --val_delta_days: EPISODES holds the "
                        "native stride's count, so without this the chain stops "
                        "early. Measure with ContinuousWrapper.get_splits(val_delta=...).")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Seeds per episode. Must not change mid-chain: it is the "
                        "resume quorum (run_chain.sh refuses if it does).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpus", type=int, default=4, help="CPUs per job.")
    p.add_argument("--mem", default="32G", help="Memory per job.")
    p.add_argument("--cpus-per-trial", type=int, default=4)
    p.add_argument("--repo", default=DEFAULT_REPO, help="Repo path on RCI.")
    p.add_argument("--out", default=None, help="Root for logs, models and markers. "
                                               "Default: <repo>/logs/rci/grid.")
    p.add_argument("--mlflow-uri", default="http://potato.felk.cvut.cz:2222")
    p.add_argument("--mlflow-experiment-prefix", default="pelesjak_cl_rci",
                   help="Experiment is <prefix>_<mode>. Keep it off the published "
                        "grid's names: those runs' checkpoints are on another "
                        "filesystem, and only chain_id keeps them out of a resume.")
    p.add_argument("--dependency-type", default="afterany",
                   choices=["afterany", "afterok"],
                   help="afterany retries a failed chunk via --resume; afterok "
                        "stops the lane at the first failure.")
    p.add_argument("--extra", default="",
                   help="Verbatim extra flags for continuous_learning.py.")
    p.add_argument("--ssh", default=None, metavar="HOST",
                   help="Run sbatch/squeue over ssh instead of locally.")
    p.add_argument("--ignore-state", action="store_true",
                   help="Do not consult markers or squeue. Re-submits everything.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the matrix, the cost and the sbatch commands.")
    p.add_argument("--yes", action="store_true",
                   help="Submit without the interactive confirmation.")
    return p.parse_args(argv)


def resolve_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    r"""Pairs from `--tier` or `--pairs`.

    Raises:
        SystemExit: on a malformed `--pairs` entry.
    """
    if args.pairs:
        out = []
        for raw in args.pairs:
            if ":" not in raw:
                raise SystemExit(f"--pairs entry {raw!r} must look like dataset:task")
            dataset, task = raw.split(":", 1)
            out.append((dataset, task))
        return out
    return [pair for tier in args.tier for pair in TIERS[tier]]


def config_from_args(args: argparse.Namespace) -> Config:
    r"""Build the Config, resolving the partition-dependent defaults.

    Raises:
        SystemExit: if the wall limit exceeds what the partition allows, or the
            lane count exceeds the GPU cap.
    """
    partition = resolve_partition(args.partition)
    time_limit = args.time_limit if args.time_limit is not None else partition.max_hours
    if time_limit > partition.max_hours:
        raise SystemExit(
            f"--time-limit {time_limit} h exceeds {partition.name}'s "
            f"{partition.max_hours} h; the job would be rejected at submission."
        )
    if not 1 <= args.lanes <= MAX_CONCURRENT_GPUS:
        raise SystemExit(
            f"--lanes must be between 1 and {MAX_CONCURRENT_GPUS} (the RCI cap)"
        )
    return Config(
        repo=args.repo,
        out=args.out or f"{args.repo}/logs/rci/grid",
        partition=args.partition,
        time_limit_hours=time_limit,
        time_fill=args.time_fill,
        startup_hours=args.startup_hours,
        device_factor=(args.device_factor if args.device_factor is not None
                       else DEVICE_FACTOR[partition.device]),
        cpus=args.cpus,
        mem=args.mem,
        cpus_per_trial=args.cpus_per_trial,
        num_samples=args.num_samples,
        seed=args.seed,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment_prefix=args.mlflow_experiment_prefix,
        dependency_type=args.dependency_type,
        chunk_episodes=args.chunk_episodes,
        episodes_override=args.episodes_override,
        variant=args.variant,
        extra=args.extra,
    )


def run_command(command: str, ssh_host: Optional[str]) -> subprocess.CompletedProcess:
    r"""Run one shell command locally or on the cluster."""
    return subprocess.run(wrap_remote(command, ssh_host), capture_output=True,
                          text=True, check=False)


def read_state(cfg: Config, ssh_host: Optional[str]) -> ClusterState:
    r"""Ask the cluster what is already done and what is already queued.

    A missing marker directory is normal on a first run. An unreachable cluster
    is not fatal either -- the state is marked unknown and reported, so a
    dry-run works on a laptop with no Slurm and no ssh.
    """
    state = ClusterState()
    # A missing marker directory is the normal first run, so that one is
    # tolerated; a missing `squeue` is not -- an empty queue and no scheduler
    # look identical, and only one of them means "no lane is busy".
    markers = run_command(f"ls -1 {cfg.marker_dir} 2>/dev/null || true", ssh_host)
    queue = run_command("squeue --me --noheader --format=%j", ssh_host)
    if markers.returncode != 0 or queue.returncode != 0:
        state.known = False
        return state
    state.done_chunks, state.complete_chains = parse_markers(markers.stdout)
    state.busy_chains, state.busy_lanes = parse_queue(queue.stdout)
    return state


def submit(plan: Dict[int, List[Chunk]], cfg: Config,
           ssh_host: Optional[str]) -> Tuple[List[str], List[str]]:
    r"""Submit each lane as a serial dependency chain.

    Returns:
        ``(submitted_job_ids, failures)``. A lane is abandoned at its first
        failed submission: the rest of that lane would otherwise either run
        unchained -- breaking the one-job-per-chain rule -- or hang on a
        dependency that never existed.
    """
    submitted: List[str] = []
    failures: List[str] = []
    for lane in sorted(plan):
        previous: Optional[str] = None
        for chunk in plan[lane]:
            command = build_sbatch_command(chunk, lane, cfg, previous)
            result = run_command(command, ssh_host)
            if result.returncode != 0:
                failures.append(f"lane {lane} stopped at {chunk.chunk_key}: "
                                f"{result.stderr.strip() or result.stdout.strip()}")
                break
            job_id = result.stdout.strip().splitlines()[-1].split(";")[0]
            submitted.append(job_id)
            print(f"  [{job_id}] lane {lane} {chunk.chunk_key}"
                  + (f" after {previous}" if previous else ""))
            previous = job_id
    return submitted, failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    pairs = resolve_pairs(args)

    try:
        chains = build_matrix(pairs, args.modes, cfg)
    except ValueError as exc:
        raise SystemExit(str(exc))

    state = ClusterState() if args.ignore_state else read_state(cfg, args.ssh)
    if not state.known:
        # Lanes are how this script *schedules* against the 4-GPU limit, but they
        # are not what *enforces* it: the Slurm account is capped at 4 concurrent
        # GPUs, so anything extra queues rather than over-running. So an
        # unreadable squeue is a scheduling inefficiency -- work may be dealt into
        # a lane that is actually busy, and simply wait -- not a quota breach.
        # Warn and continue rather than refusing.
        print(
            "! could not read squeue: occupied lanes are unknown, so work may be "
            "dealt into a busy lane and queue behind it. Slurm still caps the "
            "account at 4 concurrent GPUs, so this costs latency, not quota."
        )

    remaining = [pending_chunks(chain, state) for chain in chains]
    todo = [chain for chain in remaining if chain]
    skipped = sum(len(a) - len(b) for a, b in zip(chains, remaining))

    if skipped:
        print(format_cost(chains, cfg, "PLANNED (the whole selection)"))
        print(f"\n{skipped} chunk(s) already done, complete or queued -- skipping.\n")
    if not todo:
        print("Nothing left to submit.")
        return 0

    # Use every lane, including ones with work already in flight. Slurm enforces
    # the concurrency limit itself, so extra jobs queue rather than over-running,
    # and queueing is fine. Skipping busy lanes was strictly worse: with three of
    # four lanes occupied, the whole remainder was funnelled into the single free
    # lane and ran serially on one GPU.
    lanes = list(range(args.lanes))
    if state.busy_lanes:
        print(f"lanes {sorted(state.busy_lanes)} already have work in flight; "
              f"new jobs queue behind it.")

    plan = assign_lanes(todo, lanes)
    print(format_cost(todo, cfg, "TO SUBMIT NOW"))
    print()
    print(format_provenance(cfg, len(lanes)))
    print()
    print(format_matrix(plan))
    for warning in plan_warnings(todo, cfg):
        print(f"! {warning}")

    _, _, grand = total_hours(todo, cfg)
    n_jobs = sum(len(chain) for chain in todo)
    print(f"\n==> {n_jobs} job(s), ~{grand:.1f} GPU-h of fair-share, "
          f"~{grand / len(lanes):.1f} h wall on {len(lanes)} GPU(s) "
          f"(plus queueing).")

    if args.dry_run:
        where = f"on {args.ssh} over ssh" if args.ssh else "in a local shell"
        print(f"\n-- dry run: commands that would be executed {where} --")
        print("  " + build_mkdir_command(cfg, todo))
        for lane in sorted(plan):
            previous = None
            for chunk in plan[lane]:
                print("  " + build_sbatch_command(chunk, lane, cfg, previous))
                previous = f"<jobid:{chunk.job_name(lane)}>"
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("refusing to spend fair-share non-interactively "
                             "without --yes")
        if input("Submit? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing submitted.")
            return 0

    mkdir = run_command(build_mkdir_command(cfg, todo), args.ssh)
    if mkdir.returncode != 0:
        raise SystemExit(f"could not create output directories: {mkdir.stderr.strip()}")

    submitted, failures = submit(plan, cfg, args.ssh)
    print(f"\nSubmitted {len(submitted)} job(s).")
    for failure in failures:
        print(f"! {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
