r"""The two training-budget regimes must both stay expressible.

The grid runs a *step* budget: `limit_train_batches=100` with
`max_training_steps=2000`, so every episode sees 256,000 samples regardless of
how much history it has. That equalises cost, but it also starves the
full-history regimes -- on rel-stack/user-badge `naive` gets 1.33 passes over
its own training set while `joint` and `from_scratch` get 0.07, a 19x gap that
grows as ~1/i along the very episode axis the paper studies.

The control for that confound is an *epoch* budget: a real pass over the
episode's own data, so later episodes train longer. These tests pin the defaults
(so the grid's protocol cannot drift underneath the 3,090 runs already in
MLflow) and pin that the epoch regime remains reachable.
"""

import inspect

from experiments.continuous_learning.continuous_learning import run_ray_tuner

SIG = inspect.signature(run_ray_tuner)


def test_step_budget_defaults_are_unchanged():
    """The already-collected grid used these values; drift would silently
    re-define the protocol for every future comparison against it."""
    assert SIG.parameters["max_training_steps"].default == 2000
    assert SIG.parameters["limit_train_batches"].default == 100
    assert SIG.parameters["val_check_interval"].default == 100


def test_epoch_budget_is_expressible():
    """`max_epochs` must exist and default to off, so the step budget stays the
    default protocol and the epoch arm is opt-in."""
    assert "max_epochs" in SIG.parameters
    assert SIG.parameters["max_epochs"].default is None


def test_both_budget_knobs_reach_the_trial_config():
    """They are useless unless they are forwarded into `param_space`: the
    trainer reads them out of `config`, not out of the tuner's frame."""
    source = inspect.getsource(run_ray_tuner)
    assert '"limit_train_batches": limit_train_batches' in source
    assert '"max_epochs": max_epochs' in source
