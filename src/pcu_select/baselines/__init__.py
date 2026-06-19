"""Baseline data-selection methods (design §2.3 / §1.6).

Every baseline produces a list of selected `SampleID`s given the *same* cached
features and task query that PCU-Select consumes, so the comparison is
apples-to-apples (the only variable is the selection rule). PCU-Select itself is
not here — it is `pcu_select.pipeline.apply.run_apply`.
"""

from pcu_select.baselines.selectors import (
    BASELINES,
    BaselineInputs,
    score_baseline,
    select_baseline,
)

__all__ = ["BASELINES", "BaselineInputs", "score_baseline", "select_baseline"]
