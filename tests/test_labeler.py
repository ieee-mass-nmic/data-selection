"""HiFidelityLabeler orchestration + RankNorm aggregation (design §10.3).

No model: a fake updater returns deterministic deltas so we can assert the
rank-normalization and horizon/anchor aggregation are correct.
"""

from __future__ import annotations

from pathlib import Path

from pcu_select.hi_fidelity.anchors import AnchorRegistry, AnchorSpec
from pcu_select.hi_fidelity.labeler import HiFidelityLabeler, LabelerConfig
from pcu_select.hi_fidelity.sampler import TripleSample
from pcu_select.types import PEFTConfig, PEFTRecipe, Sample, ValidationSketch


class _FakeUpdater:
    """delta depends only on sample_id → deterministic, rank-able."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def delta(self, *, peft, sample, sketch, horizon, seed):
        base = float(int(sample.sample_id[1:]))  # "s0","s1",... → 0,1,...
        return self.scale * base * (1.0 + 0.1 * horizon)


def _make_labeler(n_samples=4, anchors=2, scale_per_anchor=(1.0, 2.0)):
    reg = AnchorRegistry(Path("/tmp/pcu_anchors_test"))
    for i in range(anchors):
        reg.register(AnchorSpec(anchor_id=f"a{i}", checkpoint_path=Path("/dev/null")))

    samples = {f"s{i}": Sample(sample_id=f"s{i}", instruction="i", response="r") for i in range(n_samples)}
    peft = PEFTConfig(peft_id="p0", family="lora", target_modules=["q_proj"],
                      target_layers=[0], rank=4, recipe=PEFTRecipe())
    sketch = ValidationSketch(task_id="t0", samples=[samples["s0"]], sketch_seed=0)

    scales = iter(scale_per_anchor)

    def factory(spec):
        return _FakeUpdater(scale=next(scales))

    labeler = HiFidelityLabeler(
        anchors=reg, samples_by_id=samples, pefts_by_id={"p0": peft},
        sketches_by_id={"t0": sketch}, updater_factory=factory,
        cfg=LabelerConfig(horizons=(1, 4), horizon_weights=(0.4, 0.6), seed=0),
    )
    triples = [TripleSample(sample_id=f"s{i}", peft_id="p0", task_id="t0", phase=1)
               for i in range(n_samples)]
    return labeler, triples, n_samples


def test_labeler_produces_one_label_per_triple():
    labeler, triples, n = _make_labeler()
    labels = labeler.run(triples)
    assert len(labels) == n
    keys = {(label.sample_id, label.peft_id, label.task_id) for label in labels}
    assert len(keys) == n


def test_rank_norm_maps_to_unit_interval_and_orders():
    labeler, triples, _ = _make_labeler()
    labels = labeler.run(triples)
    by_sid = {label.sample_id: label for label in labels}
    # u_hi is rank-normalized to [0,1] within bucket; monotone in raw delta.
    for label in labels:
        assert 0.0 - 1e-9 <= label.u_hi <= 1.0 + 1e-9
    # s0 has the smallest delta (0) → lowest rank; s3 the largest → highest.
    assert by_sid["s0"].u_hi < by_sid["s3"].u_hi


def test_sigma_est_zero_when_anchors_agree_in_rank():
    # Both anchors scale deltas linearly → identical RankNorm per horizon →
    # zero std across anchors within each horizon bucket.
    labeler, triples, _ = _make_labeler(scale_per_anchor=(1.0, 5.0))
    labels = labeler.run(triples)
    for label in labels:
        assert label.sigma_est == 0.0


def test_empty_triples_returns_empty():
    labeler, _, _ = _make_labeler()
    assert labeler.run([]) == []
