"""Shared constants/helpers for the motivation experiments (Table 1, F1, F2).

These precede E1–E5: they establish the *premise* of the project (data value is
PEFT-dependent) with signals that are deliberately **independent of PCU's own
scorer** — real short-update Δ (truth) and LESS-style per-PEFT influence — so the
argument is not circular. See docs/pcu_select_motivation_design.md.

The PEFT subsets are drawn from `experiments.registry.PEFT_REGISTRY`. F1 spans
three structural axes (family/operator, capacity, placement) so the figure can
show that disagreement *tracks structural distance*, not just family names. F2
uses a smaller spanning subset to keep the P×P training matrix affordable.
"""

from __future__ import annotations

# Figure-1 / Table-1 PEFT set (8): spans family, capacity and placement axes.
#   family/operator : L-r8-qv (add. low-rank) · IA3-attnmlp (mult.) · AD-b64 (bottleneck)
#   placement       : L-r8-qv (attn) ↔ L-r8-mlp · L-r8-lowlayers ↔ L-r8-highlayers
#   capacity        : L-r4-qv ↔ L-r8-qv ↔ L-r32-qkvo
F1_PEFTS: list[str] = [
    "L-r8-qv", "L-r8-mlp", "L-r4-qv", "L-r32-qkvo",
    "L-r8-lowlayers", "L-r8-highlayers", "IA3-attnmlp", "AD-b64",
]
TABLE1_PEFTS: list[str] = list(F1_PEFTS)

# Figure-2 transfer-matrix PEFT set (5): a spanning subset cheap enough for a
# P×P × tasks × seeds target-training sweep. All native-backend trainable.
F2_PEFTS: list[str] = ["L-r8-qv", "L-r32-qkvo", "L-r8-mlp", "IA3-attnmlp", "AD-b64"]

# Approximate backbone parameter counts, for Table 1's "% of backbone" column
# without loading the model. Pass --from-model to build_table1.py for exact counts.
APPROX_BACKBONE_PARAMS: dict[str, float] = {
    "llama2-7b": 6.738e9,
    "llama2-13b": 13.02e9,
    "qwen2.5-7b": 7.62e9,
    "qwen2.5-14b": 14.77e9,
}


def structural_bucket(spec_a, spec_b) -> str:
    """Classify a PEFT pair by structural distance (design §3.5).

    `spec_a`/`spec_b` are `experiments.registry.PeftSpec`. Returns one of
    "same-fam-capacity" / "same-fam-placement" / "cross-family", used by the F1
    plot to show that ranking disagreement grows with structural distance.
    """
    if spec_a.family != spec_b.family:
        return "cross-family"
    same_place = (spec_a.modules_key == spec_b.modules_key
                  and spec_a.layer_range == spec_b.layer_range)
    return "same-fam-capacity" if same_place else "same-fam-placement"
