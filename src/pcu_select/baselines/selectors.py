"""Concrete baseline selectors over the cached feature store.

All selectors share the `BaselineInputs` bundle (cached features + an optional
task query) and return the chosen `SampleID`s. They are deliberately cheap: the
expensive signals (forward stats, gradient signatures) were already computed
once in the offline feature stage, so a baseline is just a ranking rule on top.

Difficulty-vector layout (design §7.2, see `features/difficulty.py`):
    0 log_len_instr   1 log_len_resp   2 log_total_len   3 resp/instr_ratio
    4 loss_mean       5 loss_std       6 loss_max        7 perplexity
    8 avg_logprob     9 entropy_mean  10 entropy_max    11 is_cot
   12 is_code        13 is_qa         14,15 language one-hot

Methods that require task-specific signals beyond the core cache read them from
`features/baseline_scores.parquet`, aligned by `sample_id`. If a required signal
is absent, the selector fails instead of substituting an unrelated feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pcu_select.features.cache import FeatureCache
from pcu_select.peft_space.site_mask import SiteSpace, alpha_vector
from pcu_select.selection.cluster import ClusterConfig, cluster_samples
from pcu_select.types import PEFTConfig

# Difficulty-vector indices used by name.
_LEN = 2
_LOSS = 4
_LOSS_STD = 5
_PPL = 7
_LOGPROB = 8


@dataclass
class BaselineInputs:
    """Everything a baseline may read, loaded once per (task) by the runner."""

    cache: FeatureCache
    sites: SiteSpace
    sample_ids: list[str]
    d_x: np.ndarray  # (N, 16) difficulty stats
    joint: np.ndarray  # (N, d_sem) joint semantic embeddings
    task_query_joint: np.ndarray | None = None  # (d_sem,) sketch mean embedding
    task_grad: np.ndarray | None = None  # (|Ω|, d_proj) task grad signature
    external_scores: dict[str, np.ndarray] = field(default_factory=dict)
    _grads: np.ndarray | None = None  # (N, |Ω|, d_proj), lazily stacked

    @classmethod
    def from_cache(
        cls,
        cache: FeatureCache,
        sites: SiteSpace,
        *,
        task_query_joint: np.ndarray | None = None,
        task_grad: np.ndarray | None = None,
    ) -> "BaselineInputs":
        feats = cache.read_features()
        ids = cache.read_sample_id_index()
        d_x = np.stack([feats[i].d_x.vector for i in ids], axis=0).astype(np.float32)
        joint = np.stack([feats[i].e_x.joint for i in ids], axis=0).astype(np.float32)
        external_scores = _load_external_scores(cache, ids)
        return cls(cache=cache, sites=sites, sample_ids=ids, d_x=d_x, joint=joint,
                   task_query_joint=task_query_joint, task_grad=task_grad,
                   external_scores=external_scores)

    def grads(self) -> np.ndarray:
        if self._grads is None:
            mats = [np.asarray(self.cache.read_grad_signature(s)) for s in self.sites.all_sites]
            self._grads = np.stack(mats, axis=1).astype(np.float32)  # (N, |Ω|, d_proj)
        return self._grads


def _resolve_budget(budget: float | int, n: int) -> int:
    return int(budget) if budget >= 1 else max(1, int(round(float(budget) * n)))


def _top_ids(ids: list[str], scores: np.ndarray, k: int) -> list[str]:
    idx = np.argsort(-scores)[:k]
    return [ids[i] for i in idx]


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def _load_external_scores(cache: FeatureCache, ids: list[str]) -> dict[str, np.ndarray]:
    path = cache.paths.root / "baseline_scores.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} must contain a sample_id column")
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_id values")
    indexed = df.set_index("sample_id")
    missing = [sid for sid in ids if sid not in indexed.index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{path} is missing baseline scores for sample ids: {preview}")
    out: dict[str, np.ndarray] = {}
    for name in ("ifd", "s2l"):
        if name in indexed.columns:
            out[name] = indexed.loc[ids, name].to_numpy(dtype=np.float32)
    return out


def _required_external_score(inp: BaselineInputs, name: str) -> np.ndarray:
    if name not in inp.external_scores:
        raise ValueError(
            f"baseline {name!r} requires features/baseline_scores.parquet with a {name!r} column"
        )
    return inp.external_scores[name]


# ---------------------------------------------------------------------------
# Selectors. Each: (inp, budget, peft, seed) -> list[SampleID]
# ---------------------------------------------------------------------------


def _random(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(inp.sample_ids))[:k]
    return [inp.sample_ids[i] for i in idx]


def _balanced_random(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    """Stratified random over domain proxy = KMeans clusters of joint embeddings.

    (Design's "balanced random" stratifies by source/domain; the feature cache
    does not persist `source`, so we use a clustering proxy for domain.)
    """
    asg = cluster_samples(sample_ids=inp.sample_ids, joint_embeddings=inp.joint,
                          cfg=ClusterConfig())
    rng = np.random.default_rng(seed)
    clusters = np.unique(asg.cluster_ids)
    per = max(1, k // len(clusters))
    picked: list[str] = []
    for c in clusters:
        members = np.where(asg.cluster_ids == c)[0]
        rng.shuffle(members)
        picked.extend(inp.sample_ids[i] for i in members[:per])
    return picked[:k]


def _length(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    return _top_ids(inp.sample_ids, inp.d_x[:, _LEN], k)


def _loss(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    return _top_ids(inp.sample_ids, inp.d_x[:, _LOSS], k)


def _perplexity(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    return _top_ids(inp.sample_ids, inp.d_x[:, _PPL], k)


def _ifd(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    """IFD selection from an exported instruction-free difficulty score."""
    return _top_ids(inp.sample_ids, _required_external_score(inp, "ifd"), k)


def _s2l(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    """S2L selection from an exported per-sample learnability score."""
    return _top_ids(inp.sample_ids, _required_external_score(inp, "s2l"), k)


def _rds_plus(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    """RDS+: cosine of standardized joint embedding to the sketch mean."""
    if inp.task_query_joint is None:
        raise ValueError("rds_plus needs task_query_joint (sketch mean embedding)")
    z = _standardize(inp.joint)
    q = (inp.task_query_joint - inp.joint.mean(0)) / (inp.joint.std(0) + 1e-6)
    q = q / (np.linalg.norm(q) + 1e-8)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    return _top_ids(inp.sample_ids, zn @ q, k)


def _embedding_nn(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    """Raw cosine of joint embedding to sketch mean (no standardization)."""
    if inp.task_query_joint is None:
        raise ValueError("embedding_nn needs task_query_joint")
    q = inp.task_query_joint / (np.linalg.norm(inp.task_query_joint) + 1e-8)
    zn = inp.joint / (np.linalg.norm(inp.joint, axis=1, keepdims=True) + 1e-8)
    return _top_ids(inp.sample_ids, zn @ q, k)


def _diversity(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    """Coverage-only: round-robin nearest-to-centroid across KMeans clusters."""
    asg = cluster_samples(sample_ids=inp.sample_ids, joint_embeddings=inp.joint,
                          cfg=ClusterConfig())
    picked: list[str] = []
    clusters = list(np.unique(asg.cluster_ids))
    # Pre-sort each cluster by distance to its centroid.
    order: dict[int, list[int]] = {}
    for c in clusters:
        members = np.where(asg.cluster_ids == c)[0]
        centroid = inp.joint[members].mean(0)
        d = np.linalg.norm(inp.joint[members] - centroid, axis=1)
        order[int(c)] = [int(members[i]) for i in np.argsort(d)]
    ptr = {int(c): 0 for c in clusters}
    while len(picked) < k and any(ptr[c] < len(order[c]) for c in ptr):
        for c in clusters:
            c = int(c)
            if ptr[c] < len(order[c]):
                picked.append(inp.sample_ids[order[c][ptr[c]]])
                ptr[c] += 1
                if len(picked) >= k:
                    break
    return picked[:k]


def _grad_sim(inp: BaselineInputs, k: int, peft, seed: int) -> list[str]:
    """Influence-style baseline: PEFT-agnostic mean cosine over all sites.

    u(x) = mean_ω cos(g_x^ω, g_t^ω). No PEFT conditioning — this is exactly the
    signal PCU-Select augments with α_p^ω, learned scoring and uncertainty.
    """
    if inp.task_grad is None:
        raise ValueError("grad_sim needs task_grad")
    g = inp.grads()  # (N, |Ω|, d_proj); rows already unit-normalized
    cos = np.einsum("noi,oi->no", g, inp.task_grad, optimize=True)  # (N, |Ω|)
    return _top_ids(inp.sample_ids, cos.mean(axis=-1), k)


def _less(inp: BaselineInputs, k: int, peft: PEFTConfig | None, seed: int) -> list[str]:
    """LESS-style per-PEFT influence: α_p^ω-weighted gradient similarity.

    This is the strongest non-learned baseline and is PEFT-*specific* (its
    selection changes with `peft`). In E2 its cost is charged per target PEFT
    (it must be recomputed for every configuration), unlike PCU-Select's
    amortized scorer.
    """
    if inp.task_grad is None or peft is None:
        raise ValueError("less needs task_grad and a PEFT config")
    g = inp.grads()
    cos = np.einsum("noi,oi->no", g, inp.task_grad, optimize=True)  # (N, |Ω|)
    alpha = alpha_vector(peft, inp.sites, normalize=True)  # (|Ω|,)
    return _top_ids(inp.sample_ids, (cos * alpha[None, :]).sum(axis=-1), k)


BASELINES = {
    "random": _random,
    "balanced_random": _balanced_random,
    "length": _length,
    "loss": _loss,
    "perplexity": _perplexity,
    "ifd": _ifd,
    "s2l": _s2l,
    "rds_plus": _rds_plus,
    "embedding_nn": _embedding_nn,
    "diversity": _diversity,
    "grad_sim": _grad_sim,  # influence (PEFT-agnostic)
    "less": _less,          # influence (PEFT-specific)
}


def select_baseline(
    name: str,
    inp: BaselineInputs,
    *,
    budget: float | int,
    peft: PEFTConfig | None = None,
    seed: int = 0,
) -> list[str]:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    k = _resolve_budget(budget, len(inp.sample_ids))
    return BASELINES[name](inp, k, peft, seed)


# ---------------------------------------------------------------------------
# Dense per-sample scores, for selection-quality metrics vs u^hi (design §1.6).
# Returns scores aligned with `inp.sample_ids`, or None when the baseline has no
# meaningful continuous score (random / balanced_random / diversity).
# ---------------------------------------------------------------------------


def score_baseline(name: str, inp: BaselineInputs, peft: PEFTConfig | None = None
                   ) -> np.ndarray | None:
    if name in ("random", "balanced_random", "diversity"):
        return None
    if name == "length":
        return inp.d_x[:, _LEN]
    if name == "loss":
        return inp.d_x[:, _LOSS]
    if name == "perplexity":
        return inp.d_x[:, _PPL]
    if name == "ifd":
        return _required_external_score(inp, "ifd")
    if name == "s2l":
        return _required_external_score(inp, "s2l")
    if name == "rds_plus":
        if inp.task_query_joint is None:
            return None
        z = _standardize(inp.joint)
        q = (inp.task_query_joint - inp.joint.mean(0)) / (inp.joint.std(0) + 1e-6)
        q = q / (np.linalg.norm(q) + 1e-8)
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        return zn @ q
    if name == "embedding_nn":
        if inp.task_query_joint is None:
            return None
        q = inp.task_query_joint / (np.linalg.norm(inp.task_query_joint) + 1e-8)
        zn = inp.joint / (np.linalg.norm(inp.joint, axis=1, keepdims=True) + 1e-8)
        return zn @ q
    if name in ("grad_sim", "less"):
        if inp.task_grad is None:
            return None
        cos = np.einsum("noi,oi->no", inp.grads(), inp.task_grad, optimize=True)
        if name == "grad_sim":
            return cos.mean(axis=-1)
        if peft is None:
            return None
        alpha = alpha_vector(peft, inp.sites, normalize=True)
        return (cos * alpha[None, :]).sum(axis=-1)
    return None
