"""FeatureCache + JsonlPool round-trip (pure numpy/pandas, no torch)."""

from __future__ import annotations

import numpy as np

from pcu_select.data import JsonlPool
from pcu_select.features.cache import FeatureCache
from pcu_select.types import (
    ActivationSignature,
    DifficultyStats,
    SampleFeatures,
    SemanticEmbedding,
)


def _fake_features(n: int, d_sem: int = 8, d_diff: int = 16, d_act: int = 64):
    rng = np.random.default_rng(0)
    feats = []
    for i in range(n):
        e = SemanticEmbedding(
            instr=rng.normal(size=d_sem).astype(np.float32),
            resp=rng.normal(size=d_sem).astype(np.float32),
            joint=rng.normal(size=d_sem).astype(np.float32),
        )
        feats.append(
            SampleFeatures(
                sample_id=f"s{i}",
                e_x=e,
                d_x=DifficultyStats(vector=rng.normal(size=d_diff).astype(np.float32)),
                a_x=ActivationSignature(vector=rng.normal(size=d_act).astype(np.float32)),
            )
        )
    return feats


def test_feature_roundtrip_preserves_vectors(tmp_path):
    feats = _fake_features(4)
    cache = FeatureCache(tmp_path)
    cache.write_features(feats)
    cache.write_sample_id_index([f.sample_id for f in feats])

    loaded = cache.read_features()
    assert set(loaded) == {f.sample_id for f in feats}
    for f in feats:
        g = loaded[f.sample_id]
        assert np.allclose(g.e_x.joint, f.e_x.joint, atol=1e-6)
        assert np.allclose(g.d_x.vector, f.d_x.vector, atol=1e-6)
        # as_z_x concatenation length is stable.
        assert g.as_z_x().shape[0] == f.e_x.joint.shape[0] + 16 + 64


def test_grad_signature_roundtrip(tmp_path):
    cache = FeatureCache(tmp_path)
    site = (3, "attn_out")
    mat = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.write_grad_signature(site, mat)
    back = np.asarray(cache.read_grad_signature(site))
    assert np.array_equal(back, mat)


def test_sample_id_index_roundtrip(tmp_path):
    cache = FeatureCache(tmp_path)
    ids = ["a", "b", "c"]
    cache.write_sample_id_index(ids)
    assert cache.read_sample_id_index() == ids


def test_jsonl_pool_roundtrip(tmp_path):
    src = tmp_path / "pool.jsonl"
    src.write_text(
        '{"instruction": "1+1?", "response": "2"}\n'
        '{"instruction": "capital of France?", "response": "Paris"}\n'
    )
    pool = JsonlPool.from_jsonl(src)
    assert len(pool) == 2
    ids = [s.sample_id for s in pool]
    # content hashing yields stable, distinct ids.
    assert len(set(ids)) == 2
    taken = pool.take([ids[1]])
    assert taken[0].instruction == "capital of France?"
