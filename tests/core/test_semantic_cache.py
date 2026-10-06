# 二开测试:语义缓存(精确命中/近似命中/TTL/淘汰/运维)。
from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings
from app.core import semantic_cache

V_A = [1.0, 0.0, 0.0, 0.0]          # 主方向
V_NEAR = [1.0, 0.05, 0.0, 0.0]      # 夹角很小:sim≈0.99875
V_ORTH = [0.0, 1.0, 0.0, 0.0]       # 正交:sim=0
T0 = datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def cache_db(tmp_path, monkeypatch):
    path = str(tmp_path / "cache.sqlite")
    monkeypatch.setattr(settings, "cache_db_path", path)
    return path


def test_exact_norm_hit_beats_threshold(cache_db):
    semantic_cache.store_entry("怎么申请退货", V_ORTH, "答案A", [{"n": 1}], 0.9, "商品咨询",
                               db_path=cache_db, now=T0)
    # 向量完全不同,但归一化问题一致 → 直接命中(sim=1.0),并累计命中次数
    hit = semantic_cache.find_best(V_A, semantic_cache.normalize_question("怎么申请退货"),
                                   0.999, db_path=cache_db, now=T0)
    assert hit is not None and hit["sim"] == 1.0 and hit["answer"] == "答案A"
    assert hit["hit_count"] == 1
    hit2 = semantic_cache.find_best(V_A, semantic_cache.normalize_question("怎么申请退货"),
                                    0.999, db_path=cache_db, now=T0)
    assert hit2["hit_count"] == 2


def test_similarity_threshold(cache_db):
    semantic_cache.store_entry("怎么退货", V_A, "答案B", [], 0.9, "商品咨询",
                               db_path=cache_db, now=T0)
    norm = semantic_cache.normalize_question("如何退货")
    near = semantic_cache.find_best(V_NEAR, norm, 0.9, db_path=cache_db, now=T0)
    assert near is not None and near["answer"] == "答案B"
    # 阈值收紧到 0.9995 时,同一对向量不再命中(相似度≈0.99875)
    strict = semantic_cache.find_best(V_NEAR, norm, 0.9995, db_path=cache_db, now=T0)
    assert strict is None


def test_orthogonal_and_dim_mismatch_miss(cache_db):
    semantic_cache.store_entry("怎么退货", V_A, "答案C", [], 0.9, "", db_path=cache_db, now=T0)
    assert semantic_cache.find_best(V_ORTH, "别的问法", 0.5, db_path=cache_db, now=T0) is None
    # 维度不一致的旧条目(换过嵌入模型)不参与比较
    assert semantic_cache.find_best([1.0, 0.0, 0.0], "又一问法", 0.5,
                                    db_path=cache_db, now=T0) is None


def test_ttl_expiry(cache_db):
    semantic_cache.store_entry("怎么退货", V_A, "答案D", [], 0.9, "",
                               db_path=cache_db, now=T0, ttl_hours=1)
    assert semantic_cache.find_best(V_A, "别的问法", 0.99, db_path=cache_db, now=T0) is not None
    later = T0 + timedelta(hours=2)
    assert semantic_cache.find_best(V_A, "别的问法", 0.99, db_path=cache_db, now=later) is None
    assert semantic_cache.stats(db_path=cache_db, now=later)["expired"] == 1


def test_eviction_removes_oldest_untouched(cache_db, monkeypatch):
    monkeypatch.setattr(settings, "cache_max_entries", 2)
    semantic_cache.store_entry("问题一", V_A, "一", [], 0.9, "", db_path=cache_db, now=T0)
    # 三个向量互相正交:确保"命中的是这一条",而不是被邻居的相似度兜进来
    semantic_cache.store_entry("问题二", [0.0, 1.0, 0.0, 0.0], "二", [], 0.9, "",
                               db_path=cache_db, now=T0 + timedelta(seconds=1))
    semantic_cache.store_entry("问题三", [0.0, 0.0, 1.0, 0.0], "三", [], 0.9, "",
                               db_path=cache_db, now=T0 + timedelta(seconds=2))
    now = T0 + timedelta(seconds=3)
    assert semantic_cache.stats(db_path=cache_db, now=now)["total"] == 2
    # 最旧的「问题一」被清掉(其余条目与 V_A 正交,相似度不会兜中)
    assert semantic_cache.find_best(V_A, semantic_cache.normalize_question("问题一"), 0.9,
                                    db_path=cache_db, now=now) is None


def test_hit_recency_protects_from_eviction(cache_db, monkeypatch):
    monkeypatch.setattr(settings, "cache_max_entries", 2)
    semantic_cache.store_entry("问题一", V_A, "一", [], 0.9, "", db_path=cache_db, now=T0)
    semantic_cache.store_entry("问题二", [0.0, 1.0, 0.0, 0.0], "二", [], 0.9, "",
                               db_path=cache_db, now=T0 + timedelta(seconds=1))
    # 命中一下最旧的条目(刷新 last_hit_at),再存新条目触发淘汰
    hit = semantic_cache.find_best(V_A, semantic_cache.normalize_question("问题一"), 0.99,
                                   db_path=cache_db, now=T0 + timedelta(seconds=10))
    assert hit is not None
    semantic_cache.store_entry("问题三", [0.0, 0.0, 1.0, 0.0], "三", [], 0.9, "",
                               db_path=cache_db, now=T0 + timedelta(seconds=11))
    assert semantic_cache.stats(db_path=cache_db,
                                now=T0 + timedelta(seconds=12))["total_hits"] == 1
    # 被命中过的「问题一」留下,较旧的「问题二」被清
    now = T0 + timedelta(seconds=12)
    assert semantic_cache.find_best(V_A, semantic_cache.normalize_question("问题一"), 0.9,
                                    db_path=cache_db, now=now) is not None
    assert semantic_cache.find_best([0.0, 1.0, 0.0, 0.0],
                                    semantic_cache.normalize_question("问题二"), 0.9,
                                    db_path=cache_db, now=now) is None


def test_clear_and_stats(cache_db):
    semantic_cache.store_entry("问题一", V_A, "一", [], 0.9, "", db_path=cache_db, now=T0)
    semantic_cache.store_entry("问题二", [0.0, 1.0, 0.0, 0.0], "二", [], 0.9, "",
                               db_path=cache_db, now=T0)
    semantic_cache.find_best(V_A, semantic_cache.normalize_question("问题一"), 0.99,
                             db_path=cache_db, now=T0)
    stats = semantic_cache.stats(db_path=cache_db, now=T0)
    assert stats["total"] == 2 and stats["active"] == 2 and stats["total_hits"] == 1
    assert stats["top"][0]["query"] == "问题一" and stats["top"][0]["hits"] == 1
    assert semantic_cache.clear(db_path=cache_db) == 2
    assert semantic_cache.stats(db_path=cache_db, now=T0)["total"] == 0


@pytest.mark.asyncio
async def test_async_wrappers_with_fake_embed(cache_db, monkeypatch):
    async def fake_embed(text):
        return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(semantic_cache, "embed_query", fake_embed)
    assert await semantic_cache.store_answer("怎么退货", "答案E", [{"n": 1}], 0.8, "商品咨询")
    hit = await semantic_cache.lookup("怎么退货")
    assert hit is not None and hit["answer"] == "答案E"
    # 关掉开关:两个入口都直接放弃,不产副作用
    monkeypatch.setattr(settings, "cache_enabled", False)
    assert await semantic_cache.lookup("怎么退货") is None
    assert not await semantic_cache.store_answer("新问题", "答案F", [], 0.8, "")
