# ShopPilot 二开新增(原课程版无此模块):语义缓存。
# 高频 FAQ 问题按嵌入余弦就近命中,直接复用答案与引用,跳过检索与生成;
# 只服务知识路(商品咨询类 FAQ),订单/物流/退款等个性化链路不进缓存。
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.config import settings
from app.core.embeddings import embed_query
from app.kb.dedup import normalize_question

logger = logging.getLogger(__name__)

# 存储选 SQLite 单文件而不是 Milvus:条目千级、单机单进程,内存暴力余弦(ms 级)足够;
# 缓存是纯增益项,不该引入额外服务依赖,test 也不用起容器。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_norm          TEXT    NOT NULL UNIQUE,   -- 归一化问题:精确命中用
    query_text          TEXT    NOT NULL,          -- resolved_query 原文:运维排查用
    embedding           BLOB    NOT NULL,          -- float32 向量(bge-m3 1024 维)
    answer              TEXT    NOT NULL,
    citations_json      TEXT    NOT NULL,
    evidence_confidence REAL    NOT NULL DEFAULT 0,
    intent              TEXT    NOT NULL DEFAULT '',
    created_at          TEXT    NOT NULL,
    expires_at          TEXT    NOT NULL,
    hit_count           INTEGER NOT NULL DEFAULT 0,
    last_hit_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_semantic_cache_expires ON semantic_cache(expires_at);
"""

_CONNS: dict[str, sqlite3.Connection] = {}
_LOCK = threading.Lock()


def _conn(db_path: str | None = None) -> sqlite3.Connection:
    """按库文件路径缓存连接(测试传 tmp 路径时互不影响);WAL 让读写互不阻塞。
    模块级单锁即可:临界区是单表小操作,图节点并发也就几个会话量级。"""
    path = db_path or settings.cache_db_path
    conn = _CONNS.get(path)
    if conn is None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        _CONNS[path] = conn
    return conn


def _now(now: datetime | None = None) -> datetime:
    return now or datetime.now(timezone.utc)


def _ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _pack(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _row_to_entry(row) -> dict:
    (rid, norm, text, blob, answer, cj, conf, intent, created, expires, hits, last_hit) = row
    return {"id": rid, "query_norm": norm, "query_text": text,
            "embedding": np.frombuffer(blob, dtype=np.float32),
            "answer": answer, "citations": json.loads(cj or "[]"),
            "evidence_confidence": conf, "intent": intent,
            "created_at": created, "expires_at": expires,
            "hit_count": hits, "last_hit_at": last_hit}


def find_best(query_vec, query_norm: str, threshold: float, *,
              db_path: str | None = None, now: datetime | None = None) -> dict | None:
    """就近命中:归一化问题完全一致优先(sim=1.0,不看向量),否则算余弦取最高分;
    最高分仍低于阈值返回 None。维度对不上的旧条目(换过嵌入模型)直接跳过。
    命中即累加 hit_count/last_hit_at——命中率报表和淘汰排序都吃这份书。"""
    ts = _ts(_now(now))
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q)) or 1.0
    with _LOCK:
        conn = _conn(db_path)
        rows = conn.execute(
            "SELECT id, query_norm, query_text, embedding, answer, citations_json, "
            "evidence_confidence, intent, created_at, expires_at, hit_count, last_hit_at "
            "FROM semantic_cache WHERE expires_at > ?", (ts,)).fetchall()
        best: dict | None = None
        for row in rows:
            entry = _row_to_entry(row)
            v = entry.pop("embedding")
            if entry["query_norm"] == query_norm:
                sim = 1.0
            elif v.shape != q.shape:
                continue
            else:
                denom = (float(np.linalg.norm(v)) or 1.0) * q_norm
                sim = float(np.dot(v, q) / denom)
            if best is None or sim > best["sim"]:
                best = {**entry, "sim": sim}
        if best is None or best["sim"] < threshold:
            return None
        conn.execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (ts, best["id"]))
        conn.commit()
        best["hit_count"] += 1
        best["last_hit_at"] = ts
        return best


def store_entry(query_text: str, query_vec, answer: str, citations: list,
                evidence_confidence: float = 0.0, intent: str = "", *,
                db_path: str | None = None, now: datetime | None = None,
                ttl_hours: int | None = None, max_entries: int | None = None) -> bool:
    """写入(Upsert):同一归一化问题覆盖旧答案并续期(知识更新后同问题答案会变)。
    随后超量清理:先删过期,仍超按「最后命中时间(无命中按创建时间)」从旧到新删。
    返回 True=新写入,False=覆盖更新或空问题被拒。"""
    ts = _now(now)
    ttl = settings.cache_ttl_hours if ttl_hours is None else ttl_hours
    cap = settings.cache_max_entries if max_entries is None else max_entries
    norm = normalize_question(query_text)
    if not norm:
        return False
    with _LOCK:
        conn = _conn(db_path)
        existed = conn.execute(
            "SELECT id FROM semantic_cache WHERE query_norm = ?", (norm,)).fetchone() is not None
        conn.execute(
            "INSERT INTO semantic_cache (query_norm, query_text, embedding, answer, "
            " citations_json, evidence_confidence, intent, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(query_norm) DO UPDATE SET "
            " query_text=excluded.query_text, embedding=excluded.embedding, "
            " answer=excluded.answer, citations_json=excluded.citations_json, "
            " evidence_confidence=excluded.evidence_confidence, intent=excluded.intent, "
            " created_at=excluded.created_at, expires_at=excluded.expires_at",
            (norm, query_text, _pack(query_vec), answer,
             json.dumps(citations, ensure_ascii=False), float(evidence_confidence), intent,
             _ts(ts), _ts(ts + timedelta(hours=ttl))))
        conn.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (_ts(ts),))
        n = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        if n > cap:
            conn.execute(
                "DELETE FROM semantic_cache WHERE id IN ("
                " SELECT id FROM semantic_cache"
                " ORDER BY COALESCE(last_hit_at, created_at) ASC LIMIT ?)", (n - cap,))
        conn.commit()
        return not existed


async def lookup(query: str) -> dict | None:
    """嵌入 + 就近命中;未启用/问题为空返回 None。异常由调用方兜(缓存不拦正路)。"""
    if not settings.cache_enabled:
        return None
    norm = normalize_question(query)
    if not norm:
        return None
    vec = await embed_query(query)
    return find_best(vec, norm, settings.cache_sim_threshold)


async def store_answer(query: str, answer: str, citations: list,
                       evidence_confidence: float = 0.0, intent: str = "") -> bool:
    if not settings.cache_enabled:
        return False
    vec = await embed_query(query)
    return store_entry(query, vec, answer, citations, evidence_confidence, intent)


def stats(*, db_path: str | None = None, now: datetime | None = None) -> dict:
    """运维统计:总量/有效/过期/累计命中/最热十条(scripts/cache_admin.py 用)。"""
    ts = _ts(_now(now))
    with _LOCK:
        conn = _conn(db_path)
        total = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM semantic_cache WHERE expires_at > ?", (ts,)).fetchone()[0]
        hits = conn.execute(
            "SELECT COALESCE(SUM(hit_count), 0) FROM semantic_cache").fetchone()[0]
        top = conn.execute(
            "SELECT query_text, hit_count, last_hit_at FROM semantic_cache "
            "ORDER BY hit_count DESC, COALESCE(last_hit_at, created_at) DESC LIMIT 10"
        ).fetchall()
    return {"total": total, "active": active, "expired": total - active,
            "total_hits": int(hits),
            "top": [{"query": q, "hits": h, "last_hit_at": lh} for q, h, lh in top]}


def clear(*, db_path: str | None = None) -> int:
    with _LOCK:
        conn = _conn(db_path)
        n = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        conn.execute("DELETE FROM semantic_cache")
        conn.commit()
        return n
