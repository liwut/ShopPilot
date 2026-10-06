# 二开测试:缓存三节点与路由(命中短路/收口门控/异常降级)。
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes, routing


@pytest.mark.asyncio
async def test_lookup_skips_non_knowledge(monkeypatch):
    async def must_not_call(_):
        raise AssertionError("非知识路不应触碰缓存")

    monkeypatch.setattr(nodes.semantic_cache, "lookup", must_not_call)
    out = await nodes.cache_lookup({"route": "business", "resolved_query": "查物流"})
    assert out == {}


@pytest.mark.asyncio
async def test_lookup_miss_and_hit(monkeypatch):
    async def miss(_):
        return None

    monkeypatch.setattr(nodes.semantic_cache, "lookup", miss)
    out = await nodes.cache_lookup({"route": "knowledge", "resolved_query": "怎么退货"})
    assert out["trace"]["cache"] == "miss" and "cache_hit" not in out

    async def hit(_):
        return {"answer": "七天内可退", "citations": [{"n": 1}], "sim": 0.97, "hit_count": 3}

    monkeypatch.setattr(nodes.semantic_cache, "lookup", hit)
    out = await nodes.cache_lookup({"route": "knowledge", "resolved_query": "怎么退货"})
    assert out["cache_hit"]["answer"] == "七天内可退"
    assert out["trace"]["cache"] == "hit"


@pytest.mark.asyncio
async def test_lookup_exception_degrades_to_miss(monkeypatch):
    async def boom(_):
        raise RuntimeError("embed 上游抖动")

    monkeypatch.setattr(nodes.semantic_cache, "lookup", boom)
    out = await nodes.cache_lookup({"route": "knowledge", "resolved_query": "怎么退货"})
    assert out["trace"]["cache"] == "miss"


@pytest.mark.asyncio
async def test_reply_reuses_hit():
    out = await nodes.cache_reply(
        {"cache_hit": {"answer": "七天内可退", "citations": [{"n": 1}],
                       "sim": 0.975, "hit_count": 2}})
    assert out["answer"] == "七天内可退"
    assert out["citations"] == [{"n": 1}]
    assert out["trace"]["cache"] == "hit" and out["trace"]["cache_sim"] == 0.975


def _eligible_state(**over):
    state = {"route": "knowledge", "steps": 1, "evidence_strong": True,
             "evidence_confidence": 0.5, "citations": [{"n": 1}],
             "resolved_query": "怎么退货", "intent": "商品咨询", "conversation_id": 1,
             "messages": [HumanMessage("怎么退货"), AIMessage("七天内无理由退货,详见[1]")]}
    state.update(over)
    return state


@pytest.mark.asyncio
async def test_store_gates(monkeypatch):
    calls = []

    async def fake_store(**kw):
        calls.append(kw)
        return True

    monkeypatch.setattr(nodes.semantic_cache, "store_answer", fake_store)

    out = await nodes.cache_store(_eligible_state())
    assert out["trace"]["cache_store"] == "stored"
    assert calls[0]["query"] == "怎么退货"
    assert calls[0]["answer"] == "七天内无理由退货,详见[1]"
    assert calls[0]["intent"] == "商品咨询"

    # 任一收口条件不满足都不落:调过工具/证据弱/低置信/有动作建议/非知识路/无引用/兜底过
    for bad in ({"steps": 2}, {"evidence_strong": False}, {"evidence_confidence": 0.1},
                {"suggested_actions": [{"type": "transfer_human"}]},
                {"route": "business"}, {"citations": []}, {"fallback_source": "self_check"}):
        assert await nodes.cache_store(_eligible_state(**bad)) == {}
    # 总开关关掉:直接不落
    monkeypatch.setattr(nodes.settings, "cache_enabled", False)
    assert await nodes.cache_store(_eligible_state()) == {}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_store_failure_is_swallowed(monkeypatch):
    async def boom(**kw):
        raise RuntimeError("sqlite 写失败")

    monkeypatch.setattr(nodes.semantic_cache, "store_answer", boom)
    out = await nodes.cache_store(_eligible_state())
    assert out["trace"]["cache_store"] == "skip"


def test_route_after_cache():
    assert routing.route_after_cache({"cache_hit": {"answer": "x"}}) == "hit"
    assert routing.route_after_cache({"intent": "商品咨询"}) == "knowledge"
    assert routing.route_after_cache({"intent": "投诉"}) == "escalate"
    assert routing.route_after_cache({}) == "business"
