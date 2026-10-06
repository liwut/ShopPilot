# 二开测试:价保子流程(意图路由/政策检索/确认-提交闭环/续跑守卫)。
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import nodes, routing
from app.graph.build import build_graph
from app.graph.runtime import _graph_input
from app.tools import engine, registry


def test_price_route_mapping():
    assert routing.INTENT_TO_ROUTE["价保"] == "price_flow"
    assert routing.route_by_intent({"intent": "价保"}) == "price_flow"
    assert routing.route_after_fetch_order({"route": "price_flow"}) == "price_flow"
    assert routing.route_after_fetch_order({"route": "refund_flow"}) == "refund_flow"
    assert routing.route_after_fetch_order({}) == "refund_flow"


@pytest.mark.asyncio
async def test_retrieve_price_policy_seed_and_trace(monkeypatch):
    seen = {}

    async def fake_expand(seed):
        seen["seed"] = seed
        return ["价保条件", "价保时效"]

    async def fake_search(q, **k):
        return [{"id": 1, "question": "价保", "answer": "下单后 7 天内可申请",
                 "rerank_score": 0.9, "section_path": "价保/规则", "content_type": "policy"}]

    monkeypatch.setattr(nodes.query_understanding, "expand_queries", fake_expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)

    out = await nodes.retrieve_price_policy(
        {"resolved_query": "这单能价保吗", "order_data": {"product": "猫粮 5kg"}})
    assert "价格保护" in seen["seed"] and "猫粮 5kg" in seen["seed"]   # 种子带价保语义词
    assert out["citations"][0]["id"] == 1 and "[1]" in out["evidence"]
    assert out["trace"]["retrieve_price_policy"]["hits"] == 1


def _price_spec() -> registry.ToolSpec:
    """价保工具测试桩:名字进 WRITE_TOOLS → 自动按写操作管控(重试/确认/审计同真工具)。"""

    async def fake_apply(order_id: str) -> dict:
        return {"order_id": order_id, "apply_code": "ACCEPTED", "protection_no": "PB123456"}

    tool = StructuredTool.from_function(coroutine=fake_apply, name="apply_price_protection",
                                        description="提交价保申请(测试桩)")
    return registry.spec_from_langchain_tool(tool, source="mcp", mcp_server="aftersales")


class FakeModel:
    """第一次调用发 apply_price_protection tool_call,之后回普通文本收敛。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def bind(self, **kw):
        return self

    async def ainvoke(self, msgs, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "apply_price_protection", "id": "pc-1", "args": {"order_id": "1001"}}])
        return AIMessage(content="已为您提交价保申请,请留意处理结果。")


@pytest.fixture()
def wired_price(monkeypatch):
    async def fake_coref(q, h):
        return q

    async def fake_intent(q, h):
        return {"intent": "价保", "confidence": 0.92}

    async def fake_expand(seed):
        return ["价保政策"]

    async def fake_search(q, **k):
        return [{"id": 1, "question": "价保", "answer": "下单后 7 天内可申请",
                 "rerank_score": 0.9, "section_path": "价保/规则", "content_type": "policy"}]

    monkeypatch.setattr(nodes.coref, "resolve", fake_coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_intent)
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", fake_expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)
    model = FakeModel()     # 实例要复用:每次新实例会让 calls 归零,续跑时模型又发一次工具调用
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: model)

    async def only_price():
        return [_price_spec()]

    monkeypatch.setattr(nodes.registry, "get_all_specs", only_price)

    audits = []

    async def fake_audit(**kw):
        audits.append(kw)

    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)

    async def no_msg(*a, **kw):
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", no_msg)
    return audits


async def test_price_confirm_true_submits(wired_price):
    audits = wired_price
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tp1"}}
    st = await g.ainvoke(_graph_input("u1", "订单1001降价了,我要价保", 1, 1, "", 0), cfg)
    intr = st["__interrupt__"][0].value
    assert intr["type"] == "confirm_price_protection"
    assert intr["preview"]["order_id"] == "1001"                # interrupt 时未执行
    assert not any(a["tool_name"] == "apply_price_protection" for a in audits)

    st2 = await g.ainvoke(Command(resume={"confirmed": True}), cfg)
    assert any(a["status"] == "成功" and a["tool_name"] == "apply_price_protection"
               for a in audits)
    assert "PB123456" in str(st2["messages"])                   # 受理号回灌到 ToolMessage


async def test_price_confirm_false_denied(wired_price):
    audits = wired_price
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tp2"}}
    await g.ainvoke(_graph_input("u1", "订单1001降价了,我要价保", 1, 1, "", 0), cfg)
    st2 = await g.ainvoke(Command(resume={"confirmed": False}), cfg)
    assert any(a["status"] == "权限拒绝" and a["tool_name"] == "apply_price_protection"
               for a in audits)
    assert "取消" in str(st2["messages"])                        # 取消语境回灌模型


async def test_price_resume_mismatch_treated_as_cancel(wired_price):
    """resume 值形状守卫:价保确认卡挂起时误传 order_id 字符串(三种中断共用一个端点),
    按未确认处理——不提交、不炸节点。"""
    audits = wired_price
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tp3"}}
    await g.ainvoke(_graph_input("u1", "订单1001降价了,我要价保", 1, 1, "", 0), cfg)
    st2 = await g.ainvoke(Command(resume="1001"), cfg)           # 错配:字符串而非 {"confirmed": bool}
    assert any(a["status"] == "权限拒绝" for a in audits)
    assert "__interrupt__" not in st2, f"意料外的再次中断: {st2.get('__interrupt__')}"  # 没炸
