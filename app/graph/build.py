from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routing import (
    confidence_gate, route_after_cache, route_after_fetch_order, route_by_intent, should_continue,
)
from app.graph.state import ConversationState


def _builder() -> StateGraph:
    b = StateGraph(ConversationState)
    # 节点
    b.add_node("resolve_reference", nodes.resolve_reference)
    b.add_node("classify_intent", nodes.classify_intent)
    b.add_node("cache_lookup", nodes.cache_lookup)                  # 二开:知识路语义缓存查询
    b.add_node("cache_reply", nodes.cache_reply)                    # 二开:缓存命中出口
    b.add_node("cache_store", nodes.cache_store)                    # 二开:知识路终稿落缓存
    b.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    b.add_node("confidence_check", nodes.confidence_check)  # 实体节点(trace 门控);判断在其后条件边
    b.add_node("main_agent", nodes.main_agent)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_node("complaint_reply", nodes.complaint_reply)
    b.add_node("script_reply", nodes.script_reply)
    b.add_node("fallback_reply", nodes.fallback_reply)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_node("retrieve_policy", nodes.retrieve_policy)
    b.add_node("retrieve_price_policy", nodes.retrieve_price_policy)  # 二开:价保政策检索
    b.add_node("log", nodes.log_node)

    # 骨架:消解 → 意图 → 缓存探测(二开) → 分流
    b.add_edge(START, "resolve_reference")
    b.add_edge("resolve_reference", "classify_intent")
    b.add_edge("classify_intent", "cache_lookup")
    b.add_conditional_edges("cache_lookup", route_after_cache, {
        "hit": "cache_reply",           # 二开:命中复用缓存答案与引用,跳过检索+生成
        "escalate": "complaint_reply",
        "fallback_script": "script_reply",
        "knowledge": "retrieve_knowledge",
        "refund_flow": "fetch_order",
        "price_flow": "fetch_order",    # 二开:价保路与退款路共用取单节点
        "business": "main_agent",
    })
    b.add_edge("cache_reply", "log")
    # 确定性子流程链:取单 → 政策检索(按 route 二次分流) → 交主力 Agent 判定
    b.add_conditional_edges("fetch_order", route_after_fetch_order, {
        "refund_flow": "retrieve_policy",
        "price_flow": "retrieve_price_policy",   # 二开:价保政策
    })
    b.add_edge("retrieve_policy", "main_agent")
    b.add_edge("retrieve_price_policy", "main_agent")
    # 知识路:检索 → 生成前证据闸
    b.add_edge("retrieve_knowledge", "confidence_check")
    b.add_conditional_edges("confidence_check", confidence_gate, {
        "strong": "main_agent",
        "weak": "fallback_reply",
    })
    # 主力 Agent ReAct 环
    b.add_conditional_edges("main_agent", should_continue, {
        "continue": "agent_tools",
        "stop": "cache_store",          # 二开:收敛后先过缓存收口(仅知识路单步落缓存)
    })
    b.add_edge("agent_tools", "main_agent")
    # 确定性出口 → 日志 → END
    b.add_edge("complaint_reply", "log")
    b.add_edge("script_reply", "log")
    b.add_edge("fallback_reply", "log")
    b.add_edge("cache_store", "log")
    b.add_edge("log", END)
    return b


def build_graph(checkpointer=None):
    return _builder().compile(checkpointer=checkpointer)
