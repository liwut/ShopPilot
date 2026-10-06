"""售后 MCP Server(ch08 自建,mock 数据,不接真实系统、不建表)。
独立进程:uv run python mcp_servers/aftersales_server.py
工具:query_warranty(查在保)、query_return_status(查退货进度)、
apply_price_protection(提交价保申请——二开新增的 mock 写操作,不落库,返回受理码)。
MOCK_DELAY_SECONDS 同物流 Server,可注入延迟演示超时。"""
import asyncio
import os
import random
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("aftersales",
              host="127.0.0.1", port=int(os.environ.get("PORT", "8102")))

_DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))


@mcp.tool()
async def query_warranty(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单商品是否在保修期内(在保状态、到期日)。用于用户问保修/在保时。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"warranty:{order_id}")
    code = rng.choice(["IN_WARRANTY", "EXPIRED"])         # 内部枚举码,client 侧翻人话
    return {"order_id": order_id, "warranty_code": code,
            "warranty_until": f"2026-{rng.randint(8, 12):02d}-{rng.randint(1, 28):02d}",
            "policy_ref": "AS-POLICY-07"}                 # 内部政策编号(回答用不上,client 侧应剔除)


@mcp.tool()
async def query_return_status(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单的退货进度(审核中/退货中/已退款/无退货记录)。用于用户问退货到哪一步了。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"return:{order_id}")
    code = rng.choice(["AUDITING", "RETURNING", "REFUNDED", "NONE"])
    return {"order_id": order_id, "return_code": code,
            "updated_at": f"2026-07-{rng.randint(1, 16):02d} 10:00"}


@mcp.tool()
async def apply_price_protection(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """提交价格保护(价保)申请:核验该订单并受理价保。仅当用户明确要求申请价保时调用。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"priceprotect:{order_id}")
    code = rng.choice(["ACCEPTED", "REJECTED_PRICE_STABLE"])   # 内部枚举码,client 侧翻人话
    return {"order_id": order_id, "apply_code": code,
            "protection_no": f"PB{rng.randint(10**9, 10**10 - 1)}"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
