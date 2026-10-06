from openai import AsyncOpenAI

from app.config import settings

_CLIENT: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    """直连嵌入上游(硅基流动的 bge-m3,OpenAI 兼容)。
    进程内单例:复用同一 httpx 连接池,避免每次 embed 新建客户端累积连接/fd。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)
    return _CLIENT


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """二开:分片调用(默认 10 条/请求)——阿里云百炼 text-embedding-v4 单请求最多 10 行,
    超了直接 400;统一按批切齐,对硅基流动等更宽的上游只是多几次调用,行为不变。
    顺序不变:按请求顺序展开,调用方按下标对位不受影响。"""
    out: list[list[float]] = []
    batch = max(1, settings.embed_batch_size)
    for i in range(0, len(texts), batch):
        resp = await _client().embeddings.create(
            model=settings.embed_model, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)
    return out


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
