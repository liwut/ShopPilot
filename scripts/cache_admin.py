"""语义缓存运维工具(二开):stats / clear。

用法:
  PYTHONPATH=. uv run python scripts/cache_admin.py stats
  PYTHONPATH=. uv run python scripts/cache_admin.py clear
"""
import argparse
import json
import sys

from app.core import semantic_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="语义缓存运维(stats/clear)")
    parser.add_argument("action", choices=["stats", "clear"])
    args = parser.parse_args()
    if args.action == "stats":
        print(json.dumps(semantic_cache.stats(), ensure_ascii=False, indent=2))
    else:
        n = semantic_cache.clear()
        print(f"已清空 {n} 条缓存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
