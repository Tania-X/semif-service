#!/usr/bin/env python3
"""semif-service 的最小客户端。

用途：冻结契约、验证哈希、跑单条判定。Java 网关用的是同一个 HTTP 契约。

用法：
    python client.py render --url http://127.0.0.1:8080 \
        --criterion "May copies be sold?" \
        --evidence "The architect forbids selling copies." \
        --order insufficient,prohibited,permitted \
        --point semif.rule_application@1

    python client.py decide …（同上参数）
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post(base: str, path: str, payload: dict, timeout: int = 120) -> dict | list:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        print(f"HTTP {error.code}: {body[:500]}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as error:
        print(f"连接失败 {base}: {error}", file=sys.stderr)
        raise SystemExit(1)


def build_state(args) -> dict:
    state = {"criterion": args.criterion, "evidence": args.evidence}
    if args.order:
        state["option_order"] = [item.strip() for item in args.order.split(",") if item.strip()]
    if args.case_id:
        state["case_id"] = args.case_id
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("render", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8080")
        p.add_argument("--point", required=True, help="point_ref，例如 semif.rule_application@1")
        p.add_argument("--criterion", required=True, help="判据")
        p.add_argument("--evidence", required=True, help="证据（字符串；JSON 请用 --evidence-json）")
        p.add_argument("--evidence-json", help="证据为 JSON 时使用，优先于 --evidence")
        p.add_argument("--order", help="选项顺序，逗号分隔；省略则用注册表冻结顺序")
        p.add_argument("--case-id", default="demo")

    args = parser.parse_args()
    if args.evidence_json:
        args.evidence = json.loads(args.evidence_json)

    state = build_state(args)
    point = {"id": args.point.split("@")[0], "point_ref": args.point, "prompt_sha256": "0" * 64}

    rendered = post(args.url, "/render", {"state": state, "points": [point]})[0]
    point["prompt_sha256"] = rendered["prompt_sha256"]

    if args.command == "render":
        print(json.dumps(rendered, ensure_ascii=False, indent=2))
        return

    body = post(args.url, "/decide", {"state_hash": "client", "state": state, "points": [point]})
    result = body["results"][0]
    distribution = dict(zip(result["option_ids"], result["probabilities"]))

    print(f"判定点     : {result['point_id']}")
    print(f"prompt 哈希: {rendered['prompt_sha256']}")
    print(f"输入 token : {result['input_tokens']}")
    print(f"执行形状   : {body['provenance']['execution_shape']}")
    print(f"耗时       : {body['provenance']['latency_ms']} ms")
    print("分布（按 option_id 对齐，顺序无关）:")
    for option_id, probability in sorted(distribution.items(), key=lambda kv: -kv[1]):
        print(f"  {option_id:16s} {probability:.6f}")
    winner = max(distribution, key=distribution.get)
    top = sorted(distribution.values(), reverse=True)
    margin = top[0] - top[1] if len(top) > 1 else top[0]
    print(f"argmax     : {winner}   margin = {margin:.6f}"
          + ("   ← 精确平局" if margin < 1e-9 else ""))


if __name__ == "__main__":
    main()
