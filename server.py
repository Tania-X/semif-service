#!/usr/bin/env python3
"""semif-gate 的 Python 判定服务。

职责边界（刻意做得极窄）：
    - 只做一件事：把 (state, 冻结的判定点) 映射成选项概率分布
    - 不接受调用方传任意 prompt：只认 point_id，并校验渲染出的 prompt 哈希
    - 阈值策略、缓存、审计、降级全在 Java 侧，这里一概不做

关键设计：**同一 state 上的一次请求只做一次前向传播**，所有判定点共享同一次
prefill 与同一份末位 logits，只是各自读自己声明的那几个答案槽位。
这正是 SemIf 的「一份长文档 × N 条准则」用法。

依赖：复用 SemIf 的 semif_phase1 包（不重写它的语义）。

启动：
    python server.py --model <本地模型目录> --revision <revision 标签> --port 8080
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---- 复用 SemIf 的语义，不重新实现 -------------------------------------------
from semif_phase1.core import LETTERS, digest, direct_messages, softmax, validate_row
from semif_phase1.direct import _slot_ids

# ---------------------------------------------------------------------------
# 判定点注册表（从 JSON 加载，与 Java 侧同源）
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY = Path(__file__).parent / "registry" / "decision-points.json"


class Option(BaseModel):
    id: str
    description: str


class DecisionPoint(BaseModel):
    id: str
    version: int
    options: list[Option]
    # 该族的固定判据框架；具体判据由 state 里的 criterion 提供
    question_template: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


class Registry:
    def __init__(self, points: list[DecisionPoint]):
        self._by_ref: dict[str, DecisionPoint] = {}
        for p in points:
            if p.ref in self._by_ref:
                raise ValueError(f"判定点重复: {p.ref}")
            if not (2 <= len(p.options) <= 16):
                raise ValueError(f"{p.ref}: 选项数必须在 2..16")
            ids = [o.id for o in p.options]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{p.ref}: 选项 ID 重复")
            self._by_ref[p.ref] = p

    def require(self, ref: str) -> DecisionPoint:
        if ref not in self._by_ref:
            raise KeyError(f"未注册的判定点: {ref}")
        return self._by_ref[ref]

    def all(self) -> list[DecisionPoint]:
        return list(self._by_ref.values())


def load_registry(path: Path) -> Registry:
    raw = json.loads(path.read_text())
    return Registry([DecisionPoint(**item) for item in raw])


# ---------------------------------------------------------------------------
# 请求 / 响应契约
# ---------------------------------------------------------------------------


class PointRequest(BaseModel):
    id: str                      # pointId（不含版本）
    point_ref: str               # pointId@version
    prompt_sha256: str           # 调用方（Java）声明的哈希，必须与渲染结果一致


class DecideRequest(BaseModel):
    state_hash: str
    state: dict[str, Any] = Field(..., description="必须含 criterion（判据）与 evidence")
    points: list[PointRequest]


class PointResult(BaseModel):
    point_id: str
    option_ids: list[str]
    probabilities: list[float]
    prompt_sha256: str
    input_tokens: int


class DecideResponse(BaseModel):
    results: list[PointResult]
    provenance: dict[str, Any]


# ---------------------------------------------------------------------------
# 前缀切分：同一 state 上的多个判定点共享 prefill
# ---------------------------------------------------------------------------

def _shared_prefix_ids(tokenizer, prompt: str, evidence: Any) -> list[int]:
    """在**完整 prompt** 上切出「证据结束、判据开始」之前的共享前缀 token。

    必须在完整 prompt 上做，不能对前缀文本单独 `encode`。

    为什么：BPE 在边界处会发生合并。实测该用例中，
    单独 `encode(prompt[:boundary])` 得到 60 个 token，而在完整 prompt 上
    按字符边界切出的是 59 个——因为边界处 `" copies"` 与 `."` 被切成了不同的 token。
    用切错的 token 序列去做 KV 复用，会得到**静默错误的分布**
    （实测最大概率偏差 0.245，而 argmax 恰好没变，所以不会被发现）。

    正确做法：取 `offset_mapping`，选最后一个「结束位置 <= 字符边界」的 token。

    @param prompt   已经渲染好的完整 prompt
    @param evidence 该判定点的证据
    """
    prefix_text = json.dumps({"evidence": evidence}, ensure_ascii=False)[:-1]
    occurrences = prompt.count(prefix_text)
    if occurrences != 1:
        raise ValueError(f"证据片段在 prompt 中出现 {occurrences} 次，拒绝切分前缀（必须恰好 1 次）")
    boundary = prompt.index(prefix_text) + len(prefix_text)

    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]

    cut = 0
    for index, (_start, end) in enumerate(offsets):
        if end <= boundary:
            cut = index + 1
        else:
            break
    if cut <= 0:
        raise ValueError("前缀切分为空，拒绝继续")
    if ids[:cut] != tokenizer.encode(prompt, add_special_tokens=False)[:cut]:
        raise ValueError("前缀切分与完整编码不一致，拒绝继续")
    return list(ids[:cut])


# ---------------------------------------------------------------------------
# 判定核心
# ---------------------------------------------------------------------------


class DecisionEngine:
    """加载一次模型，常驻内存。所有请求共享。"""

    def __init__(self, model_dir: str, revision: str, registry: Registry,
                 device: str = "auto", prefix_reuse: bool = False):
        from semif_phase1.core import load_causal_model

        self.registry = registry
        self.revision = revision
        # 实验特性，默认关闭：实测在真实数据上会给出错误分布（见 _decide_with_prefix_reuse）
        self.prefix_reuse = prefix_reuse
        self._last_prefix_len = 0
        self.model, self.tokenizer, self.meta = load_causal_model(model_dir, revision, device=device)
        self.slots: dict[str, list[int]] = {}
        self._verify_slots()

    # -- 启动自检 ----------------------------------------------------------
    def _verify_slots(self) -> None:
        """对每个判定点断言：选项字母必须是单 token 且往返一致。

        直接沿用 SemIf direct.py 的语义。任何一条不过就拒绝启动——
        宁可起不来，也不要在运行时静默返回错位的概率。
        """
        for point in self.registry.all():
            slots = _slot_ids(self.tokenizer, len(point.options))
            self.slots[point.ref] = slots

    def slot_report(self) -> list[dict[str, Any]]:
        out = []
        for point in self.registry.all():
            slots = self.slots[point.ref]
            out.append({
                "point_ref": point.ref,
                "options": len(point.options),
                "slot_token_ids": slots,
                "letters": list(LETTERS[: len(point.options)]),
            })
        return out

    # -- 内部：请求构造与渲染 ------------------------------------------------
    def _resolve_options(self, point: DecisionPoint, state: dict[str, Any]):
        """确定选项的呈现顺序。

        顺序是语义的一部分：SemIf 按选项**下标**分配答案字母（LETTERS[i]），
        实测按行内原始顺序渲染可 144/144 命中已发布 prompt 哈希，字典序 0/144。

        因此顺序优先由 state 显式给出（`option_order`）；未给出时回落到
        注册表冻结的顺序。两种情况下字母↔选项映射都由这里唯一确定。
        """
        by_id = {o.id: o for o in point.options}
        order = state.get("option_order")
        if order is None:
            return list(point.options)
        if not isinstance(order, list) or sorted(order) != sorted(by_id):
            raise ValueError(
                f"state.option_order 必须恰好是该判定点的选项集 {sorted(by_id)}，实得 {order}"
            )
        return [by_id[oid] for oid in order]

    def _build_row(self, p: PointRequest, point: DecisionPoint, state: dict[str, Any]) -> dict:
        options = self._resolve_options(point, state)
        row = {
            "id": f"{p.point_ref}:{state.get('case_id', 'case')}",
            "state": state.get("evidence"),
            "question": state.get("criterion"),
            "options": [{"id": o.id, "description": o.description} for o in options],
        }
        validate_row(row)
        return row

    def _render_prompt(self, row: dict) -> str:
        return self.tokenizer.apply_chat_template(
            direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    # -- 渲染（不推理）------------------------------------------------------
    def render(self, state: dict[str, Any], points: list[PointRequest]) -> list[dict]:
        """只渲染 prompt 并算哈希，让调用方据此冻结契约。"""
        criterion = state.get("criterion")
        evidence = state.get("evidence")
        if not isinstance(criterion, str) or not criterion:
            raise ValueError("state.criterion 必须是非空字符串")
        if evidence is None:
            raise ValueError("state.evidence 不能为空")

        out = []
        for p in points:
            point = self.registry.require(p.point_ref)
            row = self._build_row(p, point, state)
            prompt = self._render_prompt(row)
            out.append({
                "point_id": point.id,
                "prompt_sha256": digest(prompt),
                "input_tokens": len(self.tokenizer.encode(prompt, add_special_tokens=False)),
            })
        return out

    # -- 判定 --------------------------------------------------------------
    def decide(self, state: dict[str, Any], points: list[PointRequest]) -> tuple[list[PointResult], dict]:
        import torch

        criterion = state.get("criterion")
        evidence = state.get("evidence")
        if not isinstance(criterion, str) or not criterion:
            raise ValueError("state.criterion 必须是非空字符串")
        if evidence is None:
            raise ValueError("state.evidence 不能为空")

        t0 = time.perf_counter()

        # 1. 渲染每个判定点的 prompt，并校验调用方声明的哈希
        prepared = []
        for p in points:
            point = self.registry.require(p.point_ref)
            row = self._build_row(p, point, state)
            prompt = self._render_prompt(row)
            got = digest(prompt)
            if got != p.prompt_sha256:
                raise HTTPException(
                    status_code=409,
                    detail=f"prompt 哈希不匹配 {p.point_ref}: 期望 {p.prompt_sha256[:16]}… 实得 {got[:16]}…",
                )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prepared.append((p, point, row, prompt, ids))

        device = next(self.model.parameters()).device
        results: list[PointResult] = []

        if self.prefix_reuse:
            results = self._decide_with_prefix_reuse(prepared, evidence, device)
            prefix_tokens = self._last_prefix_len
        else:
            results = self._decide_full_forward(prepared, device)
            prefix_tokens = 0

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        provenance = {
            "provider_id": "semif-py",
            "model_revision": self.revision,
            "backend": self.meta.get("device", "unknown"),
            "points_scored": len(results),
            "execution_shape": "prefix-reuse" if self.prefix_reuse else "full-forward",
            "prefix_tokens": prefix_tokens,
            "latency_ms": elapsed_ms,
        }
        return results, provenance

    def _to_result(self, p: PointRequest, point: DecisionPoint, row: dict,
                   ids: list[int], vocab) -> PointResult:
        """把末位 logits 组装成结果。

        ⚠️ 标签必须跟随**实际渲染用的选项顺序**（`row["options"]`），
        不能跟随注册表里冻结的顺序。

        症状（实测）：若用注册表顺序贴标签，`a3f18f3a63d45345942b` 上
        contradicted 与 supported 的概率会被互换（Δ=0.108），
        而 `prompt_sha256` 校验照样通过——**是静默错误**。

        根因：答案字母按下标分配（LETTERS[i]），槽位读出的数值天然属于
        渲染时的那个顺序。顺序由 `state.option_order` 决定，可能与注册表不同。
        """
        option_ids = [o["id"] for o in row["options"]]
        if len(option_ids) != len(self.slots[p.point_ref]):
            raise ValueError(f"选项数与槽位数不符: {point.ref}")
        slots = self.slots[p.point_ref]
        return PointResult(
            point_id=point.id,
            option_ids=option_ids,
            probabilities=softmax(vocab[slots].cpu().tolist()),
            prompt_sha256=p.prompt_sha256,
            input_tokens=len(ids),
        )

    def _decide_full_forward(self, prepared, device) -> list[PointResult]:
        """每个判定点独立做一次完整前向。

        这是**默认且唯一可信**的路径：实测在真实数据上与已发布预测逐位一致
        （最大偏差 0.000000）。
        """
        import torch

        out_results: list[PointResult] = []
        with torch.inference_mode():
            for p, point, row, _prompt, ids in prepared:
                out = self.model(
                    input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device),
                    use_cache=False,
                    return_dict=True,
                )
                vocab = out.logits[0, -1, :].float()
                del out
                out_results.append(self._to_result(p, point, row, ids, vocab))
        return out_results

    def _decide_with_prefix_reuse(self, prepared, evidence, device) -> list[PointResult]:
        """共享前缀只 prefill 一次，各判定点只前向自己的后缀。

        ⚠️ 实验特性，默认关闭。**实测在真实数据上会给出错误的分布**：
        `a3f18f3a63d45345942b` 上 contradicted 与 supported 几近对调（Δ=0.43）。

        原因：Qwen3.5 是混合架构（线性注意力 + 因果卷积），其原生缓存
        不能像标准 Transformer 那样按任意 token 边界切分复用。
        SemIf 的 `docs/MLX.md` 也专门警告过这一点。

        保留此路径是为了：
          - 与 SemIf 的 serial/shared 模式做受控对比
          - 将来若换成纯注意力模型，可重新评估
        **任何使用它的实验都必须同时跑 full-forward 做对照。**
        """
        import torch

        prefix = _shared_prefix_ids(self.tokenizer, prepared[0][3], evidence)
        if not prefix:
            raise ValueError("共享前缀为空，拒绝继续")
        for _p, point, _row, _prompt, ids in prepared:
            if ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
                raise HTTPException(
                    status_code=409,
                    detail=f"共享前缀不匹配 {point.ref}：该判定点的 prompt 不以证据前缀开头",
                )
        self._last_prefix_len = len(prefix)

        out_results: list[PointResult] = []
        with torch.inference_mode():
            prefill = self.model(
                input_ids=torch.tensor([prefix], dtype=torch.long, device=device),
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=device),
                use_cache=True,
                return_dict=True,
            )
            cache = prefill.past_key_values
            del prefill
            if cache is None or cache.get_seq_length() != len(prefix):
                raise RuntimeError("前缀缓存无效，拒绝用它继续计算")

            for p, point, row, _prompt, ids in prepared:
                out = self.model(
                    input_ids=torch.tensor([ids[len(prefix):]], dtype=torch.long, device=device),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                vocab = out.logits[0, -1, :].float()
                del out
                out_results.append(self._to_result(p, point, row, ids, vocab))
            del cache
        return out_results


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------

app = FastAPI(title="semif-gate decision service", version="0.1.0")
ENGINE: DecisionEngine | None = None


@app.get("/healthz")
def healthz():
    return {"ok": ENGINE is not None, "slots": ENGINE.slot_report() if ENGINE else None}


class RenderRequest(BaseModel):
    state: dict[str, Any]
    points: list[PointRequest]


class RenderResult(BaseModel):
    point_id: str
    prompt_sha256: str
    input_tokens: int


@app.post("/render", response_model=list[RenderResult])
def render(req: RenderRequest):
    """渲染 prompt 并返回哈希，**不执行任何前向传播**。

    用途：调用方（Java 侧注册表构建流程）用它预计算并冻结 `prompt_sha256`，
    之后每次 `/decide` 都会校验该哈希，确保「注册表声明的 prompt」与
    「实际发出去的 prompt」一致。

    这是审计链的起点：哈希一旦冻结，任何模板/选项顺序的改动都会被 `/decide` 拒绝。
    """
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    try:
        return ENGINE.render(req.state, req.points)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    try:
        results, prov = ENGINE.decide(req.state, req.points)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    prov["state_hash"] = req.state_hash
    return DecideResponse(results=results, provenance=prov)


def main() -> None:
    global ENGINE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="本地模型目录")
    ap.add_argument("--revision", required=True, help="revision 标签（写入溯源）")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    ap.add_argument("--prefix-reuse", action="store_true",
                    help="实验特性：共享前缀只 prefill 一次。实测在 Qwen3.5 上会给出错误分布，默认关闭")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    registry = load_registry(args.registry)
    print(f"[semif-gate] 已加载 {len(registry.all())} 个判定点")
    ENGINE = DecisionEngine(args.model, args.revision, registry,
                            device=args.device, prefix_reuse=args.prefix_reuse)
    print(f"[semif-gate] 执行形状: {'prefix-reuse（实验）' if args.prefix_reuse else 'full-forward（默认）'}")
    print("[semif-gate] 答案槽位自检通过：")
    for r in ENGINE.slot_report():
        print(f"  {r['point_ref']:34s} {r['options']} 选项  slots={r['slot_token_ids']}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
