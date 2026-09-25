# semif-service

把「语义判定」暴露成一个**窄而可审计**的 HTTP 服务：给一份 state 和若干冻结的判定点，返回每个选项的概率分布。

这是 [SemIf](https://github.com/TheoLeeCJ/SemIf) 的直接 logits 读出法的**服务化封装**，
用作 [semif-gate](https://github.com/Tania-X/semif-gate)（Java 决策网关）的推理后端。

> 本仓库**不重新实现** SemIf 的语义——它复用 `semif_phase1` 包的 prompt 构造、
> 答案槽位校验与模型加载，只补上一层 HTTP 契约。

## 验证状态

已在 **RTX 4090 PLUS + Qwen3.5-4B (bf16)** 上与 Java 网关完成端到端联调：

| 项 | 结果 |
|---|---|
| prompt 哈希与 SemIf 已发布值 | ✅ 一致（`3cc9e3d1…`） |
| 输出分布与已发布预测 | ✅ **逐位一致**（Δ ≤ 2e-7） |
| 冷启动延迟 | p50 **95 ms** / p95 149 ms（含网关与 HTTP 往返） |
| 缓存命中 | **0.07 ms**（由网关侧缓存提供，服务零调用） |

**联调过程中发现并修掉的静默 bug**（详见网关仓库的
[`docs/integration-log-silent-bugs.md`](https://github.com/Tania-X/semif-gate/blob/main/docs/integration-log-silent-bugs.md)）：

1. **标签顺序贴错** —— 数值按渲染顺序算、标签按注册表顺序贴，
   导致 `contradicted` 与 `supported` 的概率互换（Δ=0.108），
   而 **prompt 哈希校验照样通过**。
2. **前缀复用给出错误分布** —— Qwen3.5 是混合架构（线性注意力 + 因果卷积），
   其缓存不能按任意 token 边界切分复用（Δ=0.43）。**故默认禁用**。

---

## 职责边界（刻意做得极窄）

| 谁 | 负责什么 |
|---|---|
| **本服务（Python）** | 只做一件事：`(state, 冻结判定点) → 选项概率分布` |
| **调用方（Java 网关）** | 阈值策略、幂等、缓存、审计、降级、成本统计、漂移门禁 |

本服务**刻意不做**的事：

- ❌ 不接受调用方传任意 prompt（只认 `point_ref`，并校验渲染出的哈希）
- ❌ 不做阈值判断，不返回「动作」
- ❌ 不做缓存（缓存键由网关掌握，且需要跨 provider 一致）
- ❌ 不做重试与降级（失败就如实报错）

理由：这些全是**策略**，属于调用方。服务只提供**事实**。

---

## 快速开始

### 1. 前置：安装 SemIf

本服务依赖 SemIf 的 `semif_phase1` 包（按官方说明安装）：

```bash
git clone https://github.com/TheoLeeCJ/SemIf.git
cd SemIf
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
```

模型权重（`Qwen/Qwen3.5-4B`）可用官方方式下载，或用 ModelScope 加速：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-4B'))"
```

### 2. 安装本服务

```bash
pip install -r requirements.txt
```

### 3. 启动

```bash
python server.py \
  --model /path/to/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --port 8080
```

`--revision` 对本地目录是**溯源标签**，会写进每条响应的 provenance。

**启动时会做答案槽位自检**，任一条不过就拒绝启动：

```
[semif-gate] 已加载 3 个判定点
[semif-gate] 执行形状: full-forward（默认）
[semif-gate] 答案槽位自检通过：
  semif.candidate_selection@1        3 选项  slots=[32, 33, 34]
  semif.evidence_interpretation@1    3 选项  slots=[32, 33, 34]
  semif.rule_application@1           3 选项  slots=[32, 33, 34]
```

它断言的是：每个选项字母必须是**恰好一个 token**且**解码后往返一致**。
这是 SemIf 语义的地基——不满足就宁可起不来，也不要在运行时静默返回错位的概率。

---

## API

### `GET /healthz`

返回服务状态与各判定点的答案槽位映射。

### `POST /render` — 冻结 prompt 哈希（不推理）

调用方用它**预计算并冻结** `prompt_sha256`，之后每次 `/decide` 都会校验。

```jsonc
// 请求
{
  "state": {
    "criterion": "May copies be sold?",
    "evidence": "The architect forbids selling copies.",
    "option_order": ["insufficient", "prohibited", "permitted"],
    "case_id": "demo-1"
  },
  "points": [{"id": "semif.rule_application", "point_ref": "semif.rule_application@1",
              "prompt_sha256": "0000…"}]
}
// 响应
[{"point_id": "semif.rule_application", "prompt_sha256": "182c69c2…", "input_tokens": 150}]
```

### `POST /decide` — 判定

```jsonc
// 请求
{
  "state_hash": "9f2c…",          // 由调用方计算并保留，仅回显
  "state": { …同上… },
  "points": [{"id": "semif.rule_application", "point_ref": "semif.rule_application@1",
              "prompt_sha256": "182c69c2…"}]
}
// 响应
{
  "results": [{
    "point_id": "semif.rule_application",
    "option_ids": ["insufficient", "prohibited", "permitted"],
    "probabilities": [0.474969, 0.474969, 0.050061],
    "prompt_sha256": "182c69c2…",
    "input_tokens": 150
  }],
  "provenance": {
    "provider_id": "semif-py", "model_revision": "851bf6e8…",
    "execution_shape": "full-forward", "points_scored": 1,
    "prefix_tokens": 0, "latency_ms": 412, "state_hash": "9f2c…"
  }
}
```

**`prompt_sha256` 不匹配时返回 `409`**，而不是返回一个可能错位的分布。

---

## state 契约

| 字段 | 必填 | 说明 |
|---|---|---|
| `criterion` | ✅ | 判据（自然语言问题），逐行提供 |
| `evidence` | ✅ | 证据（任意 JSON），逐行提供 |
| `option_order` | ⬜ | 选项的呈现顺序；**省略则用注册表冻结顺序** |
| `case_id` | ⬜ | 仅用于日志与 prompt 内的行标识 |

### `option_order` 为什么存在

**选项顺序是语义的一部分，不是显示细节。**

SemIf 的答案字母按选项**下标**分配（`LETTERS[i]`），所以顺序一变，
字母↔选项映射就变、prompt 文本就变、模型行为就变。实测证据：

| 变化 | 对判定翻转率的影响 |
|---|---:|
| 调换选项顺序 | **30.6%** |
| 换 GPU 架构（3090 → 4090） | 0.7% |
| 换执行模式（全量前向 → 前缀复用） | 0% |

**顺序对判定的影响比换一张显卡大 40 倍。**

若省略 `option_order`，服务使用注册表冻结的顺序——此时**只有该顺序恰好匹配的评据行**
才能通过哈希校验。要完整复现 SemIf 的评测集，必须逐行提供 `option_order`
（其每族有 6 种不同的顺序）。

---

## 注册表

`registry/decision-points.json` 声明三类判定点。选项描述**逐字取自 SemIf 的冻结评测集**，
差一个字符 prompt 哈希就对不上。

```jsonc
{
  "id": "semif.rule_application",
  "version": 1,
  "options": [
    {"id": "insufficient", "description": "The supplied information does not settle whether the rule permits the action"},
    {"id": "prohibited",   "description": "The stated rule prohibits the action"},
    {"id": "permitted",    "description": "The stated rule permits the action"}
  ]
}
```

> ⚠️ 注意：`insufficient` 的描述在 `evidence_interpretation` 与 `rule_application`
> 两族里**措辞不同**；`candidate_selection` 的 `insufficient` 结尾是
> "requested **evidence**"。不要凭印象改写。

---

## 客户端

```bash
# 渲染并打印哈希（先用它冻结契约）
python client.py render --url http://127.0.0.1:8080 \
  --criterion "May copies be sold?" \
  --evidence "The architect forbids selling copies." \
  --order insufficient,prohibited,permitted \
  --point semif.rule_application@1

# 直接判定（内部先 render 再 decide，校验哈希一致）
python client.py decide --url http://127.0.0.1:8080 \
  --criterion "May copies be sold?" \
  --evidence "The architect forbids selling copies." \
  --order insufficient,prohibited,permitted \
  --point semif.rule_application@1
```

---

## 执行形状：`--prefix-reuse` 是实验特性，默认关闭

默认走**全量前向**（每个判定点独立一次前向），实测在真实数据上与 SemIf 的
已发布预测**逐位一致（Δ = 0.00e+00）**。

`--prefix-reuse` 让共享前缀只 prefill 一次。**实测在 Qwen3.5 上会给出错误分布**
（`contradicted` 与 `supported` 几近对调，Δ = 0.43）。

原因：Qwen3.5 是**混合架构**（线性注意力 + 因果卷积），其原生缓存不能像标准
Transformer 那样按任意 token 边界切分复用。SemIf 的 `docs/MLX.md` 也警告过这一点。

该开关保留是为了与 SemIf 的 serial/shared 模式做受控对比。
**任何使用它的实验都必须同时跑全量前向做对照。**

---

## 已知限制

- **单进程、单模型、无并发优化**。`/decide` 是同步的，长 state 会占住 worker。
- **无鉴权**。设计为在网关背后的内网/本机运行，不要直接暴露公网。
- **未做批处理**：多个 state 需要多次调用。
- `--prefix-reuse` 不可用于生产（见上）。

---

## 相关仓库

- [SemIf](https://github.com/TheoLeeCJ/SemIf) — 语义与评测方法的来源
- semif-gate — Java 决策网关（缓存、审计、策略、漂移门禁）

## 许可

MIT
