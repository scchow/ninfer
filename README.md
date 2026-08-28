# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen checkpoints on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

NInfer supports five artifact identities. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

The artifact identity fixes the exact model and weight profile. Every artifact also embeds the
tokenizer, chat template, and media frontend resources required by its registered target.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, CUDA Toolkit 13.1 or newer, CMake 3.28 or
newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, and `libswscale >= 7`), and
`libcurl >= 7.85`. The build rejects CUDA architectures other than `sm_120a`.

Build the two product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests, benchmarks, and maintainer tools are excluded from the default build. There is no install
target or packaged binary distribution; run NInfer from its source build tree.

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Loading progress, reasoning, timings, throughput, memory, and
speculative-decoding statistics are written to stderr. Use `--messages FILE` and `--vision` for
structured image/video input; see the [CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Chat-style support (`--chat-style`)

This fork adds `--chat-style default|sharp-v22.1` to the CLI and server, implementing a
[Sharp v22.1](https://huggingface.co/froggeric/Qwen3.8-27B-Instruct-v22.1-Q6_K.gguf) overlay as a
**compiled-C++ prompt modifier** rather than a Jinja template swap.

### What it does

The overlay appends Sharp's terseness instruction to the effective system content and changes
reasoning-effort defaults:
- **System content**: Sharp's `_terse` instruction is appended to any existing system/developer
  message, or becomes the system block itself when none is provided.
- **Reasoning effort**: Default effort shifts from `XHigh` (stock) to `Medium` (Sharp). Six
  discrete levels (`none`→`minimal`→`low`→`medium`→`high`→`xhigh`) are exposed; `none`
  disables thinking; `minimal`/`low` map to concise reasoning, `high`/`xhigh` to thorough.
- **Thinking-history rendering**: Historical turns without reasoning content omit the empty
  thinking wrapper, matching Sharp's `{% if message.reasoning_content or message.reasoning %}`.

All other rendering (tool format, per-turn formatting, generation prompt) is unchanged from stock
NInfer. The official `.ninfer` artifact's embedded template hash is not modified — the same
artifact loads with either style.

### Token-efficiency results (RTX 5090, Qwen3.8-27B NVFP4, INT8 KV)

| Metric | Value |
|--------|-------:|
| **Median completion-token reduction** (24 tasks, MTP3, medium effort) | **−42.2%** |
| Range (24 tasks) | −70.3% to +2.4% |
| **Median wall-time reduction** | **−22.6%** |
| **Raw decode TPS change** (median) | **−0.0%** |
| MTP acceptance Δ (mean) | −0.0 pp |

**Methodology note.** The A/B is matched-effort: each prompt runs stock-vs-sharp at the
same `reasoning_effort`, same seed, alternating server order. The headline median (−42%,
medium effort) reflects the template's terseness instruction plus Sharp's medium default;
at explicitly requested xhigh on reasoning-heavy prompts the gap narrows but persists
(−41.6% on the hard subset). Numbers are from non-tool single-turn requests; agentic
tool-loop behavior depends on the client harness.

Sharp produces equivalent-quality answers on objective checks (math reasoning, JSON structure, code
correctness). The effect is prompt-driven (terse system instruction), not MTP-dependent — MTP0
shows the same 31–64% token reduction with similar TPS.

**MTP3 breakdown by reasoning effort** (matched config, system message present):

| Effort | Stock tokens | Sharp tokens | Token reduction | Wall reduction | Raw TPS (stock/sharp) |
|--------|------------:|------------:|----------------:|---------------:|----------------------:|
| medium | 1,184 | 609 | −48.6% | −26.8% | 142.6 / 142.1 |
| high | 1,084 | 511 | −52.9% | −41.5% | 130.3 / 129.6 |
| xhigh | 1,084 | 511 | −52.9% | −34.6% | 131.6 / 129.6 |

**Hard-subset token reduction** (out-of-distribution difficulty, MTP3): −7% to −59% (analytic
−58.7%, coding −33.5%).

### Usage

```bash
# CLI
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --chat-style sharp-v22.1 \
  --reasoning-effort medium \
  --messages chat.json

# Server (all requests use the configured style)
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --chat-style sharp-v22.1 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The flag is server-wide; individual per-request override is not currently supported.

### Reproducing these results

To reproduce the byte-exact system-block validation and token-efficiency benchmark from scratch:

```bash
# 1. Clone and build
git clone https://github.com/mr-september/ninfer-sharp.git
cd ninfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Download a Qwen3.8-27B NVFP4 model artifact
pip install huggingface-hub
hf download unsloth/Qwen3.8-27B-NVFP4 \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# 3. Download the Sharp v22.1 template for oracle validation
pip install jinja2
wget -O work/chat_template.jinja \
  https://huggingface.co/froggeric/Qwen3.8-27B-Instruct-v22.1-Q6_K.gguf/raw/main/chat_template.jinja

# 4. Build and run the Sharp overlay unit test
#    (compares system-block output against the real Sharp Jinja template)
cmake --build build --target ninfer_qwen3_6_chat_template_sharp_test
./build/tests/ninfer_qwen3_6_chat_template_sharp_test

# 5. Run a quick manual smoke test
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --chat-style sharp-v22.1 \
  --max-context 131072 --kv-dtype int8 --no-prefix-reuse \
  --greedy --seed 42 --default-max-tokens 256

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "You are a precise assistant."},
      {"role": "user", "content": "Explain virtual memory in one sentence."}
    ],
    "max_tokens": 256
  }'
```

A full A/B benchmark requires the task suite and analysis scripts in `work/` (included in the fork
at `work/ab_harness2.py`, `work/ab_analyze2.py`). Example run:

```bash
# Start two servers (default and sharp) on different ports
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --port 8100 --chat-style default --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-context 131072 --kv-dtype int8 --greedy --seed 42

./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --port 8101 --chat-style sharp-v22.1 --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-context 131072 --kv-dtype int8 --greedy --seed 42

# Run the benchmark harness
pip install httpx
python3 work/ab_harness2.py --default http://127.0.0.1:8100 --sharp http://127.0.0.1:8101

# Analyze results
python3 work/ab_analyze2.py
```

### Why not replace the Jinja template directly

The simplest-appearing approach — swap the Jinja template inside the `.ninfer` artifact and
repack — does not work with NInfer's architecture. NInfer does not interpret Jinja at runtime.
Its `chat_template.cpp` computes `sha256(template_source)` and matches it against a hardcoded
constexpr allowlist of exactly two digests: one for ThinkingToggle semantics and one for
ReasoningEffort. A third-party template (Sharp's SHA-256 `d1f22a89…`) is rejected at load.
Patching the allowlist would make the artifact loadable, but the runtime still invokes the
**compiled C++ renderer**, not the Jinja template — so Sharp's `reasoning_effort`, terse
instruction, tool formatting, and disabled-thinking behavior would remain inert.

Repacking Sharp's Jinja into the artifact would be the "correct" surface-level fix but is
**insufficient** without either:

- embedding a Jinja interpreter (heavy, fragile, defeats the purpose of a compiled renderer); or
- rewriting the C++ renderer to accommodate Sharp's semantics — which is what this `--chat-style`
  overlay already does.

The overlay approach (compiled C++ modifier on the existing renderer) was chosen because it:

- leaves the official artifact untouched (no digest repack);
- adds zero runtime overhead (a ~300-byte prompt append + enum dispatch);
- lets stock and Sharp share one code-path, reducing divergence surface area;
- exposes the configuration through a straightforward CLI flag.

### Limitations

1. **System-message dependency**: the overlay synthesizes a system block when none exists, but
   clients that intentionally suppress system content (transparent proxy layers) will not observe
   Sharp's effect. This matches Sharp's own Jinja behavior.
2. **Tool-formatting is NInfer-native, not Sharp-v22.1-format**: This is the main reason the
   overlay is recommended for **non-tool chat** when byte-exact Sharp compatibility matters.
   The overlay does not change NInfer's `<tool_call>` XML format to match Sharp's Jinja
   `_render_tool_schema` (`# Tools` header, `⚠️` error-escalation warnings, different JSON
   argument rendering). For tool-using clients, the rendered prompt is therefore stock NInfer's
   tool format with the Sharp terse overlay — not Sharp's exact template output. If your
   workflow does not use tool calling, or does not require byte-exact Sharp tool-format
   compatibility, the overlay's token savings still apply. Tool output is identical between
   `--chat-style default` and `--chat-style sharp-v22.1`.
3. **Server-wide style**: `--chat-style` is set at process startup and applies to all requests.
   Per-request override is not implemented.
4. **Qwen3.6/3.8 only**: the overlay targets the Qwen3.6/3.8 ReasoningEffort renderer (the C++
   `CompiledChatTemplate`). Other model families (future targets using ThinkingToggle or other
   semantics) are unaffected.

## Performance

Published measurements use an RTX 5090. [Performance](docs/performance.md) records the exact
benchmark profiles and methodology.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Values are aggregate committed decode throughput and MTP acceptance from complete
intervals whose actual decode batch equaled the configured concurrency.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
in the performance document.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B `groupwise-int` | 15,544.3 tok/s | 5,157.1 tok/s | 770.9 tok/s |
| Qwen3.6-27B `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| Qwen3.6-27B `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| Qwen3.8-27B `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Artifact and startup notes

Current builds accept only version-2 `.ninfer` containers. All five published downloads are version
2. Migration is needed only for Qwen3.6 artifacts downloaded before their version-2 publication:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with the exact older Qwen3.6 NVFP4 or 35B-A3B file. Migration updates container
metadata without rewriting the weight payload.

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` selects Vision residency. DFlash is available for text-only Qwen3.6-35B-A3B execution.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8 group-64, and row-scaled FP8 E4M3 KV storage;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports text-only DFlash with draft windows from one to fifteen.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run `./build/apps/ninfer --help` or `./build/apps/ninfer-serve --help` for the exact current option
contract.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
