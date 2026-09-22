# CLAUDE.md — swift-llm (LLMKit)

One Swift API for LLMs across tiers — Apple on-device, local MLX (text **and** vision), Anthropic's native Messages API, or any OpenAI-compatible cloud — behind a single `LLMEngine` protocol. macOS/iOS. Same shape as swift-stable-audio / swift-tts. Companion chat app: `../../swift-llm-tester` (LLMChat).

## Why this exists
To unify the AI code already scattered across the apps: **ReceiptBunny** and **RenameMaid** use Apple FoundationModels with `@Generable`; **SupplementScan** has its own `AIService` hitting Cloudflare AI Gateway (OpenAI-compatible vision + JSON); **FolderMaid/Scaffolder** hand-rolled Claude + OpenAI request code with a hard-coded model list. Each app should pick a tier, gain the tiers it lacked, and get a fallback chain — rather than four private implementations.

## Build & test
```bash
swift build --target LLMKit      # core only, seconds (a bare `swift build` pulls mlx-swift-lm — slow first build)
swift test                       # 83 offline unit tests; AnthropicLiveTests skip without a key
swift test --filter AnthropicLiveTests   # 14 real calls on Haiku 4.5, < 1¢; key from ANTHROPIC_API_KEY or
                                         # credentials.anthropic.apiKey in ~/.config/arraypress/credentials.json
swift build --product llm-run    # on-device MLX smoke test (use --product, NOT --target — target compiles but won't link)
```

## Module map
- `LLMKit/` (core, dependency-light) — `LLMEngine` protocol, `LLM` facade, `LLMMessage` (`images: [Data]` + `attachments: [LLMAttachment]`, merged by `allAttachments`), `LLMAttachment` (image / imageURL / pdf / pdfURL / text / file), `GenerationOptions`, `LLMError` (incl. `.refused`), `ModelListing` + `LLMModelInfo`, `JSONExtractor` (first balanced JSON object → structured output), **`Reasoning.swift`** (`String.splitReasoning()` → `(reasoning?, answer)` + `strippingReasoning()`; handles `<think>/<thinking>/<reasoning>` and gpt-oss harmony), plus:
  - `FoundationEngine` — Apple FoundationModels (on-device; live gen needs Apple Intelligence). Streams by diffing the session's cumulative snapshots. **Text only, deliberately:** `.image()` needs the macOS 27 SDK — wire it then. Text attachments are inlined into the prompt; others throw.
  - `RemoteEngine` + `RemoteEndpoint` — any OpenAI-compatible endpoint. SSE streaming, `GET /models`, `maxTokensField` per dialect (`max_completion_tokens` for OpenAI/Groq — `gpt-5` rejects `max_tokens`). Images inline or by URL; text documents inlined; PDFs/files throw (`checkAttachments`).
  - `Anthropic/` — `AnthropicEngine` (neutral `LLMEngine` door + typed `send` / `run` / `streamEvents` / `countTokens` / Files API), `AnthropicTypes` (configuration, thinking, effort, tool choice, response, usage, citations), `AnthropicTool` (client tools with optional handler; server tools), `AnthropicRequestBuilder` (pure body/turn/header assembly), `AnthropicStream` (event enum + accumulator that rebuilds the full response), `AnthropicFiles`.
  - `Support/` — `HTTPTransport` (the one URLSession seam: JSON, multipart, streaming lines, provider error text), `ServerSentEvents`, `JSONValue` (Sendable JSON tree; tool inputs, schemas, raw blocks), `ImageData` (media type sniff), `MultipartForm`, `ISO8601` (with/without fractional seconds).
- `LLMKitMLX/` (opt-in) — `MLXLLMEngine` + `MLXModel` registry. Links MLXLLM **and MLXVLM**. Renders `images` only.

## Anthropic rules that are verified, not assumed (2026-09-22, live)
- Sampling (`temperature` / `top_p` / `top_k`) is **rejected with 400 on Opus 4.7, 4.8, 5, Sonnet 5, Fable/Mythos**; `acceptsSamplingParameters(model:)` gates it by id substring, `sendsSamplingParameters:` overrides.
- With **thinking on** (adaptive or budget) the API rejects any `temperature` but 1 — the builder drops sampling entirely when `configuration.thinking?.isEnabled`.
- `GET /models` lists **dated ids** (`claude-haiku-4-5-20251001`); the undated alias works in requests. Match by prefix.
- Files API timestamps carry **microseconds** (`2026-09-22T10:25:51.350741Z`), model timestamps don't — `ISO8601.date(from:)` handles both.
- Structured output (`output_config.format`) returns the JSON in a **text block**; it is incompatible with citations.
- A `refusal` stop reason is an HTTP 200; `respond` throws `LLMError.refused`, `send`/`run` return the response with `stopReason == .refusal`.
- Tool loop: `run` echoes the assistant turn verbatim (`raw["content"]`, thinking signatures included), runs handlers in parallel, sends all results in one user turn, resumes `pause_turn`, and stops on `max_tokens`, a refusal, or a tool without a handler.
- Strict tools compile the schema on first use — the first strict call took ~25 s on Haiku, later ones seconds.

## Engine capabilities (`MLXLLMEngine`)
- **Streaming:** `LLMEngine.streamResponse(to:options:)` (protocol req + default wrapping `respond`); `MLXLLMEngine` overrides with real token streaming via `ChatSession.streamResponse`. `LLM.stream(_:)` on the facade.
- **Vision:** attached `LLMMessage.images` (JPEG/PNG `Data`) are decoded to `CIImage` → `UserInput.Image.ciImage` and passed to the session. Works because linking **MLXVLM** auto-registers its factory (`ModelFactoryRegistry` trampoline via `NSClassFromString`), so `loadModelContainer(from:directory:using:)` loads VLMs on the same path.
- **Download:** `prepare()` fetches with swift-transformers **`HubApi.shared.snapshot`** (reliable + real progress), then loads from the returned dir. ⚠️ **HubApi still STALLS on heavily-SHARDED repos** (e.g. gpt-oss = 3×5GB); pre-fetch those with `.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('<repo>', max_workers=8)"`. Single-file models are fine in-app.

## `MLXModel` registry
Each model has `group` (General / Vision / Uncensored / Creative & NSFW), `maker`, `blurb`, `supportsVision`, size/license. ~24 models — treat that as a **deliberate stopping point**; more entries are diminishing returns, not coverage.

Notes that save a debugging round: **Qwen3 is a reasoning model** and returns its `<think>…</think>` block verbatim — strip it with `splitReasoning()`, or use Phi-4-mini when you want clean chat output. **Gemma 4 vision** (`Gemma4ForConditionalGeneration`) is *not* supported and must be omitted; Gemma 4 **text** is fine. Target machine is an **M3 Max / 36 GB**, which runs everything here up to ~30B including the 35B-A3B MoE.

**Arch gate:** only add models whose `model_type` is registered in the pinned mlx-swift-lm — check `.build/checkouts/mlx-swift-lm/Libraries/MLXLLM/Models/` and `…/MLXVLM/Models/` (one file per arch). Supported incl. qwen3/qwen3_5(_moe)/llama/mistral/phi/gemma2/gemma3/gemma4(text)/gpt_oss; VLMs qwen3_vl/qwen2_5_vl/gemma3/pixtral/smolvlm2. **Verify a candidate is TEXT** (config has no `vision_config`) unless it's a VLM group entry.

## `llm-run` harness
```bash
swift build --product llm-run
cp <mlx.metallib> .build/arm64-apple-macosx/debug/mlx.metallib   # mlx-swift needs a Metal lib; only Xcode APP builds bundle it
.build/arm64-apple-macosx/debug/llm-run <modelID> "<prompt>" [imagePath]
```
Downloads on first run (streams live, splits reasoning). Optional `imagePath` attaches an image to test a VLM. Xcode app builds bundle the metallib automatically.

## Conventions
- Core (`LLMKit`) stays dependency-light and unit-testable offline; anything MLX lives in `LLMKitMLX`.
- Wire shaping lives in pure static helpers (`requestBody`, `parseContent`, `streamDelta`, the request builder, the accumulator) so it is asserted offline; the live suite is the ground truth and skips without a key.
- New API features: read the field names from the Claude docs (`platform.claude.com/docs/en/…`), never from memory, and add a live check before claiming support.
- Commit only when asked; branch off `main` first if needed.
- Downloaded weights live in `~/.cache/huggingface/hub`, never in git.
