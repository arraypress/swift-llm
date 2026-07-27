# CLAUDE.md — swift-llm (LLMKit)

One Swift API for LLMs across tiers — Apple on-device, local MLX (text **and** vision), or any OpenAI-compatible cloud — behind a single `LLMEngine` protocol. macOS/iOS. Same shape as swift-stable-audio / swift-tts. Companion chat app: `../../swift-llm-tester` (LLMChat).

## Why this exists
To unify the AI code already scattered across the apps: **ReceiptBunny** and **RenameMaid** use Apple FoundationModels with `@Generable`; **SupplementScan** has its own `AIService` hitting Cloudflare AI Gateway (OpenAI-compatible vision + JSON). Each app should pick a tier, gain the tiers it lacked, and get a fallback chain — rather than three private implementations.

## Build & test
```bash
swift build                      # all targets (LLMKitMLX pulls mlx-swift-lm — slow first build)
swift test                       # 25 offline unit tests (JSONExtractor, RemoteEngine shaping, reasoning split)
swift build --product llm-run    # on-device MLX smoke test (use --product, NOT --target — target compiles but won't link)
```

## Module map
- `LLMKit/` (core, dependency-light) — `LLMEngine` protocol, `LLM` facade, `LLMMessage` (carries `images: [Data]`), `GenerationOptions`, `LLMError`, `JSONExtractor` (first balanced JSON object → structured output), **`Reasoning.swift`** (`String.splitReasoning()` → `(reasoning?, answer)` + `strippingReasoning()`; handles `<think>/<thinking>/<reasoning>`), plus:
  - `FoundationEngine` — Apple FoundationModels (on-device; live gen needs Apple Intelligence). **Text only, deliberately:** this machine's SDK is macOS 26.5 (WWDC25 FoundationModels), where the API has no image input. `.image()` arrived at WWDC26 and needs the macOS 27 SDK — wire it then, don't try to work around it now.
  - `RemoteEngine` — any OpenAI-compatible endpoint.
- `LLMKitMLX/` (opt-in) — `MLXLLMEngine` + `MLXModel` registry. Links MLXLLM **and MLXVLM**.

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
- Commit only when asked; branch off `main` first if needed.
- Downloaded weights live in `~/.cache/huggingface/hub`, never in git.
