# CLAUDE.md — swift-llm (LLMKit)

One Swift API for LLMs across tiers — Apple on-device, local MLX, or any OpenAI-compatible cloud — behind a single `LLMEngine` protocol. macOS/iOS. Same shape as swift-stable-audio / swift-tts.

## Build & test
```bash
swift build                     # all targets (LLMKitMLX pulls mlx-swift-lm — slow first build)
swift test                      # 19 offline unit tests (core: RemoteEngine shaping + JSONExtractor)
swift build --target llm-run    # the on-device MLX smoke test
```

## Module map
- `LLMKit/` (core, zero heavy deps) — `LLMEngine` protocol, `LLM` facade, `LLMMessage`, `GenerationOptions`, `LLMError`, `JSONExtractor` (pulls the first balanced JSON object out of a reply for structured output), plus two engines:
  - `FoundationEngine` — Apple FoundationModels (on-device, free; live gen needs Apple Intelligence).
  - `RemoteEngine` — any OpenAI-compatible endpoint (OpenAI/Groq/OpenRouter/Cloudflare Gateway/local Ollama·LM Studio).
- `LLMKitMLX/` (opt-in) — `MLXLLMEngine` (local models via mlx-swift-lm) + `MLXModel` registry (Qwen3, Phi-4-mini, SmolLM3, Mistral Small 3, Llama 3.x). All presets are 4-bit MLX-community weights.

## Verifying the MLX engine (the `llm-run` harness)
`MLXLLMEngine` can't be exercised by `swift test` — mlx-swift needs a Metal library that only ships in Xcode **app** builds. To prove it on device headless:
```bash
swift build
# colocate a Metal library next to the binary (see the stable-audio venv, or any Xcode MLX app build):
cp <some>/mlx.metallib .build/arm64-apple-macosx/debug/mlx.metallib
.build/arm64-apple-macosx/debug/llm-run qwen3-0.6b "In one sentence, what is a haiku?"
```
`llm-run [modelID] [prompt]` downloads the model on first run (mlx-community, public, no HF auth), loads it on the GPU, and prints one generation + timings. Default model `qwen3-0.6b` (~450 MB) is the quick smoke test. Xcode app builds bundle the metallib automatically — nothing to colocate there.

## Conventions
- Core (`LLMKit`) stays dependency-light and fully unit-testable offline; anything MLX lives in `LLMKitMLX`.
- Commit only when asked; branch off `main` first if needed.
- Downloaded model weights live in the HuggingFace cache, never in git.
