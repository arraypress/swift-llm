# LLMKit

One Swift API for language models across every tier — **Apple's on-device model, a local MLX model, or any cloud endpoint** — for macOS and iOS. Program against one protocol; the model (and the tier: private/free · local · most-capable) becomes a swappable choice, not a rewrite.

Text · vision · structured output, behind a single `LLMEngine`.

## Engines

| Engine | Tier | What it is |
|---|---|---|
| **`FoundationEngine`** | on-device, free | Apple's system model (FoundationModels) — no download, private |
| **`RemoteEngine`** | cloud | **any OpenAI-compatible endpoint** — OpenAI, Claude (via proxy), Cloudflare AI Gateway, Groq, OpenRouter, or a local Ollama / LM Studio server |
| **`MLXLLMEngine`** | on-device, downloaded | Qwen3 / Phi-4-mini / SmolLM3 / Mistral / Llama via MLX — downloaded & cached |

## Usage

**Cloud (OpenAI-compatible):**
```swift
import LLMKit

let llm = LLM(RemoteEngine(model: "gpt-4o", endpoint: .openAI, apiKey: key))
let reply = try await llm.generate("Summarize this in one line: …")
```
Works with any provider by swapping the endpoint — `.groq`, `.openRouter`, `.localServer()`, or `.cloudflareGateway(accountID:gateway:provider:)`.

**On-device (Apple):**
```swift
let llm = LLM(FoundationEngine(instructions: "You extract structured data."))
let reply = try await llm.generate("…")   // free, private, no download
```

**On-device (local MLX model):**
```swift
import LLMKit
import LLMKitMLX

let engine = MLXLLMEngine(.qwen3_0_6B)                     // 4-bit, ~450 MB, Apache-2.0
try await engine.prepare { print("downloading \(Int($0 * 100))%") }   // first run downloads + caches
let reply = try await LLM(engine).generate("Explain a haiku in one line.")
```

**Vision** (RemoteEngine today; FoundationModels on newer OS):
```swift
let jpeg = image.jpegData(compressionQuality: 0.8)!
let text = try await llm.respond(to: [.user("What is this?", images: [jpeg])])
```

**Structured output** — parse the reply into a `Decodable`:
```swift
struct Expense: Decodable { let title: String; let amount: String }
let e: Expense = try await llm.respond(
    to: [.user("Return JSON {title, amount} for: coffee £3.50")],
    generating: Expense.self)
```
`JSONExtractor` pulls the first balanced JSON object out of the reply (handling prose and ```json fences), so this works across engines.

## Why one library
Your apps already span all three tiers — Apple FoundationModels (ReceiptBunny, RenameMaid), and OpenAI-compatible cloud (RenameMaid, SupplementScan's Cloudflare AI Gateway). LLMKit unifies them: each app picks the tier it wants, per request, and gains the tiers it was missing — plus a fallback path (on-device → local → cloud).

## Status
- ✅ **Core + `RemoteEngine` + `FoundationEngine` — built and unit-tested (19 tests).** RemoteEngine's request/response shaping and the JSON extractor are covered by pure tests; the cloud path is standard URLSession; the Apple path compiles against the real FoundationModels SDK (its *live* generation needs Apple Intelligence enabled).
- ✅ **`MLXLLMEngine` — implemented and verified on device.** Local downloaded models via `mlx-swift-lm` (Qwen3 · Phi-4-mini · SmolLM3 · Mistral Small 3 · Llama 3.x). End-to-end smoke test (`llm-run`) confirmed: Qwen3-0.6B downloaded + loaded in ~34s, generated in ~11s on the Apple Silicon GPU. See [Verifying the MLX engine](#verifying-the-mlx-engine).
- 🔜 **Image input for `FoundationEngine`** — text-only on this SDK; maps to `.image()` on the newer OS.

## Requirements
- macOS 26+ / iOS 26+ · Swift 6.2+
- `RemoteEngine` needs an API key (except local servers); `FoundationEngine` needs Apple Intelligence.

## Testing
```bash
swift test          # 19 offline unit tests (core: RemoteEngine shaping + JSONExtractor)
```

## Verifying the MLX engine
`swift test` can't exercise `MLXLLMEngine` — mlx-swift needs a Metal library that only ships in Xcode app builds. The `llm-run` smoke test proves it on device: it downloads a small model and runs one generation.
```bash
swift build --product llm-run
# colocate a Metal library next to the binary (see CLAUDE.md for a source):
cp <some>/mlx.metallib .build/arm64-apple-macosx/debug/mlx.metallib
.build/arm64-apple-macosx/debug/llm-run qwen3-0.6b "In one sentence, what is a haiku?"
```
Xcode app builds bundle the metallib automatically.

## License
MIT — see LICENSE.

## Author
Created by David Sherlock ([ArrayPress](https://github.com/arraypress)) in 2026.
