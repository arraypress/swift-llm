# LLMKit

One Swift API for language models across every tier — **Apple's on-device model, a local MLX model, Anthropic's Claude, or any OpenAI-compatible cloud endpoint** — for macOS and iOS. Program against one protocol; the model (and the tier: private/free · local · most-capable) becomes a swappable choice, not a rewrite.

Text · vision · documents · streaming · tools · structured output, behind a single `LLMEngine`.

## Engines

| Engine | Tier | What it is |
|---|---|---|
| **`FoundationEngine`** | on-device, free | Apple's system model (FoundationModels) — no download, private |
| **`AnthropicEngine`** | cloud | Claude via Anthropic's **native Messages API** — images, PDFs, Files API, tools, thinking, structured output, caching |
| **`RemoteEngine`** | cloud | **any OpenAI-compatible endpoint** — OpenAI, Cloudflare AI Gateway, Groq, OpenRouter, or a local Ollama / LM Studio server |
| **`MLXLLMEngine`** | on-device, downloaded | Qwen3 / Phi-4-mini / SmolLM3 / Mistral / Llama via MLX — downloaded & cached |

Every engine streams live tokens (`LLM.stream`), and the cloud engines list their models (`LLM.availableModels()`), so a picker never goes stale.

## Usage

**Claude (native):**
```swift
import LLMKit

let llm = LLM(AnthropicEngine(model: "claude-opus-5", apiKey: key))
let reply = try await llm.generate("Summarize this in one line: …")
for try await token in llm.stream("Write a haiku") { print(token, terminator: "") }
```

**Cloud (OpenAI-compatible):**
```swift
let llm = LLM(RemoteEngine(model: "gpt-5-mini", endpoint: .openAI, apiKey: key))
let reply = try await llm.generate("Summarize this in one line: …")
```
Works with any provider by swapping the endpoint — `.groq`, `.openRouter`, `.localServer()`, or `.cloudflareGateway(accountID:gateway:provider:)`. OpenAI and Groq get `max_completion_tokens` automatically; the rest keep `max_tokens`.

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

**Vision and documents** — attachments ride on the message; each engine renders the kinds it can and throws `LLMError.unavailable` for the rest rather than dropping them:
```swift
let jpeg = image.jpegData(compressionQuality: 0.8)!
let text = try await llm.respond(to: [.user("What is this?", images: [jpeg])])

// Anthropic: PDFs, text documents, URLs and uploaded files too
let answer = try await llm.respond(to: [.user("Summarise the contract.", attachments: [
    .pdf(contractData, title: "Contract"),
    .imageURL(URL(string: "https://example.com/chart.png")!),
    .text(notes, title: "Notes"),
])])
```

**Structured output** — parse the reply into a `Decodable`:
```swift
struct Expense: Decodable { let title: String; let amount: String }
let e: Expense = try await llm.respond(
    to: [.user("Return JSON {title, amount} for: coffee £3.50")],
    generating: Expense.self)
```
`JSONExtractor` pulls the first balanced JSON object out of the reply (handling prose and ```json fences), so this works across engines. On Anthropic, set `AnthropicConfiguration.outputSchema` and the API guarantees the JSON matches your schema.

**Model listing:**
```swift
let models = try await llm.availableModels()   // newest first; Anthropic + OpenAI-compatible
```

## Claude in depth

`AnthropicEngine` has two doors. The neutral `LLMEngine` calls return text; the typed calls return everything the API said.

```swift
var config = AnthropicConfiguration()
config.effort = .low                              // output_config.effort
config.thinking = .adaptive(display: .summarized) // or .budget(tokens:) on Haiku 4.5
config.cacheSystemPrompt = true                   // cache_control on the system prompt
config.stopSequences = ["END"]
config.citations = true                           // documents come back cited
config.tools = [AnthropicTool(
    name: "get_weather",
    description: "Call this when the user asks about current weather.",
    inputSchema: ["type": "object", "properties": ["city": ["type": "string"]],
                  "required": ["city"], "additionalProperties": false],
    strict: true
) { input in try await weather(for: input["city"]?.stringValue ?? "") }]
config.serverTools = [.webSearch(maxUses: 3)]

let claude = AnthropicEngine(model: "claude-opus-5", apiKey: key, configuration: config)

let response = try await claude.run([.user("Is it raining in Oslo?")])   // drives the tool loop
print(response.text, response.stopReason, response.usage.outputTokens, response.citations)

for try await event in claude.streamEvents([.user("…")]) {             // typed stream
    switch event {
    case .textDelta(let t): print(t, terminator: "")
    case .toolUseStart(_, _, let name): print("calling \(name)")
    case .completed(let final): print(final.usage)
    default: break
    }
}

let count = try await claude.countTokens([.user("…")])                  // free, before sending
let file = try await claude.uploadFile(pdfData, filename: "report.pdf", mimeType: "application/pdf")
let reply = try await claude.send([.user("Summarise it.", attachments: [.file(id: file.id, kind: .document)])])
try await claude.deleteFile(id: file.id)
```

Things the engine knows so you don't have to: sampling parameters are omitted on models that reject them (Opus 4.7 onward) and whenever thinking is on; `.system` messages become the top-level `system`; consecutive same-role messages merge into one turn; a `refusal` stop reason surfaces as `LLMError.refused`; `pause_turn` from server tools is resumed automatically.

## Why one library
Your apps already span all three tiers — Apple FoundationModels (ReceiptBunny, RenameMaid), and OpenAI-compatible cloud (RenameMaid, SupplementScan's Cloudflare AI Gateway). LLMKit unifies them: each app picks the tier it wants, per request, and gains the tiers it was missing — plus a fallback path (on-device → local → cloud).

## Status
- ✅ **Core + `RemoteEngine` + `FoundationEngine` + `AnthropicEngine` — built and unit-tested (83 offline tests).** Request/response shaping, SSE streaming, the stream accumulator, attachments, the JSON extractor, multipart uploads and model listing are pure functions with tests; the cloud paths are plain URLSession.
- ✅ **`AnthropicEngine` verified against the live API (2026-09-22, Haiku 4.5):** model listing, text with usage, streamed deltas assembling to the final reply, stop sequences, an inline image, a text document with citations, a CoreGraphics-generated PDF, Files API upload → use → metadata → delete, a strict tool loop, JSON-schema structured output, `count_tokens`, a thinking budget, and a bad key surfacing the provider's 401 message — 14 checks in ~40 s for well under a cent. Two rules were learned from the API rather than the docs: thinking forbids any `temperature` but 1, and file timestamps carry microseconds where model timestamps don't.
- ✅ **`MLXLLMEngine` — implemented and verified on device.** Local downloaded models via `mlx-swift-lm` (Qwen3 · Phi-4-mini · SmolLM3 · Mistral Small 3 · Llama 3.x). End-to-end smoke test (`llm-run`) confirmed: Qwen3-0.6B downloaded + loaded in ~34s, generated in ~11s on the Apple Silicon GPU. See [Verifying the MLX engine](#verifying-the-mlx-engine).
- 🔜 **Image input for `FoundationEngine`** — text-only on this SDK; maps to `.image()` on the newer OS. `MLXLLMEngine` renders `images` only; other attachment kinds are for the cloud engines.
- ⏭ Not covered, on purpose: Message Batches, compaction and context editing, MCP connector, Skills and Managed Agents. Pass any beta id through `AnthropicConfiguration.betas` if you need to reach one by hand.

## Requirements
- macOS 26+ / iOS 26+ · Swift 6.2+
- `AnthropicEngine` and `RemoteEngine` need an API key (except local servers); `FoundationEngine` needs Apple Intelligence.

## Testing
```bash
swift test                              # 83 offline unit tests; the live suite skips without a key
swift test --filter AnthropicLiveTests  # 14 real calls on Haiku 4.5 (ANTHROPIC_API_KEY, or credentials.anthropic.apiKey
                                        # in ~/.config/arraypress/credentials.json); costs well under a cent
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
