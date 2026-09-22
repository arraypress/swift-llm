//
//  AnthropicTypes.swift
//  LLMKit
//
//  The request knobs and response shapes of Anthropic's Messages API that have
//  no counterpart in the engine-neutral `GenerationOptions` / `String` reply:
//  thinking and effort, stop sequences, caching, structured output, service
//  tier, fallbacks, and the typed response with its content blocks and usage.
//

import Foundation

// MARK: - Configuration

/// Anthropic-specific request settings, fixed on the engine so the neutral
/// `LLMEngine` calls stay neutral. Engines are cheap structs — make one per
/// configuration rather than mutating.
public struct AnthropicConfiguration: Sendable {

    /// Extended thinking. `nil` leaves the provider default (adaptive on
    /// Opus 5 and later, off on Opus 4.7 / 4.8, none on Haiku 4.5).
    public var thinking: AnthropicThinking?

    /// `output_config.effort` — how hard the model works; `nil` is `high`.
    public var effort: AnthropicEffort?

    /// Custom sequences that end generation (`stop_sequences`).
    public var stopSequences: [String] = []

    /// `metadata.user_id` — an opaque end-user id for abuse detection, never
    /// personal data. At most 512 characters.
    public var userID: String?

    /// `top_k` sampling, sent only when sampling parameters are sent at all.
    public var topK: Int?

    /// A JSON Schema the reply must match (`output_config.format`). The reply
    /// arrives as JSON text in a text block; every object in the schema needs
    /// `additionalProperties: false`.
    public var outputSchema: JSONValue?

    /// Client tools the model may call. Tools with a `handler` are run by
    /// `AnthropicEngine.run`; tools without one are returned to the caller.
    public var tools: [AnthropicTool] = []

    /// Anthropic-hosted tools (web search, web fetch, code execution).
    public var serverTools: [AnthropicServerTool] = []

    /// How the model chooses among tools; `nil` is `auto`.
    public var toolChoice: AnthropicToolChoice?

    /// Ask for citations on every document attachment. Incompatible with
    /// `outputSchema` (the API returns 400).
    public var citations: Bool = false

    /// Mark the system prompt as a cache breakpoint (`cache_control`), so a
    /// long, stable system prompt is billed at cache-read rates after the
    /// first call. Short prompts silently don't cache.
    public var cacheSystemPrompt: Bool = false

    /// How long a cache entry lives.
    public var cacheTTL: AnthropicCacheTTL = .fiveMinutes

    /// Priority capacity (`service_tier`).
    public var serviceTier: AnthropicServiceTier?

    /// Pin inference to a region (`inference_geo`, e.g. `"us"`).
    public var inferenceGeo: String?

    /// Re-run a policy-declined request on another model inside the same call.
    public var refusalFallbacks: AnthropicRefusalFallbacks?

    /// Extra `anthropic-beta` feature ids.
    public var betas: [String] = []

    /// How many model turns `run` may take before giving up on a tool loop.
    public var maxToolIterations: Int = 10

    /// The defaults: nothing beyond what the API needs.
    public init() {}
}

/// Extended thinking modes.
public enum AnthropicThinking: Sendable, Equatable {

    /// The model decides when and how much to think (Opus 4.6 and later).
    case adaptive(display: Display? = nil)

    /// A fixed budget (`{type: "enabled", budget_tokens}`) for models before
    /// Opus 4.6 such as Haiku 4.5. Minimum 1024, and less than `max_tokens`.
    case budget(tokens: Int, display: Display? = nil)

    /// Thinking off. Rejected at effort `xhigh` / `max` on Opus 5.
    case disabled

    /// What the response's `thinking` blocks carry.
    public enum Display: String, Sendable {
        case summarized
        case omitted
        case updates
    }

    /// Whether the model will think. While it does, the API accepts no
    /// sampling parameters (`temperature` may only be 1), so the request
    /// builder drops them.
    public var isEnabled: Bool {
        if case .disabled = self { return false }
        return true
    }

    var json: JSONValue {
        switch self {
        case .adaptive(let display):
            var object: [String: JSONValue] = ["type": "adaptive"]
            if let display { object["display"] = .string(display.rawValue) }
            return .object(object)
        case .budget(let tokens, let display):
            var object: [String: JSONValue] = ["type": "enabled", "budget_tokens": .number(Double(tokens))]
            if let display { object["display"] = .string(display.rawValue) }
            return .object(object)
        case .disabled:
            return ["type": "disabled"]
        }
    }
}

/// `output_config.effort`.
public enum AnthropicEffort: String, Sendable {
    case low, medium, high, xhigh, max
}

/// `tool_choice`.
public enum AnthropicToolChoice: Sendable, Equatable {
    /// The model decides (the default).
    case auto(disableParallelToolUse: Bool = false)
    /// The model must call some tool. Rejected by Claude Fable 5.1.
    case any(disableParallelToolUse: Bool = false)
    /// The model must call this tool. Rejected by Claude Fable 5.1.
    case tool(name: String, disableParallelToolUse: Bool = false)
    /// No tool calls this turn.
    case none

    var json: JSONValue {
        switch self {
        case .auto(let single):
            return single ? ["type": "auto", "disable_parallel_tool_use": true] : ["type": "auto"]
        case .any(let single):
            return single ? ["type": "any", "disable_parallel_tool_use": true] : ["type": "any"]
        case .tool(let name, let single):
            var object: [String: JSONValue] = ["type": "tool", "name": .string(name)]
            if single { object["disable_parallel_tool_use"] = true }
            return .object(object)
        case .none:
            return ["type": "none"]
        }
    }
}

/// Prompt-cache lifetime.
public enum AnthropicCacheTTL: String, Sendable {
    case fiveMinutes = "5m"
    case oneHour = "1h"
}

/// `service_tier`.
public enum AnthropicServiceTier: String, Sendable {
    case auto
    case standardOnly = "standard_only"
}

/// `fallbacks`: what to do when the safety system declines the request.
public enum AnthropicRefusalFallbacks: Sendable, Equatable {
    /// Let Anthropic route by refusal category (`fallbacks: "default"`).
    case `default`
    /// Try these models in order.
    case models([String])

    /// The beta header each form requires; pairing them the other way is a 400.
    var beta: String {
        switch self {
        case .default: return "server-side-fallback-2026-07-01"
        case .models: return "server-side-fallback-2026-06-01"
        }
    }

    var json: JSONValue {
        switch self {
        case .default: return "default"
        case .models(let ids): return .array(ids.map { ["model": .string($0)] })
        }
    }
}

// MARK: - Response

/// Why generation stopped.
public enum AnthropicStopReason: String, Sendable, Equatable {
    case endTurn = "end_turn"
    case maxTokens = "max_tokens"
    case stopSequence = "stop_sequence"
    case toolUse = "tool_use"
    case pauseTurn = "pause_turn"
    case refusal
    /// A value this version of the library doesn't know.
    case unknown

    init(wire value: String?) {
        self = value.flatMap(AnthropicStopReason.init(rawValue:)) ?? .unknown
    }
}

/// Token accounting for one call.
public struct AnthropicUsage: Sendable, Equatable {
    public let inputTokens: Int
    public let outputTokens: Int
    public let cacheCreationInputTokens: Int
    public let cacheReadInputTokens: Int

    public init(inputTokens: Int = 0, outputTokens: Int = 0, cacheCreationInputTokens: Int = 0, cacheReadInputTokens: Int = 0) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.cacheCreationInputTokens = cacheCreationInputTokens
        self.cacheReadInputTokens = cacheReadInputTokens
    }

    static func parse(_ json: JSONValue?) -> AnthropicUsage {
        AnthropicUsage(
            inputTokens: json?["input_tokens"]?.intValue ?? 0,
            outputTokens: json?["output_tokens"]?.intValue ?? 0,
            cacheCreationInputTokens: json?["cache_creation_input_tokens"]?.intValue ?? 0,
            cacheReadInputTokens: json?["cache_read_input_tokens"]?.intValue ?? 0
        )
    }
}

/// What `count_tokens` reports.
public struct AnthropicTokenCount: Sendable, Equatable {
    public let inputTokens: Int
    public let cacheCreationInputTokens: Int
    public let cacheReadInputTokens: Int

    static func parse(_ json: JSONValue) -> AnthropicTokenCount {
        AnthropicTokenCount(
            inputTokens: json["input_tokens"]?.intValue ?? 0,
            cacheCreationInputTokens: json["cache_creation_input_tokens"]?.intValue ?? 0,
            cacheReadInputTokens: json["cache_read_input_tokens"]?.intValue ?? 0
        )
    }
}

/// A citation attached to a text block when documents were sent with
/// `citations` on.
public struct AnthropicCitation: Sendable, Equatable {
    public let citedText: String
    public let documentIndex: Int
    public let documentTitle: String?
    /// 1-indexed pages, for PDF documents.
    public let startPage: Int?
    public let endPage: Int?
    /// Character offsets, for plain-text documents.
    public let startCharIndex: Int?
    public let endCharIndex: Int?

    static func parse(_ json: JSONValue) -> AnthropicCitation? {
        guard let citedText = json["cited_text"]?.stringValue else { return nil }
        return AnthropicCitation(
            citedText: citedText,
            documentIndex: json["document_index"]?.intValue ?? 0,
            documentTitle: json["document_title"]?.stringValue,
            startPage: json["start_page_number"]?.intValue,
            endPage: json["end_page_number"]?.intValue,
            startCharIndex: json["start_char_index"]?.intValue,
            endCharIndex: json["end_char_index"]?.intValue
        )
    }
}

/// A model reply with everything the API said about it.
public struct AnthropicResponse: Sendable, Equatable {

    /// One block of the reply.
    public enum Block: Sendable, Equatable {
        case text(String, citations: [AnthropicCitation])
        case thinking(String)
        case redactedThinking
        case toolUse(ToolUse)
        case serverToolUse(id: String, name: String, input: JSONValue)
        /// Anything else (tool results from server tools, fallback markers…),
        /// kept raw so nothing is lost.
        case other(type: String, raw: JSONValue)
    }

    /// A client tool the model wants run.
    public struct ToolUse: Sendable, Equatable {
        public let id: String
        public let name: String
        public let input: JSONValue
    }

    /// Detail on a `refusal` stop.
    public struct StopDetails: Sendable, Equatable {
        public let category: String?
        public let explanation: String?
    }

    public let id: String
    public let model: String
    public let content: [Block]
    public let stopReason: AnthropicStopReason
    public let stopSequence: String?
    public let stopDetails: StopDetails?
    public let usage: AnthropicUsage
    public let serviceTier: String?

    /// The reply as the API sent it, for anything the typed view leaves out
    /// and for echoing an assistant turn back verbatim.
    public let raw: JSONValue

    /// The text blocks joined.
    public var text: String {
        content.compactMap { block -> String? in
            if case .text(let text, _) = block { return text }
            return nil
        }.joined()
    }

    /// Every client tool call in the reply, in order.
    public var toolUses: [ToolUse] {
        content.compactMap { block -> ToolUse? in
            if case .toolUse(let use) = block { return use }
            return nil
        }
    }

    /// The reply's citations, in order.
    public var citations: [AnthropicCitation] {
        content.flatMap { block -> [AnthropicCitation] in
            if case .text(_, let citations) = block { return citations }
            return []
        }
    }

    /// Parse a Messages API reply.
    static func parse(_ json: JSONValue) throws -> AnthropicResponse {
        guard let blocks = json["content"]?.arrayValue else {
            throw LLMError.decodingFailed("no `content` array in the reply")
        }
        let details = json["stop_details"].map {
            StopDetails(category: $0["category"]?.stringValue, explanation: $0["explanation"]?.stringValue)
        }
        return AnthropicResponse(
            id: json["id"]?.stringValue ?? "",
            model: json["model"]?.stringValue ?? "",
            content: blocks.map(Block.parse),
            stopReason: AnthropicStopReason(wire: json["stop_reason"]?.stringValue),
            stopSequence: json["stop_sequence"]?.stringValue,
            stopDetails: details,
            usage: AnthropicUsage.parse(json["usage"]),
            serviceTier: json["service_tier"]?.stringValue,
            raw: json
        )
    }

    /// The explanation of a refusal, for `LLMError.refused`.
    var refusalDetail: String {
        stopDetails?.explanation
            ?? stopDetails?.category.map { "declined for category \($0)" }
            ?? "the request was declined by the model's safety system"
    }
}

extension AnthropicResponse.Block {

    /// Parse one content block.
    static func parse(_ json: JSONValue) -> AnthropicResponse.Block {
        let type = json["type"]?.stringValue ?? ""
        switch type {
        case "text":
            let citations = (json["citations"]?.arrayValue ?? []).compactMap(AnthropicCitation.parse)
            return .text(json["text"]?.stringValue ?? "", citations: citations)
        case "thinking":
            return .thinking(json["thinking"]?.stringValue ?? "")
        case "redacted_thinking":
            return .redactedThinking
        case "tool_use":
            return .toolUse(AnthropicResponse.ToolUse(
                id: json["id"]?.stringValue ?? "",
                name: json["name"]?.stringValue ?? "",
                input: json["input"] ?? .object([:])
            ))
        case "server_tool_use":
            return .serverToolUse(
                id: json["id"]?.stringValue ?? "",
                name: json["name"]?.stringValue ?? "",
                input: json["input"] ?? .object([:])
            )
        default:
            return .other(type: type, raw: json)
        }
    }
}
