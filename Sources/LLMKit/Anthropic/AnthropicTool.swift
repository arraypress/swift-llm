//
//  AnthropicTool.swift
//  LLMKit
//
//  Tools the model can call: client tools you define (and optionally let the
//  engine run for you), and Anthropic-hosted server tools that need nothing
//  from you but a declaration.
//

import Foundation

/// A client tool: a name, a description the model reads to decide when to
/// call it, a JSON Schema for its input, and optionally a handler.
///
/// ```swift
/// let weather = AnthropicTool(
///     name: "get_weather",
///     description: "Call this when the user asks about current weather.",
///     inputSchema: ["type": "object",
///                   "properties": ["city": ["type": "string"]],
///                   "required": ["city"],
///                   "additionalProperties": false],
///     strict: true
/// ) { input in
///     "18°C and clear in \(input["city"]?.stringValue ?? "?")"
/// }
/// ```
public struct AnthropicTool: Sendable {

    /// What the model runs, and what it returns to the model as the tool
    /// result. Throwing reports an error result the model can react to.
    public typealias Handler = @Sendable (JSONValue) async throws -> String

    /// The tool name (`get_weather`).
    public var name: String

    /// When and why to call it. Be prescriptive about *when*.
    public var description: String

    /// JSON Schema for the input object.
    public var inputSchema: JSONValue

    /// Guarantee the input validates against the schema exactly. Requires
    /// `additionalProperties: false` and `required` on every object.
    public var strict: Bool

    /// Stream large inputs as they are generated instead of after validation.
    /// Only meaningful on streamed requests; the input then needs validating
    /// client-side.
    public var eagerInputStreaming: Bool

    /// Runs the tool when the engine drives the loop. Without one, `run`
    /// returns the `tool_use` blocks for the caller to handle.
    public var handler: Handler?

    /// Create a tool.
    public init(
        name: String,
        description: String,
        inputSchema: JSONValue,
        strict: Bool = false,
        eagerInputStreaming: Bool = false,
        handler: Handler? = nil
    ) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.strict = strict
        self.eagerInputStreaming = eagerInputStreaming
        self.handler = handler
    }

    /// The `tools[]` entry.
    var json: JSONValue {
        var object: [String: JSONValue] = [
            "name": .string(name),
            "description": .string(description),
            "input_schema": inputSchema,
        ]
        if strict { object["strict"] = true }
        if eagerInputStreaming { object["eager_input_streaming"] = true }
        return .object(object)
    }
}

/// An Anthropic-hosted tool. Results arrive inside the reply; nothing runs
/// on the client.
public enum AnthropicServerTool: Sendable, Equatable {

    /// Web search with built-in result filtering (`web_search_20260209`).
    /// Use `allowedDomains` or `blockedDomains`, never both.
    case webSearch(maxUses: Int? = nil, allowedDomains: [String] = [], blockedDomains: [String] = [])

    /// Fetch a page already mentioned in the conversation (`web_fetch_20260209`).
    case webFetch(maxUses: Int? = nil, allowedDomains: [String] = [], blockedDomains: [String] = [], maxContentTokens: Int? = nil)

    /// Run code in Anthropic's sandbox (`code_execution_20260521`). Leave it
    /// out when web tools are declared; they bring their own sandbox.
    case codeExecution

    /// The `tools[]` entry.
    var json: JSONValue {
        switch self {
        case .webSearch(let maxUses, let allowed, let blocked):
            var object: [String: JSONValue] = ["type": "web_search_20260209", "name": "web_search"]
            if let maxUses { object["max_uses"] = .number(Double(maxUses)) }
            if !allowed.isEmpty { object["allowed_domains"] = .array(allowed.map(JSONValue.string)) }
            if !blocked.isEmpty { object["blocked_domains"] = .array(blocked.map(JSONValue.string)) }
            return .object(object)
        case .webFetch(let maxUses, let allowed, let blocked, let maxContentTokens):
            var object: [String: JSONValue] = ["type": "web_fetch_20260209", "name": "web_fetch"]
            if let maxUses { object["max_uses"] = .number(Double(maxUses)) }
            if !allowed.isEmpty { object["allowed_domains"] = .array(allowed.map(JSONValue.string)) }
            if !blocked.isEmpty { object["blocked_domains"] = .array(blocked.map(JSONValue.string)) }
            if let maxContentTokens { object["max_content_tokens"] = .number(Double(maxContentTokens)) }
            return .object(object)
        case .codeExecution:
            return ["type": "code_execution_20260521", "name": "code_execution"]
        }
    }
}
