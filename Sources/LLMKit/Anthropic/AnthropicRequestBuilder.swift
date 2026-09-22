//
//  AnthropicRequestBuilder.swift
//  LLMKit
//
//  Turns messages, attachments and configuration into a Messages API body —
//  pure functions over `JSONValue`, so every shape here is asserted in tests
//  without a network. The rules that matter: `.system` messages join into the
//  top-level `system`; turns must alternate user / assistant, so consecutive
//  same-role messages merge; attachments go before text, as the vision guide
//  recommends; sampling parameters are omitted on models that reject them.
//

import Foundation

/// Request assembly for `AnthropicEngine`.
enum AnthropicRequestBuilder {

    // MARK: - Turns

    /// The `messages[]` turns for a conversation, alternating by role.
    static func turns(from messages: [LLMMessage], citations: Bool) -> [JSONValue] {
        var entries: [(role: String, blocks: [JSONValue])] = []
        for message in messages where message.role != .system {
            let blocks = contentBlocks(for: message, citations: citations)
            if let last = entries.indices.last, entries[last].role == message.role.rawValue {
                entries[last].blocks += blocks
            } else {
                entries.append((message.role.rawValue, blocks))
            }
        }
        return entries.map { entry in
            // A lone text block may travel as a plain string, the common case.
            if entry.blocks.count == 1, entry.blocks[0]["type"] == "text",
               let text = entry.blocks[0]["text"] {
                return ["role": .string(entry.role), "content": text]
            }
            return ["role": .string(entry.role), "content": .array(entry.blocks)]
        }
    }

    /// One message as content blocks: its attachments, then its text (omitted
    /// when empty, which an attachment-only message is).
    static func contentBlocks(for message: LLMMessage, citations: Bool) -> [JSONValue] {
        var blocks = message.allAttachments.map { block(for: $0, citations: citations) }
        if !message.text.isEmpty {
            blocks.append(["type": "text", "text": .string(message.text)])
        }
        return blocks
    }

    /// The content block for one attachment.
    static func block(for attachment: LLMAttachment, citations: Bool) -> JSONValue {
        func document(_ source: JSONValue, title: String?) -> JSONValue {
            var object: [String: JSONValue] = ["type": "document", "source": source]
            if let title { object["title"] = .string(title) }
            if citations { object["citations"] = ["enabled": true] }
            return .object(object)
        }
        switch attachment {
        case .image(let data):
            return ["type": "image",
                    "source": ["type": "base64",
                               "media_type": .string(ImageData.mediaType(of: data)),
                               "data": .string(data.base64EncodedString())]]
        case .imageURL(let url):
            return ["type": "image", "source": ["type": "url", "url": .string(url.absoluteString)]]
        case .pdf(let data, let title):
            return document(["type": "base64", "media_type": "application/pdf",
                             "data": .string(data.base64EncodedString())], title: title)
        case .pdfURL(let url, let title):
            return document(["type": "url", "url": .string(url.absoluteString)], title: title)
        case .text(let text, let title):
            return document(["type": "text", "media_type": "text/plain", "data": .string(text)], title: title)
        case .file(let id, let kind):
            switch kind {
            case .image:
                return ["type": "image", "source": ["type": "file", "file_id": .string(id)]]
            case .document:
                return document(["type": "file", "file_id": .string(id)], title: nil)
            }
        }
    }

    /// An assistant turn that echoes a reply's content verbatim, thinking
    /// blocks and signatures included, which a tool loop must do.
    static func assistantTurn(echoing response: AnthropicResponse) -> JSONValue {
        ["role": "assistant", "content": response.raw["content"] ?? .array([])]
    }

    /// A user turn carrying tool results, in the order the calls were made.
    static func toolResultTurn(_ results: [(id: String, output: String, isError: Bool)]) -> JSONValue {
        let blocks: [JSONValue] = results.map { result in
            var object: [String: JSONValue] = [
                "type": "tool_result",
                "tool_use_id": .string(result.id),
                "content": .string(result.output),
            ]
            if result.isError { object["is_error"] = true }
            return .object(object)
        }
        return ["role": "user", "content": .array(blocks)]
    }

    // MARK: - System

    /// The top-level `system`: a string, or a text block with `cache_control`
    /// when caching is on. `nil` when there is no system text.
    static func system(from messages: [LLMMessage], cache: Bool, ttl: AnthropicCacheTTL) -> JSONValue? {
        let text = messages.filter { $0.role == .system }.map(\.text).joined(separator: "\n")
        guard !text.isEmpty else { return nil }
        guard cache else { return .string(text) }
        var control: [String: JSONValue] = ["type": "ephemeral"]
        if ttl != .fiveMinutes { control["ttl"] = .string(ttl.rawValue) }
        return .array([["type": "text", "text": .string(text), "cache_control": .object(control)]])
    }

    // MARK: - Body

    /// The full request body.
    static func body(
        model: String,
        turns: [JSONValue],
        system: JSONValue?,
        options: GenerationOptions,
        configuration: AnthropicConfiguration,
        sendsSamplingParameters: Bool,
        stream: Bool
    ) -> JSONValue {
        var body: [String: JSONValue] = [
            "model": .string(model),
            "max_tokens": .number(Double(options.maxTokens ?? AnthropicEngine.defaultMaxTokens)),
            "messages": .array(turns),
        ]
        if let system { body["system"] = system }
        // Thinking and sampling are mutually exclusive on the wire: with
        // thinking on, any temperature but 1 is a 400 (verified live).
        let thinkingOn = configuration.thinking?.isEnabled ?? false
        if sendsSamplingParameters && !thinkingOn {
            body["temperature"] = .number(options.temperature)
            if let topP = options.topP { body["top_p"] = .number(topP) }
            if let topK = configuration.topK { body["top_k"] = .number(Double(topK)) }
        }
        if stream { body["stream"] = true }
        if let thinking = configuration.thinking { body["thinking"] = thinking.json }

        var outputConfig: [String: JSONValue] = [:]
        if let effort = configuration.effort { outputConfig["effort"] = .string(effort.rawValue) }
        if let schema = configuration.outputSchema {
            outputConfig["format"] = ["type": "json_schema", "schema": schema]
        }
        if !outputConfig.isEmpty { body["output_config"] = .object(outputConfig) }

        let tools = configuration.tools.map(\.json) + configuration.serverTools.map(\.json)
        if !tools.isEmpty { body["tools"] = .array(tools) }
        if let choice = configuration.toolChoice { body["tool_choice"] = choice.json }

        if !configuration.stopSequences.isEmpty {
            body["stop_sequences"] = .array(configuration.stopSequences.map(JSONValue.string))
        }
        if let userID = configuration.userID { body["metadata"] = ["user_id": .string(userID)] }
        if let tier = configuration.serviceTier { body["service_tier"] = .string(tier.rawValue) }
        if let geo = configuration.inferenceGeo { body["inference_geo"] = .string(geo) }
        if let fallbacks = configuration.refusalFallbacks { body["fallbacks"] = fallbacks.json }
        return .object(body)
    }

    /// The `count_tokens` body: what shapes the prompt, without generation
    /// settings.
    static func countTokensBody(
        model: String,
        turns: [JSONValue],
        system: JSONValue?,
        configuration: AnthropicConfiguration
    ) -> JSONValue {
        var body: [String: JSONValue] = ["model": .string(model), "messages": .array(turns)]
        if let system { body["system"] = system }
        if let thinking = configuration.thinking { body["thinking"] = thinking.json }
        let tools = configuration.tools.map(\.json) + configuration.serverTools.map(\.json)
        if !tools.isEmpty { body["tools"] = .array(tools) }
        if let choice = configuration.toolChoice { body["tool_choice"] = choice.json }
        return .object(body)
    }

    // MARK: - Headers

    /// Authentication, version, and any beta ids the configuration implies.
    static func headers(apiKey: String, configuration: AnthropicConfiguration) -> [String: String] {
        var headers = ["x-api-key": apiKey, "anthropic-version": AnthropicEngine.apiVersion]
        var betas = configuration.betas
        if let fallbacks = configuration.refusalFallbacks, !betas.contains(fallbacks.beta) {
            betas.append(fallbacks.beta)
        }
        if !betas.isEmpty { headers["anthropic-beta"] = betas.joined(separator: ",") }
        return headers
    }
}
