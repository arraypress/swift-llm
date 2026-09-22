//
//  AnthropicStream.swift
//  LLMKit
//
//  The Messages API streams a reply as events that build content blocks one
//  index at a time: `content_block_start`, a run of `content_block_delta`,
//  `content_block_stop`, then `message_delta` with the stop reason and
//  `message_stop`. The accumulator replays those into the same
//  `AnthropicResponse` a non-streamed call returns, while surfacing the deltas
//  live. Pure over `JSONValue`, so it is tested with recorded event lines.
//

import Foundation

/// One thing that happened on a streamed reply.
public enum AnthropicStreamEvent: Sendable, Equatable {
    /// The reply has begun; `id` and `model` are known.
    case messageStart(id: String, model: String)
    /// More reply text.
    case textDelta(String)
    /// More (summarised) thinking.
    case thinkingDelta(String)
    /// The model has started a tool call.
    case toolUseStart(index: Int, id: String, name: String)
    /// More of a tool call's input, as a JSON fragment.
    case toolInputDelta(index: Int, partialJSON: String)
    /// A content block is complete.
    case blockStop(index: Int)
    /// The reply is complete, assembled into the same shape a non-streamed
    /// call returns.
    case completed(AnthropicResponse)
}

/// Rebuilds a reply from its stream events.
struct AnthropicStreamAccumulator {

    /// A content block under construction.
    private struct PartialBlock {
        var type: String
        var text = ""
        var thinking = ""
        var signature = ""
        var toolID = ""
        var toolName = ""
        var partialJSON = ""
        var citations: [JSONValue] = []
        var raw: JSONValue
    }

    private var id = ""
    private var model = ""
    private var usage: JSONValue = .object([:])
    private var stopReason: String?
    private var stopSequence: String?
    private var stopDetails: JSONValue?
    private var blocks: [Int: PartialBlock] = [:]

    init() {}

    /// Feed one SSE line. Framing lines yield nothing; an `error` event
    /// throws `requestFailed`.
    mutating func consume(line: String) throws -> [AnthropicStreamEvent] {
        guard let object = ServerSentEvents.jsonObject(of: line) else { return [] }
        return try consume(event: JSONValue(any: object) ?? .null)
    }

    /// Feed one decoded event.
    mutating func consume(event: JSONValue) throws -> [AnthropicStreamEvent] {
        switch event["type"]?.stringValue {
        case "message_start":
            let message = event["message"]
            id = message?["id"]?.stringValue ?? ""
            model = message?["model"]?.stringValue ?? ""
            usage = message?["usage"] ?? .object([:])
            return [.messageStart(id: id, model: model)]

        case "content_block_start":
            guard let index = event["index"]?.intValue, let block = event["content_block"] else { return [] }
            let type = block["type"]?.stringValue ?? ""
            var partial = PartialBlock(type: type, raw: block)
            partial.text = block["text"]?.stringValue ?? ""
            partial.thinking = block["thinking"]?.stringValue ?? ""
            partial.toolID = block["id"]?.stringValue ?? ""
            partial.toolName = block["name"]?.stringValue ?? ""
            blocks[index] = partial
            if type == "tool_use" || type == "server_tool_use" {
                return [.toolUseStart(index: index, id: partial.toolID, name: partial.toolName)]
            }
            return []

        case "content_block_delta":
            guard let index = event["index"]?.intValue, let delta = event["delta"] else { return [] }
            var partial = blocks[index] ?? PartialBlock(type: "text", raw: .object(["type": "text"]))
            defer { blocks[index] = partial }
            switch delta["type"]?.stringValue {
            case "text_delta":
                let text = delta["text"]?.stringValue ?? ""
                partial.text += text
                return text.isEmpty ? [] : [.textDelta(text)]
            case "thinking_delta":
                let thinking = delta["thinking"]?.stringValue ?? ""
                partial.thinking += thinking
                return thinking.isEmpty ? [] : [.thinkingDelta(thinking)]
            case "input_json_delta":
                let fragment = delta["partial_json"]?.stringValue ?? ""
                partial.partialJSON += fragment
                return fragment.isEmpty ? [] : [.toolInputDelta(index: index, partialJSON: fragment)]
            case "signature_delta":
                partial.signature += delta["signature"]?.stringValue ?? ""
                return []
            case "citations_delta":
                if let citation = delta["citation"] { partial.citations.append(citation) }
                return []
            default:
                return []
            }

        case "content_block_stop":
            guard let index = event["index"]?.intValue else { return [] }
            return [.blockStop(index: index)]

        case "message_delta":
            let delta = event["delta"]
            stopReason = delta?["stop_reason"]?.stringValue ?? stopReason
            stopSequence = delta?["stop_sequence"]?.stringValue ?? stopSequence
            stopDetails = delta?["stop_details"] ?? stopDetails
            if let outputTokens = event["usage"]?["output_tokens"], var object = usage.objectValue {
                object["output_tokens"] = outputTokens
                usage = .object(object)
            }
            return []

        case "message_stop":
            return [.completed(try response())]

        case "error":
            let error = event["error"]
            throw LLMError.requestFailed(error?["message"]?.stringValue ?? "stream error")

        default:
            return []
        }
    }

    /// The assembled reply.
    private func response() throws -> AnthropicResponse {
        let content: [JSONValue] = blocks.keys.sorted().map { index in
            let block = blocks[index]!
            var object = block.raw.objectValue ?? ["type": .string(block.type)]
            switch block.type {
            case "text":
                object["text"] = .string(block.text)
                if !block.citations.isEmpty { object["citations"] = .array(block.citations) }
            case "thinking":
                object["thinking"] = .string(block.thinking)
                if !block.signature.isEmpty { object["signature"] = .string(block.signature) }
            case "tool_use", "server_tool_use":
                object["id"] = .string(block.toolID)
                object["name"] = .string(block.toolName)
                let data = Data(block.partialJSON.utf8)
                object["input"] = block.partialJSON.isEmpty ? .object([:]) : ((try? JSONValue.parse(data)) ?? .object([:]))
            default:
                break
            }
            return .object(object)
        }
        var message: [String: JSONValue] = [
            "id": .string(id),
            "type": "message",
            "role": "assistant",
            "model": .string(model),
            "content": .array(content),
            "usage": usage,
        ]
        if let stopReason { message["stop_reason"] = .string(stopReason) }
        if let stopSequence { message["stop_sequence"] = .string(stopSequence) }
        if let stopDetails { message["stop_details"] = stopDetails }
        return try AnthropicResponse.parse(.object(message))
    }
}
