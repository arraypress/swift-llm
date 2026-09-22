//
//  ServerSentEvents.swift
//  LLMKit
//
//  The few lines of the server-sent-events wire format that streaming chat APIs
//  actually use: `data: {json}` records separated by blank lines, `event:` names
//  that can be ignored because every payload carries its own `type`, and the
//  `data: [DONE]` sentinel OpenAI-style servers send last. Pure functions, so
//  each engine's stream parser is testable without a socket.
//

import Foundation

/// Pure helpers for the SSE framing used by streaming chat APIs.
enum ServerSentEvents {

    /// The sentinel OpenAI-compatible servers send as their final record.
    static let done = "[DONE]"

    /// The payload of a `data:` line, or `nil` for anything else (`event:`,
    /// `id:`, `:` comments, blank separators). One leading space after the
    /// colon is framing and is dropped; nothing else is trimmed, because a
    /// token that starts with a space is a real token.
    static func dataPayload(of line: String) -> String? {
        guard line.hasPrefix("data:") else { return nil }
        var payload = line.dropFirst("data:".count)
        if payload.first == " " { payload = payload.dropFirst() }
        return String(payload)
    }

    /// The `data:` payload decoded as a JSON object, or `nil` when the line is
    /// not a data record, is the `[DONE]` sentinel, or isn't an object.
    static func jsonObject(of line: String) -> [String: Any]? {
        guard let payload = dataPayload(of: line), payload != done,
              let data = payload.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return object
    }
}
