//
//  Reasoning.swift
//  LLMKit
//
//  Reasoning models (Qwen3, DeepSeek-R1, …) emit a `<think>…</think>` block
//  before their answer. These helpers separate that trace from the final answer
//  so a UI can hide, collapse, or drop it — independent of which engine ran.
//

import Foundation

/// A model reply split into its optional reasoning trace and the final answer.
public struct ReasoningSplit: Sendable, Equatable {
    /// The chain-of-thought between the reasoning tags, if the reply had one.
    public let reasoning: String?
    /// The final answer, with any reasoning block removed.
    public let answer: String
}

public extension String {

    /// Split a model reply into its `<think>` reasoning trace and the answer.
    ///
    /// Recognises `<think>`, `<thinking>`, and `<reasoning>` tags
    /// (case-insensitive). If a block is opened but never closed (e.g. output
    /// was cut off mid-thought), everything after the open tag is treated as
    /// reasoning and the answer is whatever preceded it.
    func splitReasoning() -> ReasoningSplit {
        for tag in ["think", "thinking", "reasoning"] {
            guard let open = range(of: "<\(tag)>", options: .caseInsensitive) else { continue }
            let afterOpen = self[open.upperBound...]
            let before = self[..<open.lowerBound]

            if let close = afterOpen.range(of: "</\(tag)>", options: .caseInsensitive) {
                let reasoning = afterOpen[..<close.lowerBound].trimmed
                // Join the text before and after the block; a single space only
                // when both sides are non-empty (a block in the middle of text).
                let parts = [before.trimmed, afterOpen[close.upperBound...].trimmed].filter { !$0.isEmpty }
                return ReasoningSplit(reasoning: reasoning.isEmpty ? nil : reasoning,
                                      answer: parts.joined(separator: " "))
            } else {
                let reasoning = afterOpen.trimmed
                return ReasoningSplit(reasoning: reasoning.isEmpty ? nil : reasoning,
                                      answer: String(before).trimmed)
            }
        }
        return ReasoningSplit(reasoning: nil, answer: trimmed)
    }

    /// The final answer with any `<think>`-style reasoning block removed.
    func strippingReasoning() -> String { splitReasoning().answer }

    private var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}

private extension Substring {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
