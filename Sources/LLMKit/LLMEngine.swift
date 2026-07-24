//
//  LLMEngine.swift
//  LLMKit
//
//  The protocol every backend implements — Apple FoundationModels, a local MLX
//  model, or any OpenAI-compatible cloud endpoint. Callers program against this,
//  so the tier is a swappable choice, not a rewrite.
//

import Foundation

/// A language-model backend.
public protocol LLMEngine: Sendable {

    /// Whether the engine is ready to respond (model loaded / key present).
    var isReady: Bool { get async }

    /// Prepare the engine — download/load a local model, or validate a key.
    func prepare() async throws

    /// Respond to a conversation and return the assistant's text.
    func respond(to messages: [LLMMessage], options: GenerationOptions) async throws -> String

    /// Stream the assistant's reply incrementally as text chunks. Engines that
    /// can't stream yield the whole reply as a single chunk (see the default).
    func streamResponse(to messages: [LLMMessage], options: GenerationOptions) -> AsyncThrowingStream<String, Error>
}

public extension LLMEngine {

    /// Respond to a single user prompt.
    func respond(to prompt: String, options: GenerationOptions = GenerationOptions()) async throws -> String {
        try await respond(to: [.user(prompt)], options: options)
    }

    /// Respond with **structured output**: the model's text is parsed into a
    /// `Decodable`. Provide a prompt that asks for JSON matching your type.
    ///
    /// ```swift
    /// struct Expense: Decodable { let title: String; let amount: String }
    /// let e: Expense = try await engine.respond(
    ///     to: [.user("Return JSON {title, amount} for: coffee £3.50")],
    ///     generating: Expense.self)
    /// ```
    func respond<T: Decodable>(
        to messages: [LLMMessage],
        generating type: T.Type,
        options: GenerationOptions = .deterministic
    ) async throws -> T {
        let text = try await respond(to: messages, options: options)
        return try JSONExtractor.decode(type, from: text)
    }

    /// Default streaming: run `respond` and deliver the whole reply as one
    /// chunk. Engines with incremental generation (e.g. `MLXLLMEngine`) override.
    func streamResponse(to messages: [LLMMessage], options: GenerationOptions) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    continuation.yield(try await respond(to: messages, options: options))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
