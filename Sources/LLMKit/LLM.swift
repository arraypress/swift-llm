//
//  LLM.swift
//  LLMKit
//
//  A thin facade over any engine — the everyday entry point.
//

import Foundation

/// The everyday entry point: wrap an engine and query it.
///
/// ```swift
/// let llm = LLM(RemoteEngine(model: "gpt-4o", endpoint: .openAI, apiKey: key))
/// let reply = try await llm.generate("Summarize this…")
/// ```
public struct LLM: Sendable {

    /// The backing engine.
    public let engine: any LLMEngine

    /// Wrap an engine.
    public init(_ engine: any LLMEngine) {
        self.engine = engine
    }

    /// Whether the engine is ready.
    public var isReady: Bool {
        get async { await engine.isReady }
    }

    /// Prepare the engine (download/load a model, or validate a key).
    public func prepare() async throws {
        try await engine.prepare()
    }

    /// Generate a reply to a single prompt.
    public func generate(_ prompt: String, options: GenerationOptions = GenerationOptions()) async throws -> String {
        try await engine.respond(to: prompt, options: options)
    }

    /// Respond to a full conversation.
    public func respond(to messages: [LLMMessage], options: GenerationOptions = GenerationOptions()) async throws -> String {
        try await engine.respond(to: messages, options: options)
    }

    /// Structured output: parse the reply into a `Decodable`.
    public func respond<T: Decodable>(
        to messages: [LLMMessage],
        generating type: T.Type,
        options: GenerationOptions = .deterministic
    ) async throws -> T {
        try await engine.respond(to: messages, generating: type, options: options)
    }
}
