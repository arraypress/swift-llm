//
//  FoundationEngine.swift
//  LLMKit
//
//  Apple's on-device system model via the FoundationModels framework — free, no
//  download, private. `.system` messages (plus any engine instructions) become
//  the session instructions; the rest become the prompt. Streaming uses the
//  session's snapshot stream, which reports the whole reply so far on every
//  step; the engine turns those snapshots into the deltas every other engine
//  yields, so callers see one shape.
//
//  Note: image input landed in a later OS (WWDC26). On this SDK the engine is
//  text-only and throws for image messages — route those to a RemoteEngine or an
//  MLX vision model. When building on the newer SDK, map images to `.image()`.
//

import Foundation
import FoundationModels

/// Text generation via Apple's built-in system language model.
public struct FoundationEngine: LLMEngine {

    /// Persistent instructions applied to every request (merged with `.system`
    /// messages).
    public let instructions: String?

    /// Create the engine, optionally with standing instructions.
    public init(instructions: String? = nil) {
        self.instructions = instructions
    }

    public var isReady: Bool {
        get async { SystemLanguageModel.default.isAvailable }
    }

    public func prepare() async throws {
        switch SystemLanguageModel.default.availability {
        case .available:
            return
        case .unavailable(let reason):
            throw LLMError.unavailable(Self.describe(reason))
        }
    }

    public func respond(to messages: [LLMMessage], options: GenerationOptions) async throws -> String {
        let (session, prompt) = try makeSession(for: messages)
        do {
            let response = try await session.respond(to: prompt, options: Self.generationOptions(from: options))
            return response.content
        } catch {
            throw LLMError.requestFailed(String(describing: error))
        }
    }

    /// Live text: each snapshot from the session carries the reply so far, so
    /// only the part not yet delivered is yielded.
    public func streamResponse(to messages: [LLMMessage], options: GenerationOptions) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (session, prompt) = try makeSession(for: messages)
                    var delivered = ""
                    for try await snapshot in session.streamResponse(to: prompt, options: Self.generationOptions(from: options)) {
                        let delta = Self.delta(from: delivered, to: snapshot.content)
                        delivered = snapshot.content
                        if !delta.isEmpty { continuation.yield(delta) }
                    }
                    continuation.finish()
                } catch let error as LLMError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: LLMError.requestFailed(String(describing: error)))
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// A session with the merged instructions, and the prompt text, after the
    /// availability and text-only checks.
    private func makeSession(for messages: [LLMMessage]) throws -> (LanguageModelSession, String) {
        guard case .available = SystemLanguageModel.default.availability else {
            throw LLMError.unavailable("Apple Intelligence is unavailable on this device")
        }
        guard !messages.contains(where: { $0.allAttachments.contains { !$0.isText } }) else {
            throw LLMError.unavailable("FoundationModels here is text-only; route image and document requests to AnthropicEngine, a RemoteEngine or an MLX vision model.")
        }
        let systemText = ([instructions].compactMap { $0 } + messages.filter { $0.role == .system }.map(\.text))
            .joined(separator: "\n")
        let prompt = messages
            .filter { $0.role != .system }
            .map { message in
                let documents = message.allAttachments.compactMap { attachment -> String? in
                    if case .text(let text, let title) = attachment { return RemoteEngine.inlineDocument(text, title: title) }
                    return nil
                }
                let body = (documents + [message.text]).filter { !$0.isEmpty }.joined(separator: "\n")
                return message.role == .assistant ? "Assistant: \(body)" : body
            }
            .joined(separator: "\n")
        return (LanguageModelSession(instructions: systemText), prompt)
    }

    // MARK: - Pure helpers (testable without the model)

    /// The framework's options for ours: temperature always, the token cap
    /// only when the caller set one.
    static func generationOptions(from options: GenerationOptions) -> FoundationModels.GenerationOptions {
        FoundationModels.GenerationOptions(temperature: options.temperature, maximumResponseTokens: options.maxTokens)
    }

    /// The text in `current` beyond what `previous` already delivered. A
    /// snapshot that doesn't extend the previous one (the model revised
    /// earlier text) is delivered whole, so nothing is lost.
    static func delta(from previous: String, to current: String) -> String {
        current.hasPrefix(previous) ? String(current.dropFirst(previous.count)) : current
    }

    static func describe(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
        switch reason {
        case .deviceNotEligible: return "device not eligible for Apple Intelligence"
        case .appleIntelligenceNotEnabled: return "Apple Intelligence isn't enabled in Settings"
        case .modelNotReady: return "the on-device model is still preparing"
        @unknown default: return "unavailable"
        }
    }
}
