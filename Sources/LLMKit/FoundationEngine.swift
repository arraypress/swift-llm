//
//  FoundationEngine.swift
//  LLMKit
//
//  Apple's on-device system model via the FoundationModels framework — free, no
//  download, private. `.system` messages (plus any engine instructions) become
//  the session instructions; the rest become the prompt.
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
        guard case .available = SystemLanguageModel.default.availability else {
            throw LLMError.unavailable("Apple Intelligence is unavailable on this device")
        }
        guard !messages.contains(where: { !$0.images.isEmpty }) else {
            throw LLMError.unavailable("FoundationModels here is text-only; route image requests to a RemoteEngine or an MLX vision model.")
        }

        let systemText = ([instructions].compactMap { $0 } + messages.filter { $0.role == .system }.map(\.text))
            .joined(separator: "\n")
        let prompt = messages
            .filter { $0.role != .system }
            .map { $0.role == .assistant ? "Assistant: \($0.text)" : $0.text }
            .joined(separator: "\n")

        let session = LanguageModelSession(instructions: systemText)
        do {
            let response = try await session.respond(
                to: prompt,
                options: FoundationModels.GenerationOptions(temperature: options.temperature)
            )
            return response.content
        } catch {
            throw LLMError.requestFailed(String(describing: error))
        }
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
