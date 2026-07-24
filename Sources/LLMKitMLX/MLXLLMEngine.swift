//
//  MLXLLMEngine.swift
//  LLMKitMLX
//
//  A local, downloaded LLM engine backed by mlx-swift-lm (MLX/GPU). Downloads
//  the model on first `prepare()`, caches it, then runs fully on-device.
//
//  A `final class` (not an actor) because mlx-swift-lm's `ChatSession` is a
//  non-Sendable class; `@unchecked Sendable` documents prepare-then-respond use.
//

import Foundation
import LLMKit
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Local text generation via an MLX model (Qwen3, Phi-4, SmolLM3, …).
public final class MLXLLMEngine: LLMEngine, @unchecked Sendable {

    /// The model this engine runs.
    public let model: MLXModel

    private var container: ModelContainer?

    /// Create an engine for one of the curated ``MLXModel`` presets.
    public init(_ model: MLXModel) {
        self.model = model
    }

    public var isReady: Bool { container != nil }

    /// Download (if needed) and load the model, reporting `0...1` progress.
    public func prepare(onProgress: (@Sendable (Double) -> Void)? = nil) async throws {
        do {
            container = try await loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: ModelConfiguration(id: model.repoID)
            ) { progress in
                onProgress?(progress.fractionCompleted)
            }
        } catch {
            throw LLMError.unavailable(String(describing: error))
        }
        onProgress?(1.0)
    }

    public func prepare() async throws {
        try await prepare(onProgress: nil)
    }

    public func respond(to messages: [LLMMessage], options: GenerationOptions) async throws -> String {
        guard let container else { throw LLMError.notReady }

        let systemText = messages.filter { $0.role == .system }.map(\.text).joined(separator: "\n")
        let prompt = messages
            .filter { $0.role != .system }
            .map { $0.role == .assistant ? "Assistant: \($0.text)" : $0.text }
            .joined(separator: "\n")

        let session = ChatSession(
            container,
            instructions: systemText.isEmpty ? nil : systemText,
            generateParameters: GenerateParameters(temperature: Float(options.temperature))
        )
        do {
            return try await session.respond(to: prompt)
        } catch {
            throw LLMError.requestFailed(String(describing: error))
        }
    }
}
