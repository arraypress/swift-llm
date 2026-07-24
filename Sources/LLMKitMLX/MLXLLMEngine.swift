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
import CoreImage
import LLMKit
import MLXLMCommon
import MLXLLM
import MLXVLM
import MLXHuggingFace
import HuggingFace
import Tokenizers
import Hub

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
    ///
    /// The weights are fetched with swift-transformers' `HubApi` rather than
    /// mlx-swift-lm's built-in downloader — the latter stalls on large single
    /// files (grabs the small config/tokenizer files, then never starts the
    /// multi-GB `model.safetensors`). `HubApi` streams large files reliably and
    /// reports real byte-level progress; we then load straight from the cached
    /// directory it returns.
    public func prepare(onProgress: (@Sendable (Double) -> Void)? = nil) async throws {
        do {
            let directory = try await HubApi.shared.snapshot(from: model.repoID) { progress in
                onProgress?(progress.fractionCompleted)
            }
            container = try await loadModelContainer(
                from: directory, using: #huggingFaceTokenizerLoader())
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
            let imgs = Self.images(in: messages)
            return imgs.isEmpty
                ? try await session.respond(to: prompt)
                : try await session.respond(to: prompt, images: imgs, videos: [], audios: [])
        } catch {
            throw LLMError.requestFailed(String(describing: error))
        }
    }

    /// Decode any attached image data (JPEG/PNG) into MLX image inputs for a VLM.
    private static func images(in messages: [LLMMessage]) -> [UserInput.Image] {
        messages.flatMap(\.images).compactMap { data in
            CIImage(data: data).map(UserInput.Image.ciImage)
        }
    }

    /// Real incremental streaming — yields token chunks as the model generates.
    public func streamResponse(to messages: [LLMMessage], options: GenerationOptions) -> AsyncThrowingStream<String, Error> {
        guard let container else {
            return AsyncThrowingStream { $0.finish(throwing: LLMError.notReady) }
        }
        let systemText = messages.filter { $0.role == .system }.map(\.text).joined(separator: "\n")
        let prompt = messages
            .filter { $0.role != .system }
            .map { $0.role == .assistant ? "Assistant: \($0.text)" : $0.text }
            .joined(separator: "\n")

        // The ChatSession retains itself in the stream's internal task, so it
        // stays alive for the duration even though this local goes out of scope.
        let session = ChatSession(
            container,
            instructions: systemText.isEmpty ? nil : systemText,
            generateParameters: GenerateParameters(temperature: Float(options.temperature))
        )
        return session.streamResponse(to: prompt, images: Self.images(in: messages))
    }
}
