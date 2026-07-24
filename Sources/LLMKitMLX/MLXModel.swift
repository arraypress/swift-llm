//
//  MLXModel.swift
//  LLMKitMLX
//
//  The curated set of local MLX models, with the metadata the app (or the
//  library) needs to pick one per device — commercial-safe first.
//

import Foundation

/// A downloadable local MLX model.
public struct MLXModel: Sendable, Equatable {

    /// Short identifier.
    public let id: String

    /// The Hugging Face repo id for the MLX-converted weights.
    public let repoID: String

    /// Human-readable name.
    public let displayName: String

    /// Rough on-disk size once downloaded (MB, order-of-magnitude).
    public let approximateSizeMB: Int

    /// The model-weights license.
    public let license: String

    /// Whether it's light enough to run on an iPhone.
    public let runsOnMobile: Bool

    public init(
        id: String,
        repoID: String,
        displayName: String,
        approximateSizeMB: Int,
        license: String,
        runsOnMobile: Bool
    ) {
        self.id = id
        self.repoID = repoID
        self.displayName = displayName
        self.approximateSizeMB = approximateSizeMB
        self.license = license
        self.runsOnMobile = runsOnMobile
    }

    // MARK: - Curated presets (commercial-safe: Apache / MIT weights)

    /// Qwen3 0.6B (4-bit) — tiny; fast phones + a quick smoke test. Apache-2.0.
    public static let qwen3_0_6B = MLXModel(
        id: "qwen3-0.6b", repoID: "mlx-community/Qwen3-0.6B-4bit",
        displayName: "Qwen3 0.6B", approximateSizeMB: 450, license: "Apache-2.0", runsOnMobile: true)

    /// Qwen3 1.7B (4-bit) — 6 GB phones, multilingual. Apache-2.0.
    public static let qwen3_1_7B = MLXModel(
        id: "qwen3-1.7b", repoID: "mlx-community/Qwen3-1.7B-4bit",
        displayName: "Qwen3 1.7B", approximateSizeMB: 1_100, license: "Apache-2.0", runsOnMobile: true)

    /// Qwen3 4B (4-bit) — 8 GB phones, strong multilingual. Apache-2.0.
    public static let qwen3_4B = MLXModel(
        id: "qwen3-4b", repoID: "mlx-community/Qwen3-4B-4bit",
        displayName: "Qwen3 4B", approximateSizeMB: 2_400, license: "Apache-2.0", runsOnMobile: true)

    /// Phi-4-mini (4-bit) — the smart sub-4B reasoner. MIT.
    public static let phi4Mini = MLXModel(
        id: "phi4-mini", repoID: "mlx-community/Phi-4-mini-instruct-4bit",
        displayName: "Phi-4-mini", approximateSizeMB: 2_400, license: "MIT", runsOnMobile: true)

    /// SmolLM3-3B (4-bit) — long-context standout. Apache-2.0.
    public static let smolLM3_3B = MLXModel(
        id: "smollm3-3b", repoID: "mlx-community/SmolLM3-3B-4bit",
        displayName: "SmolLM3-3B", approximateSizeMB: 1_800, license: "Apache-2.0", runsOnMobile: true)

    /// Mistral Small 3 (4-bit) — throughput all-rounder; Mac-class. Apache-2.0.
    public static let mistralSmall3 = MLXModel(
        id: "mistral-small-3", repoID: "mlx-community/Mistral-Small-3-Instruct-4bit",
        displayName: "Mistral Small 3", approximateSizeMB: 4_500, license: "Apache-2.0", runsOnMobile: false)

    // MARK: - Llama family
    // Meta's Llama Community License (NOT Apache/MIT) — free for most use, but
    // it carries the >700M-MAU clause; check the terms for large-scale apps.

    /// Llama 3.2 1B (4-bit) — tiny; runs on essentially any modern iPhone.
    public static let llama3_2_1B = MLXModel(
        id: "llama-3.2-1b", repoID: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        displayName: "Llama 3.2 1B", approximateSizeMB: 700, license: "Llama 3.2 Community", runsOnMobile: true)

    /// Llama 3.2 3B (4-bit) — the sweet spot for on-device chat; 6 GB+ phones.
    public static let llama3_2_3B = MLXModel(
        id: "llama-3.2-3b", repoID: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        displayName: "Llama 3.2 3B", approximateSizeMB: 1_800, license: "Llama 3.2 Community", runsOnMobile: true)

    /// Llama 3.1 8B (4-bit) — strongest Llama that *just* fits 8 GB iPhones; Mac-easy.
    public static let llama3_1_8B = MLXModel(
        id: "llama-3.1-8b", repoID: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        displayName: "Llama 3.1 8B", approximateSizeMB: 4_500, license: "Llama 3.1 Community", runsOnMobile: false)

    /// Every curated model.
    public static let all: [MLXModel] = [
        .qwen3_0_6B, .qwen3_1_7B, .qwen3_4B, .phi4Mini, .smolLM3_3B, .mistralSmall3,
        .llama3_2_1B, .llama3_2_3B, .llama3_1_8B,
    ]
}
