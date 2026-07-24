//
//  MLXModel.swift
//  LLMKitMLX
//
//  The curated set of local MLX models, grouped by purpose so an app can show
//  them in sections (General · Uncensored · Creative & NSFW). Every entry's
//  architecture is one mlx-swift-lm can load (qwen3 / qwen3_5(_moe) / llama /
//  mistral / phi).
//

import Foundation

/// A downloadable local MLX model.
public struct MLXModel: Sendable, Equatable {

    /// A section for grouping models in a picker.
    public enum Group: String, Sendable, CaseIterable {
        case general = "General"
        case uncensored = "Uncensored"
        case creative = "Creative & NSFW"
    }

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

    /// Which section this model belongs to.
    public let group: Group

    public init(
        id: String,
        repoID: String,
        displayName: String,
        approximateSizeMB: Int,
        license: String,
        runsOnMobile: Bool,
        group: Group = .general
    ) {
        self.id = id
        self.repoID = repoID
        self.displayName = displayName
        self.approximateSizeMB = approximateSizeMB
        self.license = license
        self.runsOnMobile = runsOnMobile
        self.group = group
    }

    // MARK: - General (commercial-safe: Apache / MIT weights)

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

    // MARK: - Uncensored (abliterated general-purpose models)
    // Refusal behaviour removed (and often re-tuned). Base licenses apply. Most
    // are Qwen3 reasoning models → emit a <think> block (use splitReasoning()).

    /// Josiefied Qwen3 4B (abliterated) — uncensored, same footprint as `qwen3_4B`.
    public static let josiefiedQwen3_4B = MLXModel(
        id: "josiefied-qwen3-4b", repoID: "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        displayName: "Qwen3 4B (uncensored)", approximateSizeMB: 2_400, license: "Apache-2.0",
        runsOnMobile: true, group: .uncensored)

    /// Josiefied Qwen3 8B (abliterated) — uncensored, richer prose; Mac-class.
    public static let josiefiedQwen3_8B = MLXModel(
        id: "josiefied-qwen3-8b", repoID: "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit",
        displayName: "Qwen3 8B (uncensored)", approximateSizeMB: 4_500, license: "Apache-2.0",
        runsOnMobile: false, group: .uncensored)

    /// Huihui Qwen3.5 9B (abliterated) — newer Qwen generation, uncensored.
    public static let huihuiQwen3_5_9B = MLXModel(
        id: "qwen3.5-9b-uncensored", repoID: "huihui-ai/Huihui-Qwen3.5-9B-abliterated-mlx-4bit",
        displayName: "Qwen3.5 9B (uncensored)", approximateSizeMB: 5_100, license: "Apache-2.0",
        runsOnMobile: false, group: .uncensored)

    /// Huihui Qwen3.5 27B (abliterated) — big dense model; needs ~24 GB memory.
    public static let huihuiQwen3_5_27B = MLXModel(
        id: "qwen3.5-27b-uncensored", repoID: "mlx-community/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit",
        displayName: "Qwen3.5 27B (uncensored)", approximateSizeMB: 16_100, license: "Apache-2.0",
        runsOnMobile: false, group: .uncensored)

    /// Qwen3.6 35B-A3B (uncensored MoE) — 35B total / 3B active, fast on Apple
    /// Silicon; strong quality. ~20 GB → needs ~32 GB memory.
    public static let qwen3_6_35B_moe = MLXModel(
        id: "qwen3.6-35b-uncensored", repoID: "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
        displayName: "Qwen3.6 35B MoE (uncensored)", approximateSizeMB: 20_400, license: "Apache-2.0",
        runsOnMobile: false, group: .uncensored)

    /// Llama 3.1 8B (abliterated) — uncensored, a different (Llama) voice to Qwen.
    public static let llama3_1_8B_uncensored = MLXModel(
        id: "llama-3.1-8b-uncensored", repoID: "mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-4bit",
        displayName: "Llama 3.1 8B (uncensored)", approximateSizeMB: 4_500, license: "Llama 3.1 Community",
        runsOnMobile: false, group: .uncensored)

    // MARK: - Creative & NSFW (fiction / roleplay finetunes)
    // Tuned on stories & roleplay rather than just abliterated — best for prose.
    // Non-reasoning (no <think>).

    /// Cydonia 24B — TheDrummer's creative/roleplay finetune (Mistral-Small base).
    /// The community's go-to for rich prose. ~13 GB → needs ~24 GB memory.
    public static let cydonia24B = MLXModel(
        id: "cydonia-24b", repoID: "mlx-community/Cydonia-24B-v3-4bit",
        displayName: "Cydonia 24B (writer)", approximateSizeMB: 13_300, license: "Apache-2.0 (Mistral base)",
        runsOnMobile: false, group: .creative)

    /// Rocinante 12B — TheDrummer's Nemo-based creative/roleplay finetune; lighter.
    public static let rocinante12B = MLXModel(
        id: "rocinante-12b", repoID: "McG-221/Rocinante-X-12B-v1-absolute-heresy-mlx-8Bit",
        displayName: "Rocinante 12B (writer)", approximateSizeMB: 13_000, license: "Apache-2.0 (Nemo base)",
        runsOnMobile: false, group: .creative)

    /// Dirty Muse Writer — a small model fine-tuned specifically for adult erotica.
    /// On-topic out of the box but rough (2-bit); prefer Cydonia for quality.
    public static let museWriter = MLXModel(
        id: "muse-writer", repoID: "Jurisprudence/Dirty-Muse-Writer-v01-Uncensored-Erotica-NSFW-mlx-2Bit",
        displayName: "Muse Writer 3B (erotica, 2-bit)", approximateSizeMB: 2_900, license: "check repo",
        runsOnMobile: true, group: .creative)

    /// Every curated model, in section order.
    public static let all: [MLXModel] = [
        // General
        .qwen3_0_6B, .qwen3_1_7B, .qwen3_4B, .phi4Mini, .smolLM3_3B, .mistralSmall3,
        .llama3_2_1B, .llama3_2_3B, .llama3_1_8B,
        // Uncensored
        .josiefiedQwen3_4B, .josiefiedQwen3_8B, .huihuiQwen3_5_9B, .huihuiQwen3_5_27B,
        .qwen3_6_35B_moe, .llama3_1_8B_uncensored,
        // Creative & NSFW
        .cydonia24B, .rocinante12B, .museWriter,
    ]

    /// Models in a given section.
    public static func inGroup(_ group: Group) -> [MLXModel] {
        all.filter { $0.group == group }
    }
}
