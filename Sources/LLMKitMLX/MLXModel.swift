//
//  MLXModel.swift
//  LLMKitMLX
//
//  The curated set of local MLX models. Each carries a `group` (purpose), a
//  `maker` (company/family), and a human-readable `blurb`, so an app can present
//  them grouped either way with descriptions. Every architecture is one
//  mlx-swift-lm can load (qwen3 / qwen3_5(_moe) / llama / mistral / phi / gemma3 /
//  gpt_oss).
//

import Foundation

/// A downloadable local MLX model.
public struct MLXModel: Sendable, Equatable {

    /// A section for grouping models by purpose.
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
    /// The company / model family (for grouping by maker).
    public let maker: String
    /// One-line, human-readable description.
    public let blurb: String
    /// Rough on-disk size once downloaded (MB, order-of-magnitude).
    public let approximateSizeMB: Int
    /// The model-weights license.
    public let license: String
    /// Whether it's light enough to run on an iPhone.
    public let runsOnMobile: Bool
    /// Which purpose section this model belongs to.
    public let group: Group

    public init(
        id: String, repoID: String, displayName: String, maker: String, blurb: String,
        approximateSizeMB: Int, license: String, runsOnMobile: Bool, group: Group = .general
    ) {
        self.id = id; self.repoID = repoID; self.displayName = displayName
        self.maker = maker; self.blurb = blurb; self.approximateSizeMB = approximateSizeMB
        self.license = license; self.runsOnMobile = runsOnMobile; self.group = group
    }

    // Makers (kept as constants so grouping/ordering stays consistent).
    private static let qwen = "Alibaba · Qwen"
    private static let meta = "Meta · Llama"
    private static let google = "Google · Gemma"
    private static let openai = "OpenAI"
    private static let mistral = "Mistral AI"
    private static let microsoft = "Microsoft · Phi"
    private static let hf = "Hugging Face"
    private static let community = "Community finetunes"

    /// Preferred display order of makers.
    public static let makerOrder = [qwen, openai, google, meta, mistral, microsoft, hf, community]

    // MARK: - General

    public static let qwen3_0_6B = MLXModel(
        id: "qwen3-0.6b", repoID: "mlx-community/Qwen3-0.6B-4bit", displayName: "Qwen3 0.6B",
        maker: qwen, blurb: "Tiny & fast — quick tests, runs anywhere.",
        approximateSizeMB: 450, license: "Apache-2.0", runsOnMobile: true)

    public static let qwen3_1_7B = MLXModel(
        id: "qwen3-1.7b", repoID: "mlx-community/Qwen3-1.7B-4bit", displayName: "Qwen3 1.7B",
        maker: qwen, blurb: "Small, multilingual, punches above its weight.",
        approximateSizeMB: 1_100, license: "Apache-2.0", runsOnMobile: true)

    public static let qwen3_4B = MLXModel(
        id: "qwen3-4b", repoID: "mlx-community/Qwen3-4B-4bit", displayName: "Qwen3 4B",
        maker: qwen, blurb: "Great everyday chat; strong for its size.",
        approximateSizeMB: 2_400, license: "Apache-2.0", runsOnMobile: true)

    public static let phi4Mini = MLXModel(
        id: "phi4-mini", repoID: "mlx-community/Phi-4-mini-instruct-4bit", displayName: "Phi-4-mini",
        maker: microsoft, blurb: "Microsoft's sharp little reasoner.",
        approximateSizeMB: 2_400, license: "MIT", runsOnMobile: true)

    public static let smolLM3_3B = MLXModel(
        id: "smollm3-3b", repoID: "mlx-community/SmolLM3-3B-4bit", displayName: "SmolLM3-3B",
        maker: hf, blurb: "Long-context specialist.",
        approximateSizeMB: 1_800, license: "Apache-2.0", runsOnMobile: true)

    public static let mistralSmall3 = MLXModel(
        id: "mistral-small-3", repoID: "mlx-community/Mistral-Small-3-Instruct-4bit", displayName: "Mistral Small 3",
        maker: mistral, blurb: "Fast, capable all-rounder.",
        approximateSizeMB: 4_500, license: "Apache-2.0", runsOnMobile: false)

    public static let llama3_2_1B = MLXModel(
        id: "llama-3.2-1b", repoID: "mlx-community/Llama-3.2-1B-Instruct-4bit", displayName: "Llama 3.2 1B",
        maker: meta, blurb: "Ultra-light — runs on any modern iPhone.",
        approximateSizeMB: 700, license: "Llama 3.2 Community", runsOnMobile: true)

    public static let llama3_2_3B = MLXModel(
        id: "llama-3.2-3b", repoID: "mlx-community/Llama-3.2-3B-Instruct-4bit", displayName: "Llama 3.2 3B",
        maker: meta, blurb: "The on-device chat sweet spot.",
        approximateSizeMB: 1_800, license: "Llama 3.2 Community", runsOnMobile: true)

    public static let llama3_1_8B = MLXModel(
        id: "llama-3.1-8b", repoID: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", displayName: "Llama 3.1 8B",
        maker: meta, blurb: "Strong general Llama; easy on a Mac.",
        approximateSizeMB: 4_500, license: "Llama 3.1 Community", runsOnMobile: false)

    public static let gptOSS20B = MLXModel(
        id: "gpt-oss-20b", repoID: "mlx-community/gpt-oss-20b-MXFP4-Q8", displayName: "GPT-OSS 20B",
        maker: openai, blurb: "OpenAI's open model — strong reasoning.",
        approximateSizeMB: 12_100, license: "Apache-2.0", runsOnMobile: false)

    public static let gemma3_27B = MLXModel(
        id: "gemma3-27b", repoID: "mlx-community/gemma-3-text-27b-it-4bit", displayName: "Gemma 3 27B",
        maker: google, blurb: "Google's flagship open model.",
        approximateSizeMB: 16_000, license: "Gemma Terms", runsOnMobile: false)

    // MARK: - Uncensored (abliterated; refusals removed)

    public static let josiefiedQwen3_4B = MLXModel(
        id: "josiefied-qwen3-4b", repoID: "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
        displayName: "Qwen3 4B (uncensored)", maker: qwen, blurb: "Uncensored Qwen3 — fast, cached.",
        approximateSizeMB: 2_400, license: "Apache-2.0", runsOnMobile: true, group: .uncensored)

    public static let josiefiedQwen3_8B = MLXModel(
        id: "josiefied-qwen3-8b", repoID: "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit",
        displayName: "Qwen3 8B (uncensored)", maker: qwen, blurb: "Uncensored Qwen3 — richer prose.",
        approximateSizeMB: 4_500, license: "Apache-2.0", runsOnMobile: false, group: .uncensored)

    public static let huihuiQwen3_5_9B = MLXModel(
        id: "qwen3.5-9b-uncensored", repoID: "huihui-ai/Huihui-Qwen3.5-9B-abliterated-mlx-4bit",
        displayName: "Qwen3.5 9B (uncensored)", maker: qwen, blurb: "Newer-generation Qwen, uncensored.",
        approximateSizeMB: 5_100, license: "Apache-2.0", runsOnMobile: false, group: .uncensored)

    public static let huihuiQwen3_5_27B = MLXModel(
        id: "qwen3.5-27b-uncensored", repoID: "mlx-community/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit",
        displayName: "Qwen3.5 27B (uncensored)", maker: qwen, blurb: "Big uncensored Qwen; needs ~24 GB.",
        approximateSizeMB: 16_100, license: "Apache-2.0", runsOnMobile: false, group: .uncensored)

    public static let qwen3_6_35B_moe = MLXModel(
        id: "qwen3.6-35b-uncensored", repoID: "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
        displayName: "Qwen3.6 35B MoE (uncensored)", maker: qwen, blurb: "35B MoE (3B active) — fast & strong.",
        approximateSizeMB: 20_400, license: "Apache-2.0", runsOnMobile: false, group: .uncensored)

    public static let llama3_1_8B_uncensored = MLXModel(
        id: "llama-3.1-8b-uncensored", repoID: "mlx-community/Meta-Llama-3.1-8B-Instruct-abliterated-4bit",
        displayName: "Llama 3.1 8B (uncensored)", maker: meta, blurb: "Uncensored — a different (Llama) voice.",
        approximateSizeMB: 4_500, license: "Llama 3.1 Community", runsOnMobile: false, group: .uncensored)

    public static let gemma3_12B_uncensored = MLXModel(
        id: "gemma3-12b-uncensored", repoID: "mlx-community/gemma-3-12b-it-qat-abliterated-lm-4bit",
        displayName: "Gemma 3 12B (uncensored)", maker: google, blurb: "Uncensored Gemma — distinct voice again.",
        approximateSizeMB: 7_400, license: "Gemma Terms", runsOnMobile: false, group: .uncensored)

    // MARK: - Creative & NSFW (fiction / roleplay finetunes; non-reasoning)

    public static let cydonia24B = MLXModel(
        id: "cydonia-24b", repoID: "mlx-community/Cydonia-24B-v3-4bit",
        displayName: "Cydonia 24B (writer)", maker: mistral, blurb: "Best-in-class creative & roleplay prose.",
        approximateSizeMB: 13_300, license: "Apache-2.0 (Mistral base)", runsOnMobile: false, group: .creative)

    public static let rocinante12B = MLXModel(
        id: "rocinante-12b", repoID: "McG-221/Rocinante-X-12B-v1-absolute-heresy-mlx-8Bit",
        displayName: "Rocinante 12B (writer)", maker: mistral, blurb: "Lighter creative/roleplay writer.",
        approximateSizeMB: 13_000, license: "Apache-2.0 (Nemo base)", runsOnMobile: false, group: .creative)

    public static let museWriter = MLXModel(
        id: "muse-writer", repoID: "Jurisprudence/Dirty-Muse-Writer-v01-Uncensored-Erotica-NSFW-mlx-2Bit",
        displayName: "Muse Writer 3B", maker: community, blurb: "Erotica specialist — on-topic but rough (2-bit).",
        approximateSizeMB: 2_900, license: "check repo", runsOnMobile: true, group: .creative)

    /// Every curated model, in section order.
    public static let all: [MLXModel] = [
        .qwen3_0_6B, .qwen3_1_7B, .qwen3_4B, .phi4Mini, .smolLM3_3B, .mistralSmall3,
        .llama3_2_1B, .llama3_2_3B, .llama3_1_8B, .gptOSS20B, .gemma3_27B,
        .josiefiedQwen3_4B, .josiefiedQwen3_8B, .huihuiQwen3_5_9B, .huihuiQwen3_5_27B,
        .qwen3_6_35B_moe, .llama3_1_8B_uncensored, .gemma3_12B_uncensored,
        .cydonia24B, .rocinante12B, .museWriter,
    ]

    /// Models in a given purpose section.
    public static func inGroup(_ group: Group) -> [MLXModel] { all.filter { $0.group == group } }
}
