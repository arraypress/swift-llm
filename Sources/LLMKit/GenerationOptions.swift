//
//  GenerationOptions.swift
//  LLMKit
//
//  Per-request generation configuration, mapped by each engine to its own knobs.
//

import Foundation

/// Options for a single generation request.
public struct GenerationOptions: Sendable {

    /// Sampling temperature (`0` = deterministic, higher = more varied).
    public var temperature: Double

    /// Maximum tokens to generate, or `nil` for the engine default.
    public var maxTokens: Int?

    /// Nucleus-sampling top-p, or `nil` for the engine default.
    public var topP: Double?

    /// Create generation options.
    public init(temperature: Double = 0.7, maxTokens: Int? = nil, topP: Double? = nil) {
        self.temperature = max(0, temperature)
        self.maxTokens = maxTokens
        self.topP = topP
    }

    /// Deterministic defaults (temperature 0) — good for structured extraction.
    public static let deterministic = GenerationOptions(temperature: 0)
}
