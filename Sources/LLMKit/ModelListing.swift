//
//  ModelListing.swift
//  LLMKit
//
//  Hard-coded model pickers go stale within a generation. Providers that can
//  enumerate their catalogue expose it here, so an app can offer a live list
//  and keep only a short curated fallback for when the network is away.
//

import Foundation

/// A model a provider serves, as its own listing reports it.
public struct LLMModelInfo: Sendable, Hashable, Identifiable {

    /// The identifier to pass as `model` in requests.
    public let id: String

    /// The provider's display name, or the id when it has none.
    public let displayName: String

    /// When the provider says the model was created, if it says.
    public let created: Date?

    /// Create a model description.
    public init(id: String, displayName: String? = nil, created: Date? = nil) {
        self.id = id
        self.displayName = displayName ?? id
        self.created = created
    }
}

/// An engine whose provider can enumerate its models.
///
/// `RemoteEngine` (any OpenAI-compatible `/models`) and `AnthropicEngine`
/// conform; on-device engines have nothing to list.
public protocol ModelListing: Sendable {

    /// The models the provider serves, newest first.
    func availableModels() async throws -> [LLMModelInfo]
}

public extension LLM {

    /// The models the engine's provider serves, newest first.
    ///
    /// Throws `LLMError.unavailable` for engines with no listing (Apple, MLX).
    func availableModels() async throws -> [LLMModelInfo] {
        guard let listing = engine as? ModelListing else {
            throw LLMError.unavailable("this engine doesn't list models")
        }
        return try await listing.availableModels()
    }
}

extension Array where Element == LLMModelInfo {

    /// Newest first; models without a date keep their listing order after
    /// the dated ones, so an undated provider still returns something stable.
    func newestFirst() -> [LLMModelInfo] {
        let dated = filter { $0.created != nil }.sorted { $0.created! > $1.created! }
        let undated = filter { $0.created == nil }
        return dated + undated
    }
}
