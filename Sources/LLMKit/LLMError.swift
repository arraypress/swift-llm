//
//  LLMError.swift
//  LLMKit
//

import Foundation

/// An error from an LLM engine.
public enum LLMError: Error, LocalizedError, Equatable, Sendable {

    /// The engine isn't ready (call `prepare()` first).
    case notReady

    /// The model/engine isn't available on this device (with a reason).
    case unavailable(String)

    /// No API key was provided for a cloud engine.
    case missingAPIKey

    /// The network/request failed.
    case requestFailed(String)

    /// The engine returned no usable content.
    case emptyResponse

    /// Structured output couldn't be decoded from the response.
    case decodingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notReady:
            return "The engine isn't ready. Call prepare() first."
        case .unavailable(let reason):
            return "The model isn't available: \(reason)"
        case .missingAPIKey:
            return "No API key was provided for the cloud engine."
        case .requestFailed(let detail):
            return "The request failed: \(detail)"
        case .emptyResponse:
            return "The model returned an empty response."
        case .decodingFailed(let detail):
            return "Couldn't decode structured output: \(detail)"
        }
    }
}
