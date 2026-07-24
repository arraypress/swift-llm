//
//  RemoteEndpoint.swift
//  LLMKit
//
//  Describes an OpenAI-compatible chat-completions endpoint. Because that format
//  is the lingua franca, one description covers OpenAI, Groq, OpenRouter,
//  Cloudflare AI Gateway, and local servers (Ollama / LM Studio).
//

import Foundation

/// An OpenAI-compatible `/chat/completions` endpoint plus its auth style.
public struct RemoteEndpoint: Sendable {

    /// The full chat-completions URL.
    public var url: URL

    /// The HTTP header the API key goes in (e.g. `Authorization`,
    /// `cf-aig-authorization`).
    public var authHeaderField: String

    /// The scheme prefix for the key (e.g. `Bearer`), or empty for none.
    public var authScheme: String

    /// Create an endpoint.
    public init(url: URL, authHeaderField: String = "Authorization", authScheme: String = "Bearer") {
        self.url = url
        self.authHeaderField = authHeaderField
        self.authScheme = authScheme
    }

    // MARK: - Presets

    /// OpenAI (`api.openai.com`).
    public static let openAI = RemoteEndpoint(url: URL(string: "https://api.openai.com/v1/chat/completions")!)

    /// Groq.
    public static let groq = RemoteEndpoint(url: URL(string: "https://api.groq.com/openai/v1/chat/completions")!)

    /// OpenRouter.
    public static let openRouter = RemoteEndpoint(url: URL(string: "https://openrouter.ai/api/v1/chat/completions")!)

    /// A local OpenAI-compatible server (Ollama / LM Studio), default port.
    public static func localServer(port: Int = 11434) -> RemoteEndpoint {
        RemoteEndpoint(url: URL(string: "http://localhost:\(port)/v1/chat/completions")!, authScheme: "")
    }

    /// A Cloudflare AI Gateway route to `provider`, using the `cf-aig-authorization` header.
    public static func cloudflareGateway(accountID: String, gateway: String, provider: String = "compat") -> RemoteEndpoint {
        let url = URL(string: "https://gateway.ai.cloudflare.com/v1/\(accountID)/\(gateway)/\(provider)/chat/completions")!
        return RemoteEndpoint(url: url, authHeaderField: "cf-aig-authorization", authScheme: "Bearer")
    }

    /// Any other OpenAI-compatible base URL (appends `/chat/completions` if needed).
    public static func custom(_ url: URL, authHeaderField: String = "Authorization", authScheme: String = "Bearer") -> RemoteEndpoint {
        RemoteEndpoint(url: url, authHeaderField: authHeaderField, authScheme: authScheme)
    }
}
