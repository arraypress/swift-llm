//
//  RemoteEndpoint.swift
//  LLMKit
//
//  Describes an OpenAI-compatible chat-completions endpoint. Because that format
//  is the lingua franca, one description covers OpenAI, Groq, OpenRouter,
//  Cloudflare AI Gateway, and local servers (Ollama / LM Studio). The dialect
//  differences that do exist — which field caps output, where `/models` lives —
//  are carried here so `RemoteEngine` never has to guess by hostname.
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

    /// The request field that caps output length.
    ///
    /// OpenAI and Groq moved to `max_completion_tokens` and their newer models
    /// reject the old name (`gpt-5` answers 400 "Unsupported parameter:
    /// max_tokens"); every other compatible server still speaks `max_tokens`.
    public var maxTokensField: String

    /// Where `GET …/models` lives, when it isn't beside the chat URL.
    public var modelsURL: URL?

    /// Create an endpoint.
    public init(
        url: URL,
        authHeaderField: String = "Authorization",
        authScheme: String = "Bearer",
        maxTokensField: String = "max_tokens",
        modelsURL: URL? = nil
    ) {
        self.url = url
        self.authHeaderField = authHeaderField
        self.authScheme = authScheme
        self.maxTokensField = maxTokensField
        self.modelsURL = modelsURL
    }

    /// The models URL: `modelsURL` if set, else the chat URL with its
    /// `chat/completions` tail replaced by `models` (`…/v1/models`).
    public var resolvedModelsURL: URL {
        if let modelsURL { return modelsURL }
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return url }
        var parts = components.path.split(separator: "/").map(String.init)
        if parts.suffix(2) == ["chat", "completions"] {
            parts.removeLast(2)
        } else if !parts.isEmpty {
            parts.removeLast()
        }
        components.path = "/" + (parts + ["models"]).joined(separator: "/")
        components.query = nil
        return components.url ?? url
    }

    // MARK: - Presets

    /// OpenAI (`api.openai.com`).
    public static let openAI = RemoteEndpoint(
        url: URL(string: "https://api.openai.com/v1/chat/completions")!,
        maxTokensField: "max_completion_tokens"
    )

    /// Groq.
    public static let groq = RemoteEndpoint(
        url: URL(string: "https://api.groq.com/openai/v1/chat/completions")!,
        maxTokensField: "max_completion_tokens"
    )

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
