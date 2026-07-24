//
//  RemoteEngine.swift
//  LLMKit
//
//  An LLM engine backed by any OpenAI-compatible cloud endpoint. Supports text
//  and vision (via the `image_url` content block), and works with OpenAI, Claude
//  (through a compatible proxy), Cloudflare AI Gateway, Groq/OpenRouter, and
//  local servers. Pure URLSession — no external dependencies.
//

import Foundation

/// Text (and vision) generation via an OpenAI-compatible endpoint.
public struct RemoteEngine: LLMEngine {

    /// The model id to request (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
    public let model: String

    /// The endpoint + auth style.
    public let endpoint: RemoteEndpoint

    private let apiKey: String
    private let urlSession: URLSession

    /// Create a cloud engine.
    public init(model: String, endpoint: RemoteEndpoint, apiKey: String, urlSession: URLSession = .shared) {
        self.model = model
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.urlSession = urlSession
    }

    public var isReady: Bool {
        get async { endpoint.authScheme.isEmpty || !apiKey.isEmpty }
    }

    public func prepare() async throws {
        // Local servers may not need a key; hosted providers do.
        if !endpoint.authScheme.isEmpty && apiKey.isEmpty {
            throw LLMError.missingAPIKey
        }
    }

    public func respond(to messages: [LLMMessage], options: GenerationOptions) async throws -> String {
        var request = URLRequest(url: endpoint.url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            let value = endpoint.authScheme.isEmpty ? apiKey : "\(endpoint.authScheme) \(apiKey)"
            request.setValue(value, forHTTPHeaderField: endpoint.authHeaderField)
        }
        request.timeoutInterval = 60
        request.httpBody = try JSONSerialization.data(withJSONObject: Self.requestBody(model: model, messages: messages, options: options))

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await urlSession.data(for: request)
        } catch {
            throw LLMError.requestFailed(error.localizedDescription)
        }

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw LLMError.requestFailed("HTTP \(http.statusCode): \(body.prefix(300))")
        }

        return try Self.parseContent(data)
    }

    // MARK: - Pure helpers (testable without the network)

    /// Build the OpenAI-compatible request body. Messages with images become the
    /// array-content form with `image_url` blocks (base64 data URLs).
    static func requestBody(model: String, messages: [LLMMessage], options: GenerationOptions) -> [String: Any] {
        let encoded: [[String: Any]] = messages.map { message in
            if message.images.isEmpty {
                return ["role": message.role.rawValue, "content": message.text]
            }
            var content: [[String: Any]] = message.images.map { image in
                ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(image.base64EncodedString())"]]
            }
            content.append(["type": "text", "text": message.text])
            return ["role": message.role.rawValue, "content": content]
        }

        var body: [String: Any] = [
            "model": model,
            "messages": encoded,
            "temperature": options.temperature,
        ]
        if let maxTokens = options.maxTokens { body["max_tokens"] = maxTokens }
        if let topP = options.topP { body["top_p"] = topP }
        return body
    }

    /// Extract `choices[0].message.content` from an OpenAI-compatible response.
    static func parseContent(_ data: Data) throws -> String {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LLMError.emptyResponse
        }
        guard let choices = object["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String,
              !content.isEmpty else {
            throw LLMError.emptyResponse
        }
        return content
    }
}
