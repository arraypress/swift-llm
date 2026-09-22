//
//  RemoteEngine.swift
//  LLMKit
//
//  An LLM engine backed by any OpenAI-compatible cloud endpoint. Supports text,
//  vision (via the `image_url` content block), token streaming (`stream: true`
//  server-sent events) and model listing (`GET /models`), and works with
//  OpenAI, Cloudflare AI Gateway, Groq, OpenRouter and local servers. Pure
//  URLSession — no external dependencies. Anthropic has its own engine
//  (`AnthropicEngine`) because its API is not OpenAI-shaped.
//

import Foundation

/// Text (and vision) generation via an OpenAI-compatible endpoint.
public struct RemoteEngine: LLMEngine, ModelListing {

    /// The model id to request (e.g. `"gpt-4o"`, `"llama3.2"`).
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
        try Self.checkAttachments(in: messages)
        let request = try HTTPTransport.request(
            endpoint.url,
            method: "POST",
            headers: headers,
            body: Self.requestBody(model: model, messages: messages, options: options, endpoint: endpoint)
        )
        let data = try await HTTPTransport.data(for: request, session: urlSession)
        return try Self.parseContent(data)
    }

    /// Live tokens: the same request with `stream: true`, read as server-sent
    /// events until the `[DONE]` sentinel or the connection closes.
    public func streamResponse(to messages: [LLMMessage], options: GenerationOptions) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Self.checkAttachments(in: messages)
                    let request = try HTTPTransport.request(
                        endpoint.url,
                        method: "POST",
                        headers: headers,
                        body: Self.requestBody(model: model, messages: messages, options: options,
                                               endpoint: endpoint, stream: true)
                    )
                    let lines = try await HTTPTransport.lines(for: request, session: urlSession)
                    for try await line in lines {
                        if ServerSentEvents.dataPayload(of: line) == ServerSentEvents.done { break }
                        if let delta = try Self.streamDelta(line), !delta.isEmpty {
                            continuation.yield(delta)
                        }
                    }
                    continuation.finish()
                } catch let error as LLMError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: LLMError.requestFailed(error.localizedDescription))
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func availableModels() async throws -> [LLMModelInfo] {
        let request = try HTTPTransport.request(endpoint.resolvedModelsURL, method: "GET", headers: headers)
        let data = try await HTTPTransport.data(for: request, session: urlSession)
        return try Self.parseModels(data).newestFirst()
    }

    /// The auth header, when the endpoint wants one.
    private var headers: [String: String] {
        guard !apiKey.isEmpty else { return [:] }
        let value = endpoint.authScheme.isEmpty ? apiKey : "\(endpoint.authScheme) \(apiKey)"
        return [endpoint.authHeaderField: value]
    }

    // MARK: - Pure helpers (testable without the network)

    /// The first attachment this endpoint has no content part for. The
    /// OpenAI chat format carries images (inline or by URL) and text; PDFs and
    /// uploaded files are Anthropic-only, and dropping them silently would
    /// answer a question the model never saw.
    static func checkAttachments(in messages: [LLMMessage]) throws {
        for attachment in messages.flatMap(\.allAttachments) {
            switch attachment {
            case .image, .imageURL, .text:
                continue
            case .pdf, .pdfURL, .file:
                throw LLMError.unavailable("this endpoint accepts images and text documents only; PDFs and uploaded files need AnthropicEngine")
            }
        }
    }

    /// Build the OpenAI-compatible request body. Messages with attachments
    /// become the array-content form: `image_url` parts (base64 data URLs or
    /// plain URLs) and text parts, with text documents inlined under their
    /// title. Unsupported attachment kinds are left out here; `checkAttachments`
    /// rejects them before a request is made.
    static func requestBody(
        model: String,
        messages: [LLMMessage],
        options: GenerationOptions,
        endpoint: RemoteEndpoint,
        stream: Bool = false
    ) -> [String: Any] {
        let encoded: [[String: Any]] = messages.map { message in
            let attachments = message.allAttachments
            if attachments.isEmpty {
                return ["role": message.role.rawValue, "content": message.text]
            }
            var content: [[String: Any]] = attachments.compactMap { attachment in
                switch attachment {
                case .image(let data):
                    let url = "data:\(ImageData.mediaType(of: data));base64,\(data.base64EncodedString())"
                    return ["type": "image_url", "image_url": ["url": url]]
                case .imageURL(let url):
                    return ["type": "image_url", "image_url": ["url": url.absoluteString]]
                case .text(let text, let title):
                    return ["type": "text", "text": inlineDocument(text, title: title)]
                case .pdf, .pdfURL, .file:
                    return nil
                }
            }
            content.append(["type": "text", "text": message.text])
            return ["role": message.role.rawValue, "content": content]
        }

        var body: [String: Any] = [
            "model": model,
            "messages": encoded,
        ]
        if endpoint.acceptsSamplingParameters(model: model) {
            body["temperature"] = options.temperature
            if let topP = options.topP { body["top_p"] = topP }
        }
        if let maxTokens = options.maxTokens { body[endpoint.maxTokensField] = maxTokens }
        if stream { body["stream"] = true }
        return body
    }

    /// A text document as prompt text, fenced under its title so the model
    /// can tell it from the question.
    static func inlineDocument(_ text: String, title: String?) -> String {
        "<document\(title.map { " title=\"\($0)\"" } ?? "")>\n\(text)\n</document>"
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

    /// The text carried by one streamed line: `choices[0].delta.content`, or
    /// `nil` for framing lines, the `[DONE]` sentinel, and deltas with no text
    /// (role announcements, finish records). An in-band `{"error": …}` record
    /// throws, because servers report mid-stream failures that way with a 200.
    static func streamDelta(_ line: String) throws -> String? {
        guard let object = ServerSentEvents.jsonObject(of: line) else { return nil }
        if let error = object["error"] as? [String: Any] {
            throw LLMError.requestFailed((error["message"] as? String) ?? "stream error")
        }
        guard let choices = object["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any] else {
            return nil
        }
        return delta["content"] as? String
    }

    /// Parse an OpenAI-compatible `GET /models` reply: `data[].id`, with
    /// `created` as Unix seconds when present.
    static func parseModels(_ data: Data) throws -> [LLMModelInfo] {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let entries = object["data"] as? [[String: Any]] else {
            throw LLMError.decodingFailed("no `data` array in the models reply")
        }
        return entries.compactMap { entry in
            guard let id = entry["id"] as? String, !id.isEmpty else { return nil }
            let created = (entry["created"] as? TimeInterval).map(Date.init(timeIntervalSince1970:))
            return LLMModelInfo(id: id, displayName: entry["name"] as? String, created: created)
        }
    }
}
