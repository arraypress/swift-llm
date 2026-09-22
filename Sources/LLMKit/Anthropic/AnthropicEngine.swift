//
//  AnthropicEngine.swift
//  LLMKit
//
//  Claude via Anthropic's own Messages API. The API is not OpenAI-shaped —
//  instructions travel in a top-level `system`, attachments are typed blocks,
//  replies are lists of content blocks, tools are a loop, and the safety
//  system can end a successful request with a `refusal` stop reason — so
//  routing Claude through an OpenAI-compatible proxy loses all of that. This
//  engine speaks the native protocol with the same URLSession-only footprint
//  as `RemoteEngine`, and covers what an app can use of it: text, images and
//  documents (inline, by URL, or from the Files API), streaming, client and
//  server tools, thinking and effort, structured output, prompt caching, stop
//  sequences, token counting and usage.
//
//  Two doors: the neutral `LLMEngine` calls return text, and the typed
//  `send` / `run` / `streamEvents` return everything the API said.
//

import Foundation

/// Text, vision and document understanding via Anthropic's Messages API.
///
/// ```swift
/// let engine = AnthropicEngine(model: "claude-opus-5", apiKey: key)
/// let reply = try await LLM(engine).generate("Summarise this…")
///
/// var config = AnthropicConfiguration()
/// config.effort = .low
/// config.tools = [weatherTool]                          // with a handler
/// let agent = AnthropicEngine(model: "claude-opus-5", apiKey: key, configuration: config)
/// let response = try await agent.run([.user("Is it raining in Oslo?")])
/// print(response.text, response.usage.outputTokens)
/// ```
public struct AnthropicEngine: LLMEngine, ModelListing {

    /// The model id to request (e.g. `"claude-opus-5"`, `"claude-haiku-4-5"`).
    public let model: String

    /// The API root, `https://api.anthropic.com/v1` unless a gateway sits in
    /// front of it.
    public let baseURL: URL

    /// Anthropic-specific request settings.
    public let configuration: AnthropicConfiguration

    /// Whether `temperature` / `top_p` / `top_k` are sent. Models from the
    /// Opus 4.7 generation onward reject sampling parameters with a 400, so by
    /// default this follows `acceptsSamplingParameters(model:)`; pass a value
    /// to force it.
    public let sendsSamplingParameters: Bool

    let apiKey: String
    let urlSession: URLSession

    /// The default API root.
    public static let defaultBaseURL = URL(string: "https://api.anthropic.com/v1")!

    /// The `anthropic-version` header every request carries.
    public static let apiVersion = "2023-06-01"

    /// The output cap sent when `GenerationOptions.maxTokens` is nil. The API
    /// requires one; this is a ceiling, not a target, and costs nothing unless
    /// the model actually uses it.
    public static let defaultMaxTokens = 16_384

    /// Create a Claude engine.
    public init(
        model: String,
        apiKey: String,
        configuration: AnthropicConfiguration = AnthropicConfiguration(),
        baseURL: URL = AnthropicEngine.defaultBaseURL,
        sendsSamplingParameters: Bool? = nil,
        urlSession: URLSession = .shared
    ) {
        self.model = model
        self.apiKey = apiKey
        self.configuration = configuration
        self.baseURL = baseURL
        self.sendsSamplingParameters = sendsSamplingParameters ?? Self.acceptsSamplingParameters(model: model)
        self.urlSession = urlSession
    }

    // MARK: - LLMEngine

    public var isReady: Bool {
        get async { !apiKey.isEmpty }
    }

    public func prepare() async throws {
        if apiKey.isEmpty { throw LLMError.missingAPIKey }
    }

    /// The reply's text. Drives the tool loop when tools with handlers are
    /// configured; a refusal throws `LLMError.refused`.
    public func respond(to messages: [LLMMessage], options: GenerationOptions) async throws -> String {
        let response = try await run(messages, options: options)
        if response.stopReason == .refusal { throw LLMError.refused(response.refusalDetail) }
        let text = response.text
        guard !text.isEmpty else { throw LLMError.emptyResponse }
        return text
    }

    /// Live text: the `textDelta` events of `streamEvents`. Tool calls are
    /// not driven here; use `run` for those.
    public func streamResponse(to messages: [LLMMessage], options: GenerationOptions) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in streamEvents(messages, options: options) {
                        switch event {
                        case .textDelta(let text):
                            continuation.yield(text)
                        case .completed(let response) where response.stopReason == .refusal:
                            throw LLMError.refused(response.refusalDetail)
                        default:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Every model on the account, following `has_more` / `last_id` pages.
    public func availableModels() async throws -> [LLMModelInfo] {
        var models: [LLMModelInfo] = []
        var cursor: String?
        repeat {
            var components = URLComponents(url: baseURL.appendingPathComponent("models"), resolvingAgainstBaseURL: false)!
            components.queryItems = [URLQueryItem(name: "limit", value: "100")]
                + (cursor.map { [URLQueryItem(name: "after_id", value: $0)] } ?? [])
            let request = try HTTPTransport.request(components.url!, method: "GET", headers: headers)
            let data = try await HTTPTransport.data(for: request, session: urlSession)
            let page = try Self.parseModels(data)
            models += page.models
            cursor = page.nextCursor
        } while cursor != nil
        return models.newestFirst()
    }

    // MARK: - Typed API

    /// One call to the Messages API, with everything it returned. Tool calls
    /// come back as `toolUse` blocks for the caller to handle.
    public func send(_ messages: [LLMMessage], options: GenerationOptions = GenerationOptions()) async throws -> AnthropicResponse {
        try await send(
            turns: AnthropicRequestBuilder.turns(from: messages, citations: configuration.citations),
            system: AnthropicRequestBuilder.system(from: messages, cache: configuration.cacheSystemPrompt, ttl: configuration.cacheTTL),
            options: options
        )
    }

    /// Call the model and run its tool calls until it stops asking.
    ///
    /// Tools with a handler are executed (in parallel when the model asks for
    /// several) and their results sent back; a call to a tool without a
    /// handler, a `max_tokens` cut, or a refusal ends the loop and returns
    /// the reply as is. A `pause_turn` from server tools is resumed
    /// automatically. Gives up after `configuration.maxToolIterations`.
    public func run(_ messages: [LLMMessage], options: GenerationOptions = GenerationOptions()) async throws -> AnthropicResponse {
        var turns = AnthropicRequestBuilder.turns(from: messages, citations: configuration.citations)
        let system = AnthropicRequestBuilder.system(from: messages, cache: configuration.cacheSystemPrompt, ttl: configuration.cacheTTL)
        let handlers = Dictionary(configuration.tools.compactMap { tool in tool.handler.map { (tool.name, $0) } },
                                  uniquingKeysWith: { first, _ in first })

        for _ in 0..<max(1, configuration.maxToolIterations) {
            let response = try await send(turns: turns, system: system, options: options)
            switch response.stopReason {
            case .toolUse:
                let uses = response.toolUses
                guard !uses.isEmpty, uses.allSatisfy({ handlers[$0.name] != nil }) else { return response }
                turns.append(AnthropicRequestBuilder.assistantTurn(echoing: response))
                let results = await Self.execute(uses, handlers: handlers)
                turns.append(AnthropicRequestBuilder.toolResultTurn(results))
            case .pauseTurn:
                turns.append(AnthropicRequestBuilder.assistantTurn(echoing: response))
            default:
                return response
            }
        }
        throw LLMError.requestFailed("the tool loop did not finish within \(configuration.maxToolIterations) turns")
    }

    /// The reply as it streams: text and thinking deltas, tool-call starts and
    /// input fragments, then `.completed` with the assembled reply.
    public func streamEvents(_ messages: [LLMMessage], options: GenerationOptions = GenerationOptions()) -> AsyncThrowingStream<AnthropicStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let body = AnthropicRequestBuilder.body(
                        model: model,
                        turns: AnthropicRequestBuilder.turns(from: messages, citations: configuration.citations),
                        system: AnthropicRequestBuilder.system(from: messages, cache: configuration.cacheSystemPrompt, ttl: configuration.cacheTTL),
                        options: options,
                        configuration: configuration,
                        sendsSamplingParameters: sendsSamplingParameters,
                        stream: true
                    )
                    let request = try HTTPTransport.request(
                        baseURL.appendingPathComponent("messages"), method: "POST", headers: headers,
                        body: body.anyValue as? [String: Any]
                    )
                    var accumulator = AnthropicStreamAccumulator()
                    for try await line in try await HTTPTransport.lines(for: request, session: urlSession) {
                        for event in try accumulator.consume(line: line) {
                            continuation.yield(event)
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

    /// How many input tokens a conversation costs on this model, before
    /// sending it. Free.
    public func countTokens(_ messages: [LLMMessage]) async throws -> AnthropicTokenCount {
        let body = AnthropicRequestBuilder.countTokensBody(
            model: model,
            turns: AnthropicRequestBuilder.turns(from: messages, citations: configuration.citations),
            system: AnthropicRequestBuilder.system(from: messages, cache: configuration.cacheSystemPrompt, ttl: configuration.cacheTTL),
            configuration: configuration
        )
        let request = try HTTPTransport.request(
            baseURL.appendingPathComponent("messages/count_tokens"), method: "POST", headers: headers,
            body: body.anyValue as? [String: Any]
        )
        return AnthropicTokenCount.parse(try JSONValue.parse(await HTTPTransport.data(for: request, session: urlSession)))
    }

    // MARK: - Internals

    /// One POST to `/messages` for already-assembled turns.
    func send(turns: [JSONValue], system: JSONValue?, options: GenerationOptions) async throws -> AnthropicResponse {
        let body = AnthropicRequestBuilder.body(
            model: model, turns: turns, system: system, options: options,
            configuration: configuration, sendsSamplingParameters: sendsSamplingParameters, stream: false
        )
        let request = try HTTPTransport.request(
            baseURL.appendingPathComponent("messages"), method: "POST", headers: headers,
            body: body.anyValue as? [String: Any]
        )
        let data = try await HTTPTransport.data(for: request, session: urlSession)
        return try AnthropicResponse.parse(JSONValue.parse(data))
    }

    /// Run every requested tool concurrently; results keep the request order.
    static func execute(
        _ uses: [AnthropicResponse.ToolUse],
        handlers: [String: AnthropicTool.Handler]
    ) async -> [(id: String, output: String, isError: Bool)] {
        await withTaskGroup(of: (Int, String, Bool).self) { group in
            for (index, use) in uses.enumerated() {
                guard let handler = handlers[use.name] else { continue }
                group.addTask {
                    do {
                        return (index, try await handler(use.input), false)
                    } catch {
                        return (index, error.localizedDescription, true)
                    }
                }
            }
            var results: [(Int, String, Bool)] = []
            for await result in group { results.append(result) }
            return results.sorted { $0.0 < $1.0 }.map { (uses[$0.0].id, $0.1, $0.2) }
        }
    }

    /// Authentication, version and beta headers.
    var headers: [String: String] {
        AnthropicRequestBuilder.headers(apiKey: apiKey, configuration: configuration)
    }

    // MARK: - Pure helpers (testable without the network)

    /// Whether `model` still accepts `temperature` / `top_p` / `top_k`.
    ///
    /// Anthropic removed sampling parameters from Opus 4.7 onward — Opus 4.7,
    /// 4.8 and 5, Sonnet 5, and the Fable / Mythos models all answer 400 —
    /// while Haiku 4.5, the 4.6 pair and older models still take them. A
    /// substring test on the id is deliberately loose so dated variants of the
    /// same models match.
    public static func acceptsSamplingParameters(model: String) -> Bool {
        let id = model.lowercased()
        let rejecting = ["fable", "mythos", "opus-5", "sonnet-5", "opus-4-7", "opus-4-8"]
        return !rejecting.contains { id.contains($0) }
    }

    /// The request body for plain messages and the default configuration —
    /// the shape the neutral engine calls send.
    static func requestBody(
        model: String,
        messages: [LLMMessage],
        options: GenerationOptions,
        sendsSamplingParameters: Bool,
        stream: Bool = false,
        configuration: AnthropicConfiguration = AnthropicConfiguration()
    ) -> [String: Any] {
        let body = AnthropicRequestBuilder.body(
            model: model,
            turns: AnthropicRequestBuilder.turns(from: messages, citations: configuration.citations),
            system: AnthropicRequestBuilder.system(from: messages, cache: configuration.cacheSystemPrompt, ttl: configuration.cacheTTL),
            options: options,
            configuration: configuration,
            sendsSamplingParameters: sendsSamplingParameters,
            stream: stream
        )
        return body.anyValue as? [String: Any] ?? [:]
    }

    /// The reply's text blocks joined, after checking the stop reason: a
    /// `refusal` is a successful HTTP reply that carries no usable answer.
    static func parseContent(_ data: Data) throws -> String {
        let response = try AnthropicResponse.parse(JSONValue.parse(data))
        if response.stopReason == .refusal { throw LLMError.refused(response.refusalDetail) }
        guard !response.text.isEmpty else { throw LLMError.emptyResponse }
        return response.text
    }

    /// The text carried by one streamed line, for callers that only want
    /// text: a `text_delta`, else `nil`. An `error` event throws
    /// `requestFailed`; a `message_delta` that stops for `refusal` throws
    /// `refused`.
    static func streamDelta(_ line: String) throws -> String? {
        guard let object = ServerSentEvents.jsonObject(of: line), let event = JSONValue(any: object) else { return nil }
        switch event["type"]?.stringValue {
        case "content_block_delta":
            guard event["delta"]?["type"]?.stringValue == "text_delta" else { return nil }
            return event["delta"]?["text"]?.stringValue
        case "message_delta":
            if event["delta"]?["stop_reason"]?.stringValue == "refusal" {
                let details = event["delta"]?["stop_details"]
                throw LLMError.refused(
                    details?["explanation"]?.stringValue
                        ?? details?["category"]?.stringValue.map { "declined for category \($0)" }
                        ?? "the request was declined by the model's safety system"
                )
            }
            return nil
        case "error":
            throw LLMError.requestFailed(event["error"]?["message"]?.stringValue ?? "stream error")
        default:
            return nil
        }
    }

    /// Parse one page of `GET /models`: `data[]` with `display_name` and an
    /// ISO-8601 `created_at`, plus the cursor for the next page when
    /// `has_more` is set.
    static func parseModels(_ data: Data) throws -> (models: [LLMModelInfo], nextCursor: String?) {
        let reply = try JSONValue.parse(data)
        guard let entries = reply["data"]?.arrayValue else {
            throw LLMError.decodingFailed("no `data` array in the models reply")
        }
        let models = entries.compactMap { entry -> LLMModelInfo? in
            guard let id = entry["id"]?.stringValue, !id.isEmpty else { return nil }
            let created = ISO8601.date(from: entry["created_at"]?.stringValue)
            return LLMModelInfo(id: id, displayName: entry["display_name"]?.stringValue, created: created)
        }
        let hasMore = reply["has_more"]?.boolValue ?? false
        return (models, hasMore ? reply["last_id"]?.stringValue : nil)
    }
}
