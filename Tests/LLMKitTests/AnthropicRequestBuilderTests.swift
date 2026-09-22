//
//  AnthropicRequestBuilderTests.swift
//  LLMKitTests
//
//  Every request shape the engine can send, asserted offline against the
//  field names in Anthropic's reference: attachments of each kind, caching,
//  thinking, effort, structured output, tools, and the headers.
//

import XCTest
@testable import LLMKit

final class AnthropicRequestBuilderTests: XCTestCase {

    private let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0])
    private let pdf = Data("%PDF-1.4 fake".utf8)

    // MARK: - Attachments

    func testAttachmentBlocks_everyKind() {
        let url = URL(string: "https://example.com/a.jpg")!
        XCTAssertEqual(
            AnthropicRequestBuilder.block(for: .imageURL(url), citations: false),
            ["type": "image", "source": ["type": "url", "url": "https://example.com/a.jpg"]]
        )
        XCTAssertEqual(
            AnthropicRequestBuilder.block(for: .image(png), citations: false),
            ["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": .string(png.base64EncodedString())]]
        )
        XCTAssertEqual(
            AnthropicRequestBuilder.block(for: .pdf(pdf, title: "Report"), citations: true),
            ["type": "document", "title": "Report", "citations": ["enabled": true],
             "source": ["type": "base64", "media_type": "application/pdf", "data": .string(pdf.base64EncodedString())]]
        )
        XCTAssertEqual(
            AnthropicRequestBuilder.block(for: .pdfURL(url, title: nil), citations: false),
            ["type": "document", "source": ["type": "url", "url": "https://example.com/a.jpg"]]
        )
        XCTAssertEqual(
            AnthropicRequestBuilder.block(for: .text("hello", title: "Notes"), citations: true),
            ["type": "document", "title": "Notes", "citations": ["enabled": true],
             "source": ["type": "text", "media_type": "text/plain", "data": "hello"]]
        )
        XCTAssertEqual(
            AnthropicRequestBuilder.block(for: .file(id: "file_1", kind: .image), citations: false),
            ["type": "image", "source": ["type": "file", "file_id": "file_1"]]
        )
        XCTAssertEqual(
            AnthropicRequestBuilder.block(for: .file(id: "file_2", kind: .document), citations: true),
            ["type": "document", "citations": ["enabled": true], "source": ["type": "file", "file_id": "file_2"]]
        )
    }

    func testTurns_attachmentsPrecedeText_andImagesFieldIsMerged() {
        let message = LLMMessage.user("what?", images: [png], attachments: [.text("doc", title: nil)])
        let turns = AnthropicRequestBuilder.turns(from: [message], citations: false)
        let content = turns[0]["content"]?.arrayValue
        XCTAssertEqual(content?.map { $0["type"]?.stringValue }, ["image", "document", "text"])
    }

    func testTurns_attachmentOnlyMessageHasNoTextBlock() {
        let turns = AnthropicRequestBuilder.turns(from: [.user("", images: [png])], citations: false)
        XCTAssertEqual(turns[0]["content"]?.arrayValue?.count, 1)
    }

    // MARK: - System and caching

    func testSystem_plainAndCached() {
        let messages: [LLMMessage] = [.system("A"), .user("x"), .system("B")]
        XCTAssertEqual(AnthropicRequestBuilder.system(from: messages, cache: false, ttl: .fiveMinutes), "A\nB")
        XCTAssertEqual(
            AnthropicRequestBuilder.system(from: messages, cache: true, ttl: .oneHour),
            [["type": "text", "text": "A\nB", "cache_control": ["type": "ephemeral", "ttl": "1h"]]]
        )
        XCTAssertEqual(
            AnthropicRequestBuilder.system(from: messages, cache: true, ttl: .fiveMinutes),
            [["type": "text", "text": "A\nB", "cache_control": ["type": "ephemeral"]]]
        )
        XCTAssertNil(AnthropicRequestBuilder.system(from: [.user("x")], cache: true, ttl: .oneHour))
    }

    // MARK: - Body

    func testBody_configurationFields() {
        var configuration = AnthropicConfiguration()
        configuration.thinking = .adaptive(display: .summarized)
        configuration.effort = .xhigh
        configuration.outputSchema = ["type": "object", "properties": ["a": ["type": "string"]], "required": ["a"], "additionalProperties": false]
        configuration.stopSequences = ["END"]
        configuration.userID = "user-42"
        configuration.topK = 5
        configuration.serviceTier = .standardOnly
        configuration.inferenceGeo = "us"
        configuration.refusalFallbacks = .default
        configuration.toolChoice = .auto(disableParallelToolUse: true)
        configuration.serverTools = [.webSearch(maxUses: 3, allowedDomains: ["apple.com"])]
        configuration.tools = [AnthropicTool(name: "t", description: "d", inputSchema: ["type": "object"], strict: true, eagerInputStreaming: true)]

        let body = AnthropicRequestBuilder.body(
            model: "claude-opus-5", turns: [["role": "user", "content": "hi"]], system: nil,
            options: GenerationOptions(temperature: 0.1, maxTokens: 99, topP: 0.5),
            configuration: configuration, sendsSamplingParameters: true, stream: true
        )
        XCTAssertEqual(body["max_tokens"], 99)
        XCTAssertEqual(body["stream"], true)
        // Thinking is on, so sampling parameters must be absent (the API
        // answers 400 to any temperature but 1 while thinking).
        XCTAssertNil(body["temperature"])
        XCTAssertNil(body["top_p"])
        XCTAssertNil(body["top_k"])
        XCTAssertEqual(body["thinking"], ["type": "adaptive", "display": "summarized"])
        XCTAssertEqual(body["output_config"]?["effort"], "xhigh")
        XCTAssertEqual(body["output_config"]?["format"]?["type"], "json_schema")
        XCTAssertEqual(body["output_config"]?["format"]?["schema"]?["required"], ["a"])
        XCTAssertEqual(body["stop_sequences"], ["END"])
        XCTAssertEqual(body["metadata"], ["user_id": "user-42"])
        XCTAssertEqual(body["service_tier"], "standard_only")
        XCTAssertEqual(body["inference_geo"], "us")
        XCTAssertEqual(body["fallbacks"], "default")
        XCTAssertEqual(body["tool_choice"], ["type": "auto", "disable_parallel_tool_use": true])
        let tools = body["tools"]?.arrayValue
        XCTAssertEqual(tools?.count, 2)
        XCTAssertEqual(tools?[0]["name"], "t")
        XCTAssertEqual(tools?[0]["strict"], true)
        XCTAssertEqual(tools?[0]["eager_input_streaming"], true)
        XCTAssertEqual(tools?[1], ["type": "web_search_20260209", "name": "web_search", "max_uses": 3, "allowed_domains": ["apple.com"]])
    }

    func testBody_samplingGateDropsTopK() {
        var configuration = AnthropicConfiguration()
        configuration.topK = 5
        let body = AnthropicRequestBuilder.body(
            model: "claude-opus-5", turns: [], system: nil, options: GenerationOptions(),
            configuration: configuration, sendsSamplingParameters: false, stream: false
        )
        XCTAssertNil(body["temperature"])
        XCTAssertNil(body["top_k"])
        XCTAssertNil(body["stream"])
        XCTAssertNil(body["output_config"])
        XCTAssertNil(body["tools"])
        XCTAssertEqual(body["max_tokens"], .number(Double(AnthropicEngine.defaultMaxTokens)))
    }

    func testBody_thinkingSuppressesSampling() {
        var configuration = AnthropicConfiguration()
        configuration.thinking = .budget(tokens: 1024)
        configuration.topK = 3
        let body = AnthropicRequestBuilder.body(
            model: "claude-haiku-4-5", turns: [], system: nil,
            options: GenerationOptions(temperature: 0, topP: 0.9),
            configuration: configuration, sendsSamplingParameters: true, stream: false
        )
        XCTAssertNil(body["temperature"], "the API rejects any temperature but 1 while thinking")
        XCTAssertNil(body["top_p"])
        XCTAssertNil(body["top_k"])
        XCTAssertEqual(body["thinking"], ["type": "enabled", "budget_tokens": 1024])

        configuration.thinking = .disabled
        let off = AnthropicRequestBuilder.body(
            model: "claude-haiku-4-5", turns: [], system: nil, options: GenerationOptions(temperature: 0),
            configuration: configuration, sendsSamplingParameters: true, stream: false
        )
        XCTAssertEqual(off["temperature"], 0)
    }

    func testThinkingShapes() {
        XCTAssertEqual(AnthropicThinking.budget(tokens: 2048).json, ["type": "enabled", "budget_tokens": 2048])
        XCTAssertEqual(AnthropicThinking.disabled.json, ["type": "disabled"])
        XCTAssertEqual(AnthropicThinking.adaptive().json, ["type": "adaptive"])
    }

    func testToolChoiceAndFallbackShapes() {
        XCTAssertEqual(AnthropicToolChoice.tool(name: "x").json, ["type": "tool", "name": "x"])
        XCTAssertEqual(AnthropicToolChoice.any().json, ["type": "any"])
        XCTAssertEqual(AnthropicToolChoice.none.json, ["type": "none"])
        XCTAssertEqual(AnthropicRefusalFallbacks.models(["claude-opus-4-8"]).json, [["model": "claude-opus-4-8"]])
        XCTAssertEqual(AnthropicServerTool.codeExecution.json, ["type": "code_execution_20260521", "name": "code_execution"])
        XCTAssertEqual(
            AnthropicServerTool.webFetch(maxContentTokens: 500).json,
            ["type": "web_fetch_20260209", "name": "web_fetch", "max_content_tokens": 500]
        )
    }

    // MARK: - Tool turns

    func testAssistantEchoAndToolResultTurns() throws {
        let raw: JSONValue = ["id": "m", "model": "x", "content": [["type": "tool_use", "id": "toolu_1", "name": "t", "input": ["a": 1]]], "stop_reason": "tool_use"]
        let response = try AnthropicResponse.parse(raw)
        XCTAssertEqual(AnthropicRequestBuilder.assistantTurn(echoing: response), ["role": "assistant", "content": raw["content"]!])
        XCTAssertEqual(
            AnthropicRequestBuilder.toolResultTurn([("toolu_1", "ok", false), ("toolu_2", "boom", true)]),
            ["role": "user", "content": [
                ["type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"],
                ["type": "tool_result", "tool_use_id": "toolu_2", "content": "boom", "is_error": true],
            ]]
        )
    }

    func testCountTokensBody_hasNoGenerationSettings() {
        var configuration = AnthropicConfiguration()
        configuration.effort = .max
        configuration.tools = [AnthropicTool(name: "t", description: "d", inputSchema: ["type": "object"])]
        let body = AnthropicRequestBuilder.countTokensBody(model: "m", turns: [], system: "s", configuration: configuration)
        XCTAssertEqual(body["system"], "s")
        XCTAssertEqual(body["tools"]?.arrayValue?.count, 1)
        XCTAssertNil(body["max_tokens"])
        XCTAssertNil(body["output_config"])
    }

    // MARK: - Headers

    func testHeaders_betasIncludeFallbackHeader() {
        var configuration = AnthropicConfiguration()
        configuration.betas = ["compact-2026-01-12"]
        configuration.refusalFallbacks = .models(["claude-opus-4-8"])
        let headers = AnthropicRequestBuilder.headers(apiKey: "k", configuration: configuration)
        XCTAssertEqual(headers["x-api-key"], "k")
        XCTAssertEqual(headers["anthropic-version"], "2023-06-01")
        XCTAssertEqual(headers["anthropic-beta"], "compact-2026-01-12,server-side-fallback-2026-06-01")
        XCTAssertNil(AnthropicRequestBuilder.headers(apiKey: "k", configuration: AnthropicConfiguration())["anthropic-beta"])
    }

    // MARK: - Response parsing

    func testResponseParse_blocksUsageAndCitations() throws {
        let raw: JSONValue = [
            "id": "msg_1", "model": "claude-haiku-4-5", "stop_reason": "end_turn", "service_tier": "standard",
            "usage": ["input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 0],
            "content": [
                ["type": "thinking", "thinking": "hmm", "signature": "sig"],
                ["type": "text", "text": "Hi ", "citations": [["type": "page_location", "cited_text": "q", "document_index": 0, "document_title": "Doc", "start_page_number": 1, "end_page_number": 2]]],
                ["type": "text", "text": "there"],
                ["type": "server_tool_use", "id": "srv", "name": "web_search", "input": ["query": "x"]],
                ["type": "web_search_tool_result", "tool_use_id": "srv", "content": []],
            ],
        ]
        let response = try AnthropicResponse.parse(raw)
        XCTAssertEqual(response.id, "msg_1")
        XCTAssertEqual(response.text, "Hi there")
        XCTAssertEqual(response.stopReason, .endTurn)
        XCTAssertEqual(response.usage, AnthropicUsage(inputTokens: 10, outputTokens: 4, cacheCreationInputTokens: 0, cacheReadInputTokens: 3))
        XCTAssertEqual(response.serviceTier, "standard")
        XCTAssertEqual(response.citations.count, 1)
        XCTAssertEqual(response.citations.first?.documentTitle, "Doc")
        XCTAssertEqual(response.citations.first?.endPage, 2)
        XCTAssertEqual(response.content[0], .thinking("hmm"))
        if case .serverToolUse(let id, let name, _) = response.content[3] {
            XCTAssertEqual(id, "srv"); XCTAssertEqual(name, "web_search")
        } else { XCTFail("expected server_tool_use") }
        if case .other(let type, _) = response.content[4] { XCTAssertEqual(type, "web_search_tool_result") } else { XCTFail("expected other") }
        XCTAssertEqual(AnthropicStopReason(wire: "something_new"), .unknown)
    }

    func testExecute_runsHandlersInParallelAndKeepsOrder() async {
        let uses = [
            AnthropicResponse.ToolUse(id: "1", name: "slow", input: ["n": 1]),
            AnthropicResponse.ToolUse(id: "2", name: "fast", input: ["n": 2]),
            AnthropicResponse.ToolUse(id: "3", name: "bad", input: [:]),
        ]
        let handlers: [String: AnthropicTool.Handler] = [
            "slow": { input in try await Task.sleep(for: .milliseconds(50)); return "slow \(input["n"]?.intValue ?? 0)" },
            "fast": { input in "fast \(input["n"]?.intValue ?? 0)" },
            "bad": { _ in throw LLMError.requestFailed("nope") },
        ]
        let results = await AnthropicEngine.execute(uses, handlers: handlers)
        XCTAssertEqual(results.map(\.id), ["1", "2", "3"])
        XCTAssertEqual(results.map(\.output), ["slow 1", "fast 2", LLMError.requestFailed("nope").localizedDescription])
        XCTAssertEqual(results.map(\.isError), [false, false, true])
    }
}
