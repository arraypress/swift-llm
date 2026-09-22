//
//  AnthropicEngineTests.swift
//  LLMKitTests
//
//  The Messages API shaping and parsing, offline: what goes on the wire, what
//  comes back, and the two ways a 200 can still carry no answer.
//

import XCTest
@testable import LLMKit

final class AnthropicEngineTests: XCTestCase {

    private let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0])

    // MARK: - Request body

    func testRequestBody_systemBecomesTopLevelField() {
        let body = AnthropicEngine.requestBody(
            model: "claude-haiku-4-5",
            messages: [.system("be brief"), .system("in English"), .user("hi")],
            options: GenerationOptions(),
            sendsSamplingParameters: true
        )
        XCTAssertEqual(body["model"] as? String, "claude-haiku-4-5")
        XCTAssertEqual(body["system"] as? String, "be brief\nin English")
        XCTAssertEqual(body["max_tokens"] as? Int, AnthropicEngine.defaultMaxTokens)
        XCTAssertNil(body["stream"])
        let messages = body["messages"] as? [[String: Any]]
        XCTAssertEqual(messages?.count, 1)
        XCTAssertEqual(messages?.first?["role"] as? String, "user")
        XCTAssertEqual(messages?.first?["content"] as? String, "hi")
    }

    func testRequestBody_mergesConsecutiveSameRoleMessages() {
        let body = AnthropicEngine.requestBody(
            model: "m",
            messages: [.user("a"), .user("b"), .assistant("c"), .user("d")],
            options: GenerationOptions(),
            sendsSamplingParameters: false
        )
        let messages = body["messages"] as! [[String: Any]]
        XCTAssertEqual(messages.map { $0["role"] as? String }, ["user", "assistant", "user"])
        let merged = messages[0]["content"] as? [[String: Any]]
        XCTAssertEqual(merged?.compactMap { $0["text"] as? String }, ["a", "b"])
        XCTAssertEqual(messages[1]["content"] as? String, "c")
        XCTAssertEqual(messages[2]["content"] as? String, "d")
    }

    func testRequestBody_imagesBecomeTypedBlocksBeforeText() {
        let body = AnthropicEngine.requestBody(
            model: "m",
            messages: [.user("what is this?", images: [png])],
            options: GenerationOptions(),
            sendsSamplingParameters: false
        )
        let content = (body["messages"] as! [[String: Any]])[0]["content"] as! [[String: Any]]
        XCTAssertEqual(content.map { $0["type"] as? String }, ["image", "text"])
        let source = content[0]["source"] as? [String: Any]
        XCTAssertEqual(source?["type"] as? String, "base64")
        XCTAssertEqual(source?["media_type"] as? String, "image/png")
        XCTAssertEqual(source?["data"] as? String, png.base64EncodedString())
        XCTAssertEqual(content[1]["text"] as? String, "what is this?")
    }

    func testRequestBody_samplingGateAndStreamFlag() {
        let options = GenerationOptions(temperature: 0.2, maxTokens: 50, topP: 0.9)
        let with = AnthropicEngine.requestBody(
            model: "m", messages: [.user("x")], options: options, sendsSamplingParameters: true, stream: true
        )
        XCTAssertEqual(with["temperature"] as? Double, 0.2)
        XCTAssertEqual(with["top_p"] as? Double, 0.9)
        XCTAssertEqual(with["max_tokens"] as? Int, 50)
        XCTAssertEqual(with["stream"] as? Bool, true)

        let without = AnthropicEngine.requestBody(
            model: "m", messages: [.user("x")], options: options, sendsSamplingParameters: false
        )
        XCTAssertNil(without["temperature"])
        XCTAssertNil(without["top_p"])
        XCTAssertEqual(without["max_tokens"] as? Int, 50)
    }

    func testAcceptsSamplingParameters_byGeneration() {
        XCTAssertTrue(AnthropicEngine.acceptsSamplingParameters(model: "claude-haiku-4-5"))
        XCTAssertTrue(AnthropicEngine.acceptsSamplingParameters(model: "claude-sonnet-4-6"))
        XCTAssertTrue(AnthropicEngine.acceptsSamplingParameters(model: "claude-opus-4-5-20251101"))
        XCTAssertFalse(AnthropicEngine.acceptsSamplingParameters(model: "claude-opus-5"))
        XCTAssertFalse(AnthropicEngine.acceptsSamplingParameters(model: "claude-sonnet-5"))
        XCTAssertFalse(AnthropicEngine.acceptsSamplingParameters(model: "claude-opus-4-7"))
        XCTAssertFalse(AnthropicEngine.acceptsSamplingParameters(model: "claude-opus-4-8"))
        XCTAssertFalse(AnthropicEngine.acceptsSamplingParameters(model: "claude-fable-5-1"))
    }

    func testEngineDefaultsSamplingFromModel_andCanBeForced() {
        XCTAssertTrue(AnthropicEngine(model: "claude-haiku-4-5", apiKey: "k").sendsSamplingParameters)
        XCTAssertFalse(AnthropicEngine(model: "claude-opus-5", apiKey: "k").sendsSamplingParameters)
        XCTAssertTrue(AnthropicEngine(model: "claude-opus-5", apiKey: "k", sendsSamplingParameters: true).sendsSamplingParameters)
    }

    // MARK: - Response parsing

    func testParseContent_joinsTextBlocksAndSkipsOthers() throws {
        let json = #"{"content":[{"type":"text","text":"Hello, "},{"type":"tool_use","id":"x"},{"type":"text","text":"world"}],"stop_reason":"end_turn"}"#
        XCTAssertEqual(try AnthropicEngine.parseContent(Data(json.utf8)), "Hello, world")
    }

    func testParseContent_refusalThrows() {
        let json = #"{"content":[],"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber","explanation":"nope"}}"#
        XCTAssertThrowsError(try AnthropicEngine.parseContent(Data(json.utf8))) { error in
            XCTAssertEqual(error as? LLMError, .refused("nope"))
        }
    }

    func testParseContent_emptyThrows() {
        XCTAssertThrowsError(try AnthropicEngine.parseContent(Data(#"{"content":[]}"#.utf8))) { error in
            XCTAssertEqual(error as? LLMError, .emptyResponse)
        }
    }

    // MARK: - Streaming

    func testStreamDelta_yieldsTextDeltasOnly() throws {
        XCTAssertNil(try AnthropicEngine.streamDelta("event: message_start"))
        XCTAssertNil(try AnthropicEngine.streamDelta(#"data: {"type":"message_start","message":{}}"#))
        XCTAssertNil(try AnthropicEngine.streamDelta(#"data: {"type":"ping"}"#))
        XCTAssertNil(try AnthropicEngine.streamDelta(#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#))
        XCTAssertEqual(
            try AnthropicEngine.streamDelta(#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#),
            "Hel"
        )
        XCTAssertNil(try AnthropicEngine.streamDelta(#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{"}}"#))
        XCTAssertNil(try AnthropicEngine.streamDelta(#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#))
        XCTAssertNil(try AnthropicEngine.streamDelta(#"data: {"type":"message_stop"}"#))
    }

    func testStreamDelta_errorAndRefusalThrow() {
        XCTAssertThrowsError(try AnthropicEngine.streamDelta(#"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#)) { error in
            XCTAssertEqual(error as? LLMError, .requestFailed("Overloaded"))
        }
        XCTAssertThrowsError(try AnthropicEngine.streamDelta(#"data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"category":"bio"}}}"#)) { error in
            XCTAssertEqual(error as? LLMError, .refused("declined for category bio"))
        }
    }

    // MARK: - Model listing

    func testParseModels_pageAndCursor() throws {
        let json = #"{"data":[{"type":"model","id":"claude-opus-5","display_name":"Claude Opus 5","created_at":"2026-04-01T00:00:00Z"},{"type":"model","id":"claude-haiku-4-5","display_name":"Claude Haiku 4.5","created_at":"2025-10-01T00:00:00Z"}],"has_more":true,"first_id":"claude-opus-5","last_id":"claude-haiku-4-5"}"#
        let page = try AnthropicEngine.parseModels(Data(json.utf8))
        XCTAssertEqual(page.models.map(\.id), ["claude-opus-5", "claude-haiku-4-5"])
        XCTAssertEqual(page.models[0].displayName, "Claude Opus 5")
        XCTAssertNotNil(page.models[0].created)
        XCTAssertEqual(page.nextCursor, "claude-haiku-4-5")

        let last = try AnthropicEngine.parseModels(Data(#"{"data":[],"has_more":false,"last_id":null}"#.utf8))
        XCTAssertTrue(last.models.isEmpty)
        XCTAssertNil(last.nextCursor)
    }
}
