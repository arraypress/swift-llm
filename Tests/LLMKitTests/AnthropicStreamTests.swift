//
//  AnthropicStreamTests.swift
//  LLMKitTests
//
//  A recorded event sequence, replayed through the accumulator, must yield the
//  live deltas in order and end in the same response a non-streamed call
//  would have returned — tool input reassembled from its fragments included.
//

import XCTest
@testable import LLMKit

final class AnthropicStreamTests: XCTestCase {

    private let recorded: [String] = [
        "event: message_start",
        #"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1}}}"#,
        "",
        #"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        #"data: {"type":"ping"}"#,
        #"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me "}}"#,
        #"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"check."}}"#,
        #"data: {"type":"content_block_stop","index":0}"#,
        #"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#,
        #"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\": \"Os"}}"#,
        #"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"lo\"}"}}"#,
        #"data: {"type":"content_block_stop","index":1}"#,
        #"data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":21}}"#,
        #"data: {"type":"message_stop"}"#,
    ]

    func testAccumulator_replaysToolCallStream() throws {
        var accumulator = AnthropicStreamAccumulator()
        var events: [AnthropicStreamEvent] = []
        for line in recorded { events += try accumulator.consume(line: line) }

        guard case .completed(let response) = events.last else { return XCTFail("expected a completed event") }
        XCTAssertEqual(Array(events.dropLast()), [
            .messageStart(id: "msg_1", model: "claude-haiku-4-5"),
            .textDelta("Let me "),
            .textDelta("check."),
            .blockStop(index: 0),
            .toolUseStart(index: 1, id: "toolu_1", name: "get_weather"),
            .toolInputDelta(index: 1, partialJSON: "{\"city\": \"Os"),
            .toolInputDelta(index: 1, partialJSON: "lo\"}"),
            .blockStop(index: 1),
        ])
        XCTAssertEqual(response.id, "msg_1")
        XCTAssertEqual(response.text, "Let me check.")
        XCTAssertEqual(response.stopReason, .toolUse)
        XCTAssertEqual(response.toolUses, [AnthropicResponse.ToolUse(id: "toolu_1", name: "get_weather", input: ["city": "Oslo"])])
        XCTAssertEqual(response.usage.inputTokens, 25)
        XCTAssertEqual(response.usage.outputTokens, 21)
        // The echoed assistant turn carries the reassembled input, as a tool loop needs.
        XCTAssertEqual(AnthropicRequestBuilder.assistantTurn(echoing: response)["content"]?[1]?["input"], ["city": "Oslo"])
    }

    func testAccumulator_thinkingAndSignature() throws {
        var accumulator = AnthropicStreamAccumulator()
        let lines = [
            #"data: {"type":"message_start","message":{"id":"m","model":"claude-opus-5","usage":{"input_tokens":1}}}"#,
            #"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            #"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step 1"}}"#,
            #"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}"#,
            #"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            #"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"42"}}"#,
            #"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#,
            #"data: {"type":"message_stop"}"#,
        ]
        var events: [AnthropicStreamEvent] = []
        for line in lines { events += try accumulator.consume(line: line) }
        XCTAssertTrue(events.contains(.thinkingDelta("step 1")))
        guard case .completed(let response) = events.last else { return XCTFail("expected a completed event") }
        XCTAssertEqual(response.content[0], .thinking("step 1"))
        XCTAssertEqual(response.raw["content"]?[0]?["signature"], "abc", "signatures must survive for replay")
        XCTAssertEqual(response.text, "42")
    }

    func testAccumulator_errorEventThrows() {
        var accumulator = AnthropicStreamAccumulator()
        XCTAssertThrowsError(try accumulator.consume(line: #"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#)) { error in
            XCTAssertEqual(error as? LLMError, .requestFailed("Overloaded"))
        }
    }

    func testAccumulator_refusalSurvivesToResponse() throws {
        var accumulator = AnthropicStreamAccumulator()
        let lines = [
            #"data: {"type":"message_start","message":{"id":"m","model":"x","usage":{}}}"#,
            #"data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber"}}}"#,
            #"data: {"type":"message_stop"}"#,
        ]
        var events: [AnthropicStreamEvent] = []
        for line in lines { events += try accumulator.consume(line: line) }
        guard case .completed(let response) = events.last else { return XCTFail("expected a completed event") }
        XCTAssertEqual(response.stopReason, .refusal)
        XCTAssertEqual(response.stopDetails?.category, "cyber")
        XCTAssertEqual(response.refusalDetail, "declined for category cyber")
    }
}
