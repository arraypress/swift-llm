//
//  StreamingTests.swift
//  LLMKitTests
//
//  The wire-level pieces every streaming engine shares, offline: SSE framing,
//  the OpenAI delta shape, snapshot-to-delta for the Apple engine, and the
//  error text lifted out of a failed reply.
//

import XCTest
@testable import LLMKit

final class StreamingTests: XCTestCase {

    // MARK: - SSE framing

    func testDataPayload_framing() {
        XCTAssertEqual(ServerSentEvents.dataPayload(of: "data: {\"a\":1}"), "{\"a\":1}")
        XCTAssertEqual(ServerSentEvents.dataPayload(of: "data:{\"a\":1}"), "{\"a\":1}")
        XCTAssertEqual(ServerSentEvents.dataPayload(of: "data:  two spaces"), " two spaces")
        XCTAssertEqual(ServerSentEvents.dataPayload(of: "data: [DONE]"), ServerSentEvents.done)
        XCTAssertNil(ServerSentEvents.dataPayload(of: "event: content_block_delta"))
        XCTAssertNil(ServerSentEvents.dataPayload(of: ": keep-alive"))
        XCTAssertNil(ServerSentEvents.dataPayload(of: ""))
    }

    func testJSONObject_ignoresSentinelAndNonObjects() {
        XCTAssertNil(ServerSentEvents.jsonObject(of: "data: [DONE]"))
        XCTAssertNil(ServerSentEvents.jsonObject(of: "data: [1,2]"))
        XCTAssertNil(ServerSentEvents.jsonObject(of: "id: 7"))
        XCTAssertEqual(ServerSentEvents.jsonObject(of: #"data: {"k":"v"}"#)?["k"] as? String, "v")
    }

    // MARK: - OpenAI-compatible deltas

    func testRemoteStreamDelta() throws {
        XCTAssertEqual(try RemoteEngine.streamDelta(#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#), "Hi")
        XCTAssertNil(try RemoteEngine.streamDelta(#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#))
        XCTAssertNil(try RemoteEngine.streamDelta(#"data: {"choices":[{"delta":{"content":null},"finish_reason":"stop"}]}"#))
        XCTAssertNil(try RemoteEngine.streamDelta("data: [DONE]"))
        XCTAssertNil(try RemoteEngine.streamDelta(#"data: {"choices":[]}"#))
        XCTAssertNil(try RemoteEngine.streamDelta("event: ping"))
    }

    func testRemoteStreamDelta_inBandErrorThrows() {
        XCTAssertThrowsError(try RemoteEngine.streamDelta(#"data: {"error":{"message":"Rate limit","type":"rate_limit"}}"#)) { error in
            XCTAssertEqual(error as? LLMError, .requestFailed("Rate limit"))
        }
    }

    func testRequestBody_streamFlagOnlyWhenAsked() {
        let plain = RemoteEngine.requestBody(model: "m", messages: [.user("x")], options: GenerationOptions())
        XCTAssertNil(plain["stream"])
        let streaming = RemoteEngine.requestBody(model: "m", messages: [.user("x")], options: GenerationOptions(), stream: true)
        XCTAssertEqual(streaming["stream"] as? Bool, true)
    }

    // MARK: - Apple snapshots

    func testFoundationDelta() {
        XCTAssertEqual(FoundationEngine.delta(from: "", to: "Hel"), "Hel")
        XCTAssertEqual(FoundationEngine.delta(from: "Hel", to: "Hello"), "lo")
        XCTAssertEqual(FoundationEngine.delta(from: "Hello", to: "Hello"), "")
        XCTAssertEqual(FoundationEngine.delta(from: "Hello", to: "Goodbye"), "Goodbye")
    }

    // MARK: - Error bodies and image types

    func testErrorDetail_prefersProviderMessage() {
        XCTAssertEqual(HTTPTransport.errorDetail(in: Data(#"{"error":{"message":"bad key","type":"auth"}}"#.utf8)), "bad key")
        XCTAssertEqual(HTTPTransport.errorDetail(in: Data("  gateway timeout \n".utf8)), "gateway timeout")
        XCTAssertEqual(HTTPTransport.errorDetail(in: Data()), "no error detail")
    }

    func testImageMediaType() {
        XCTAssertEqual(ImageData.mediaType(of: Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])), "image/png")
        XCTAssertEqual(ImageData.mediaType(of: Data([0xFF, 0xD8, 0xFF, 0xE0])), "image/jpeg")
        XCTAssertEqual(ImageData.mediaType(of: Data("GIF89a".utf8)), "image/gif")
        let webp = Data([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50])
        XCTAssertEqual(ImageData.mediaType(of: webp), "image/webp")
        XCTAssertEqual(ImageData.mediaType(of: Data()), "image/jpeg")
    }
}
