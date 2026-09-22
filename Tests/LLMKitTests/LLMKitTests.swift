//
//  LLMKitTests.swift
//  LLMKitTests
//

import XCTest
@testable import LLMKit

final class LLMKitTests: XCTestCase {

    // MARK: - JSONExtractor

    func testFirstJSONObject_plain() {
        XCTAssertEqual(JSONExtractor.firstJSONObject(in: #"{"a":1}"#), #"{"a":1}"#)
    }

    func testFirstJSONObject_insideProseAndFences() {
        let text = "Sure! Here you go:\n```json\n{\"name\":\"x\",\"n\":2}\n```\nHope that helps."
        XCTAssertEqual(JSONExtractor.firstJSONObject(in: text), #"{"name":"x","n":2}"#)
    }

    func testFirstJSONObject_ignoresBracesInStrings() {
        let text = #"{"note":"a } brace","ok":true}"#
        XCTAssertEqual(JSONExtractor.firstJSONObject(in: text), text)
    }

    func testFirstJSONObject_nested() {
        let text = #"prefix {"a":{"b":1}} suffix"#
        XCTAssertEqual(JSONExtractor.firstJSONObject(in: text), #"{"a":{"b":1}}"#)
    }

    func testFirstJSONObject_none() {
        XCTAssertNil(JSONExtractor.firstJSONObject(in: "no json here"))
    }

    func testDecode_structured() throws {
        struct Expense: Decodable, Equatable { let title: String; let amount: String }
        let text = "Here: {\"title\":\"Coffee\",\"amount\":\"£3.50\"} ✅"
        let expense = try JSONExtractor.decode(Expense.self, from: text)
        XCTAssertEqual(expense, Expense(title: "Coffee", amount: "£3.50"))
    }

    func testDecode_noJSON_throws() {
        struct X: Decodable { let a: Int }
        XCTAssertThrowsError(try JSONExtractor.decode(X.self, from: "nope"))
    }

    // MARK: - RemoteEngine request body

    func testRequestBody_textOnly() {
        let body = RemoteEngine.requestBody(
            model: "gpt-4o",
            messages: [.system("be brief"), .user("hi")],
            options: GenerationOptions(temperature: 0.3, maxTokens: 100),
            endpoint: .openRouter
        )
        XCTAssertEqual(body["model"] as? String, "gpt-4o")
        XCTAssertEqual(body["temperature"] as? Double, 0.3)
        XCTAssertEqual(body["max_tokens"] as? Int, 100)
        let messages = body["messages"] as? [[String: Any]]
        XCTAssertEqual(messages?.count, 2)
        XCTAssertEqual(messages?[0]["role"] as? String, "system")
        XCTAssertEqual(messages?[1]["content"] as? String, "hi")
    }

    func testRequestBody_withImage_usesImageURLBlock() {
        let image = Data([0xFF, 0xD8, 0xFF])  // jpeg magic
        let body = RemoteEngine.requestBody(
            model: "gpt-4o",
            messages: [.user("what is this?", images: [image])],
            options: GenerationOptions(),
            endpoint: .openRouter
        )
        let messages = body["messages"] as? [[String: Any]]
        let content = messages?.first?["content"] as? [[String: Any]]
        XCTAssertEqual(content?.first?["type"] as? String, "image_url")
        let imageURL = (content?.first?["image_url"] as? [String: Any])?["url"] as? String
        XCTAssertEqual(imageURL, "data:image/jpeg;base64,\(image.base64EncodedString())")
        XCTAssertEqual(content?.last?["type"] as? String, "text")
    }

    // MARK: - RemoteEngine response parsing

    func testParseContent_openAIShape() throws {
        let json = #"{"choices":[{"message":{"role":"assistant","content":"Hello there"}}]}"#
        XCTAssertEqual(try RemoteEngine.parseContent(Data(json.utf8)), "Hello there")
    }

    func testParseContent_empty_throws() {
        XCTAssertThrowsError(try RemoteEngine.parseContent(Data(#"{"choices":[]}"#.utf8)))
    }

    // MARK: - Endpoints

    func testEndpoint_openAI() {
        XCTAssertEqual(RemoteEndpoint.openAI.url.absoluteString, "https://api.openai.com/v1/chat/completions")
        XCTAssertEqual(RemoteEndpoint.openAI.authScheme, "Bearer")
    }

    func testEndpoint_cloudflareGateway_headerAndURL() {
        let endpoint = RemoteEndpoint.cloudflareGateway(accountID: "acc", gateway: "gw", provider: "openai")
        XCTAssertEqual(endpoint.authHeaderField, "cf-aig-authorization")
        XCTAssertTrue(endpoint.url.absoluteString.hasPrefix("https://gateway.ai.cloudflare.com/v1/acc/gw/openai/"))
    }

    func testEndpoint_localServer_noAuthScheme() {
        XCTAssertEqual(RemoteEndpoint.localServer().authScheme, "")
    }

    // MARK: - Options

    func testOptions_temperatureClampsAndDeterministic() {
        XCTAssertEqual(GenerationOptions(temperature: -5).temperature, 0)
        XCTAssertEqual(GenerationOptions.deterministic.temperature, 0)
    }

    // MARK: - RemoteEngine readiness

    func testRemoteEngine_missingKey_notReady() async throws {
        let engine = RemoteEngine(model: "gpt-4o", endpoint: .openAI, apiKey: "")
        let ready = await engine.isReady
        XCTAssertFalse(ready)
        do {
            try await engine.prepare()
            XCTFail("expected missingAPIKey")
        } catch let error as LLMError {
            XCTAssertEqual(error, .missingAPIKey)
        }
    }

    func testRemoteEngine_localServer_readyWithoutKey() async {
        let engine = RemoteEngine(model: "qwen3", endpoint: .localServer(), apiKey: "")
        let ready = await engine.isReady
        XCTAssertTrue(ready)
    }

    // MARK: - FoundationEngine (guards only — machine may lack Apple Intelligence)

    func testFoundationEngine_isReady_doesNotCrash() async {
        _ = await FoundationEngine().isReady   // value depends on system state
    }

    func testFoundationEngine_unavailableReasons_haveDescriptions() {
        XCTAssertFalse(FoundationEngine.describe(.deviceNotEligible).isEmpty)
        XCTAssertFalse(FoundationEngine.describe(.appleIntelligenceNotEnabled).isEmpty)
        XCTAssertFalse(FoundationEngine.describe(.modelNotReady).isEmpty)
    }
}
