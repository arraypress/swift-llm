//
//  ModelListingTests.swift
//  LLMKitTests
//
//  Where `/models` is found for each endpoint, which field caps output, and how
//  a listing is parsed and ordered — offline.
//

import XCTest
@testable import LLMKit

final class ModelListingTests: XCTestCase {

    // MARK: - Endpoint dialects

    func testResolvedModelsURL_derivedFromChatURL() {
        XCTAssertEqual(RemoteEndpoint.openAI.resolvedModelsURL.absoluteString, "https://api.openai.com/v1/models")
        XCTAssertEqual(RemoteEndpoint.groq.resolvedModelsURL.absoluteString, "https://api.groq.com/openai/v1/models")
        XCTAssertEqual(RemoteEndpoint.openRouter.resolvedModelsURL.absoluteString, "https://openrouter.ai/api/v1/models")
        XCTAssertEqual(RemoteEndpoint.localServer().resolvedModelsURL.absoluteString, "http://localhost:11434/v1/models")
        XCTAssertEqual(
            RemoteEndpoint.cloudflareGateway(accountID: "acc", gateway: "gw").resolvedModelsURL.absoluteString,
            "https://gateway.ai.cloudflare.com/v1/acc/gw/compat/models"
        )
    }

    func testResolvedModelsURL_override() {
        let custom = RemoteEndpoint(
            url: URL(string: "https://x.test/chat/completions")!,
            modelsURL: URL(string: "https://x.test/catalogue")!
        )
        XCTAssertEqual(custom.resolvedModelsURL.absoluteString, "https://x.test/catalogue")
    }

    func testMaxTokensField_presets() {
        XCTAssertEqual(RemoteEndpoint.openAI.maxTokensField, "max_completion_tokens")
        XCTAssertEqual(RemoteEndpoint.groq.maxTokensField, "max_completion_tokens")
        XCTAssertEqual(RemoteEndpoint.openRouter.maxTokensField, "max_tokens")
        XCTAssertEqual(RemoteEndpoint.localServer().maxTokensField, "max_tokens")
    }

    func testRequestBody_usesEndpointMaxTokensField() {
        let body = RemoteEngine.requestBody(
            model: "gpt-5",
            messages: [.user("x")],
            options: GenerationOptions(maxTokens: 10),
            maxTokensField: RemoteEndpoint.openAI.maxTokensField
        )
        XCTAssertEqual(body["max_completion_tokens"] as? Int, 10)
        XCTAssertNil(body["max_tokens"])
    }

    // MARK: - Listing

    func testParseModels_openAIShape_newestFirst() throws {
        let json = #"{"object":"list","data":[{"id":"old","object":"model","created":1000},{"id":"new","object":"model","created":2000},{"id":"undated","object":"model"}]}"#
        let models = try RemoteEngine.parseModels(Data(json.utf8)).newestFirst()
        XCTAssertEqual(models.map(\.id), ["new", "old", "undated"])
        XCTAssertEqual(models[0].displayName, "new")
    }

    func testParseModels_badShapeThrows() {
        XCTAssertThrowsError(try RemoteEngine.parseModels(Data(#"{"models":[]}"#.utf8)))
    }

    func testModelInfo_displayNameFallsBackToID() {
        XCTAssertEqual(LLMModelInfo(id: "m").displayName, "m")
        XCTAssertEqual(LLMModelInfo(id: "m", displayName: "Model M").displayName, "Model M")
    }

    func testFacade_throwsForEnginesWithoutListing() async {
        struct Silent: LLMEngine {
            var isReady: Bool { get async { true } }
            func prepare() async throws {}
            func respond(to messages: [LLMMessage], options: GenerationOptions) async throws -> String { "" }
        }
        do {
            _ = try await LLM(Silent()).availableModels()
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual(error as? LLMError, .unavailable("this engine doesn't list models"))
        }
    }
}
