//
//  RemoteEndpointTests.swift
//  LLMKitTests
//
//  OpenAI's reasoning models refuse `temperature`; the request must leave it out.
//

import XCTest
@testable import LLMKit

final class RemoteEndpointTests: XCTestCase {

    func testOpenAIReasoningModelsHaveFixedSampling() {
        let endpoint = RemoteEndpoint.openAI
        XCTAssertFalse(endpoint.acceptsSamplingParameters(model: "gpt-5"))
        XCTAssertFalse(endpoint.acceptsSamplingParameters(model: "gpt-5.6-terra"))
        XCTAssertFalse(endpoint.acceptsSamplingParameters(model: "gpt-6-astra"))
        XCTAssertFalse(endpoint.acceptsSamplingParameters(model: "o3-mini"))
        XCTAssertTrue(endpoint.acceptsSamplingParameters(model: "gpt-4o"))
        XCTAssertTrue(endpoint.acceptsSamplingParameters(model: "gpt-4.1-mini"))
    }

    func testOtherEndpointsKeepSampling() {
        XCTAssertTrue(RemoteEndpoint.localServer().acceptsSamplingParameters(model: "gpt-5-lookalike"))
        XCTAssertTrue(RemoteEndpoint.openRouter.acceptsSamplingParameters(model: "openai/gpt-5"))
    }

    func testRequestBodyDropsTemperatureForFixedSamplingModels() {
        let engine = RemoteEngine(model: "gpt-5.6-terra", endpoint: .openAI, apiKey: "k")
        let body = RemoteEngine.requestBody(
            model: engine.model, messages: [.user("hi")],
            options: GenerationOptions(temperature: 0.2, maxTokens: 10),
            endpoint: engine.endpoint
        )
        XCTAssertNil(body["temperature"])
        XCTAssertNil(body["top_p"])
        XCTAssertEqual(body["max_completion_tokens"] as? Int, 10)

        let legacy = RemoteEngine.requestBody(
            model: "gpt-4o", messages: [.user("hi")],
            options: GenerationOptions(temperature: 0.2, maxTokens: 10),
            endpoint: .openAI
        )
        XCTAssertEqual(legacy["temperature"] as? Double, 0.2)
    }
}
