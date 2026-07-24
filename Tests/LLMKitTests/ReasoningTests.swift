//
//  ReasoningTests.swift
//  LLMKitTests
//

import XCTest
@testable import LLMKit

final class ReasoningTests: XCTestCase {

    func testSplitsThinkBlock() {
        let reply = "<think>5-7-5 has 17 syllables</think>A haiku is a three-line poem."
        let split = reply.splitReasoning()
        XCTAssertEqual(split.reasoning, "5-7-5 has 17 syllables")
        XCTAssertEqual(split.answer, "A haiku is a three-line poem.")
    }

    func testStripsThinkBlock() {
        let reply = "<think>reasoning here</think>\n\nThe answer."
        XCTAssertEqual(reply.strippingReasoning(), "The answer.")
    }

    func testPlainReplyUnchanged() {
        let reply = "Just a normal answer."
        let split = reply.splitReasoning()
        XCTAssertNil(split.reasoning)
        XCTAssertEqual(split.answer, "Just a normal answer.")
    }

    func testUnclosedThinkIsAllReasoning() {
        let reply = "<think>cut off mid-thought and never closed"
        let split = reply.splitReasoning()
        XCTAssertEqual(split.reasoning, "cut off mid-thought and never closed")
        XCTAssertEqual(split.answer, "")
    }

    func testAlternateTagsAndCaseInsensitive() {
        XCTAssertEqual("<Thinking>x</Thinking>done".strippingReasoning(), "done")
        XCTAssertEqual("<reasoning>y</reasoning> final".strippingReasoning(), "final")
    }

    func testAnswerBeforeAndAfterBlock() {
        let reply = "Prefix. <think>mid</think> Suffix."
        XCTAssertEqual(reply.strippingReasoning(), "Prefix. Suffix.")
    }

    // MARK: - gpt-oss harmony channels

    func testHarmonyFinalOnly() {
        let reply = "<|channel|>final<|message|>17 × 3 = 51."
        let split = reply.splitReasoning()
        XCTAssertNil(split.reasoning)
        XCTAssertEqual(split.answer, "17 × 3 = 51.")
    }

    func testHarmonyAnalysisAndFinal() {
        let reply = "<|channel|>analysis<|message|>17 minus 9 is 8 that die.<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>9 sheep are left."
        let split = reply.splitReasoning()
        XCTAssertEqual(split.reasoning, "17 minus 9 is 8 that die.")
        XCTAssertEqual(split.answer, "9 sheep are left.")
    }
}
