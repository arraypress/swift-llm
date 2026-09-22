//
//  JSONValueTests.swift
//  LLMKitTests
//

import XCTest
@testable import LLMKit

final class JSONValueTests: XCTestCase {

    func testRoundTripThroughSerialization() throws {
        let value: JSONValue = ["name": "x", "n": 2, "ratio": 0.5, "ok": true, "none": nil, "list": [1, "two"]]
        let data = try value.data()
        XCTAssertEqual(try JSONValue.parse(data), value)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"n\":2"), "whole numbers serialise without a fraction: \(text)")
        XCTAssertTrue(text.contains("\"ratio\":0.5"))
    }

    func testBoolIsNotANumber() throws {
        let parsed = try JSONValue.parse(Data(#"{"a":true,"b":1,"c":0}"#.utf8))
        XCTAssertEqual(parsed["a"], .bool(true))
        XCTAssertEqual(parsed["b"], .number(1))
        XCTAssertEqual(parsed["c"]?.boolValue, nil)
        XCTAssertEqual(parsed["c"]?.intValue, 0)
    }

    func testAccessors() {
        let value: JSONValue = ["a": ["b": [10, 2.5]]]
        XCTAssertEqual(value["a"]?["b"]?[0]?.intValue, 10)
        XCTAssertNil(value["a"]?["b"]?[1]?.intValue)
        XCTAssertEqual(value["a"]?["b"]?[1]?.doubleValue, 2.5)
        XCTAssertNil(value["missing"])
        XCTAssertNil(value["a"]?[0])
        XCTAssertTrue(JSONValue.null.isNull)
    }

    func testCodable() throws {
        struct Wrapper: Codable, Equatable { let payload: JSONValue }
        let wrapper = Wrapper(payload: ["k": [1, false, nil, "s", ["z": 3.25]]])
        let data = try JSONEncoder().encode(wrapper)
        XCTAssertEqual(try JSONDecoder().decode(Wrapper.self, from: data), wrapper)
    }

    func testInitFromAnyRejectsNonJSON() {
        XCTAssertNil(JSONValue(any: Date()))
        XCTAssertEqual(JSONValue(any: ["a": NSNull()]), ["a": nil])
    }
}
