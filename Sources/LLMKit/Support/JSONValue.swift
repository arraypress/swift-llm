//
//  JSONValue.swift
//  LLMKit
//
//  A Sendable, Equatable JSON tree. `JSONSerialization` hands back `Any`, which
//  Swift 6 will not let cross an isolation boundary; tool inputs, schemas and
//  raw response blocks all need to, so they travel as this type instead. Whole
//  numbers round-trip as integers so a schema's `10` never becomes `10.0`.
//

import Foundation

/// A JSON value that is safe to pass between tasks and compare in tests.
public enum JSONValue: Sendable, Equatable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    // MARK: - Bridging

    /// Build from what `JSONSerialization` produces; `nil` for anything that
    /// isn't JSON (a `Date`, a custom class).
    public init?(any: Any) {
        switch any {
        case let value as JSONValue:
            self = value
        case is NSNull:
            self = .null
        case let string as String:
            self = .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                self = .bool(number.boolValue)
            } else {
                self = .number(number.doubleValue)
            }
        case let array as [Any]:
            var items: [JSONValue] = []
            items.reserveCapacity(array.count)
            for element in array {
                guard let value = JSONValue(any: element) else { return nil }
                items.append(value)
            }
            self = .array(items)
        case let dictionary as [String: Any]:
            var entries: [String: JSONValue] = [:]
            for (key, element) in dictionary {
                guard let value = JSONValue(any: element) else { return nil }
                entries[key] = value
            }
            self = .object(entries)
        default:
            return nil
        }
    }

    /// The `JSONSerialization` representation. Whole numbers come back as
    /// `Int` so they serialise as `10`, not `10.0`.
    public var anyValue: Any {
        switch self {
        case .string(let string): return string
        case .number(let double): return Self.isWhole(double) ? Int(double) : double
        case .bool(let bool): return bool
        case .null: return NSNull()
        case .array(let array): return array.map(\.anyValue)
        case .object(let object): return object.mapValues(\.anyValue)
        }
    }

    /// Parse JSON bytes.
    public static func parse(_ data: Data) throws -> JSONValue {
        let raw: Any
        do {
            raw = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        } catch {
            throw LLMError.decodingFailed("invalid JSON: \(error.localizedDescription)")
        }
        guard let value = JSONValue(any: raw) else {
            throw LLMError.decodingFailed("not a JSON value")
        }
        return value
    }

    /// Serialise to JSON bytes, keys sorted so output is stable.
    public func data() throws -> Data {
        try JSONSerialization.data(withJSONObject: anyValue, options: [.fragmentsAllowed, .sortedKeys])
    }

    private static func isWhole(_ double: Double) -> Bool {
        double == double.rounded() && abs(double) < 9_007_199_254_740_992
    }

    // MARK: - Accessors

    /// A member of an object, or `nil` for anything else.
    public subscript(key: String) -> JSONValue? {
        if case .object(let object) = self { return object[key] }
        return nil
    }

    /// An element of an array, or `nil` when out of range or not an array.
    public subscript(index: Int) -> JSONValue? {
        if case .array(let array) = self, array.indices.contains(index) { return array[index] }
        return nil
    }

    public var stringValue: String? {
        if case .string(let string) = self { return string }
        return nil
    }

    public var doubleValue: Double? {
        if case .number(let double) = self { return double }
        return nil
    }

    public var intValue: Int? {
        guard let double = doubleValue, Self.isWhole(double) else { return nil }
        return Int(double)
    }

    public var boolValue: Bool? {
        if case .bool(let bool) = self { return bool }
        return nil
    }

    public var arrayValue: [JSONValue]? {
        if case .array(let array) = self { return array }
        return nil
    }

    public var objectValue: [String: JSONValue]? {
        if case .object(let object) = self { return object }
        return nil
    }

    public var isNull: Bool { self == .null }
}

// MARK: - Literals

extension JSONValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
                     ExpressibleByBooleanLiteral, ExpressibleByNilLiteral, ExpressibleByArrayLiteral,
                     ExpressibleByDictionaryLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .number(Double(value)) }
    public init(floatLiteral value: Double) { self = .number(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(nilLiteral: ()) { self = .null }
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(Dictionary(elements, uniquingKeysWith: { _, last in last }))
    }
}

// MARK: - Codable

extension JSONValue: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let double = try? container.decode(Double.self) {
            self = .number(double)
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let array = try? container.decode([JSONValue].self) {
            self = .array(array)
        } else if let object = try? container.decode([String: JSONValue].self) {
            self = .object(object)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "not a JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let bool): try container.encode(bool)
        case .number(let double):
            if Self.isWhole(double) { try container.encode(Int(double)) } else { try container.encode(double) }
        case .string(let string): try container.encode(string)
        case .array(let array): try container.encode(array)
        case .object(let object): try container.encode(object)
        }
    }
}
