//
//  JSONExtractor.swift
//  LLMKit
//
//  Pulls a JSON object out of a model's text response and decodes it. Models
//  wrap JSON in prose or ```json fences, so we find the first balanced { … }
//  rather than trusting the whole string to be valid JSON.
//

import Foundation

/// Extracts and decodes structured JSON from free-form model output.
public enum JSONExtractor {

    /// Return the first balanced top-level `{ … }` object found in `text`
    /// (ignoring braces inside strings), or `nil` if there isn't one.
    public static func firstJSONObject(in text: String) -> String? {
        let scalars = Array(text)
        guard let start = scalars.firstIndex(of: "{") else { return nil }

        var depth = 0
        var inString = false
        var escaped = false
        var index = start

        while index < scalars.count {
            let character = scalars[index]
            if inString {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == "\"" {
                    inString = false
                }
            } else {
                switch character {
                case "\"": inString = true
                case "{": depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        return String(scalars[start...index])
                    }
                default: break
                }
            }
            index += 1
        }
        return nil
    }

    /// Decode `type` from the first JSON object found in `text`.
    public static func decode<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
        guard let json = firstJSONObject(in: text), let data = json.data(using: .utf8) else {
            throw LLMError.decodingFailed("no JSON object found in response")
        }
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw LLMError.decodingFailed(String(describing: error))
        }
    }
}
