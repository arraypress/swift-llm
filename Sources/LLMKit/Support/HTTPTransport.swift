//
//  HTTPTransport.swift
//  LLMKit
//
//  The one place the cloud engines touch URLSession. Both providers put the
//  useful error text in the response body, not the status line, so a non-2xx
//  reply is read to the end and surfaced as `LLMError.requestFailed` with the
//  provider's own message. Streaming hands back the line sequence only after
//  the status has been checked, so a 401 never arrives as a "token".
//

import Foundation

/// URLSession plumbing shared by `RemoteEngine` and `AnthropicEngine`.
enum HTTPTransport {

    /// Seconds the connection may sit idle between packets. This is
    /// `URLRequest.timeoutInterval` — idle time, not total time — so a long
    /// streamed generation is fine as long as tokens keep arriving.
    static let idleTimeout: TimeInterval = 60

    /// Build a JSON request. `body` is serialised only when present, so GETs
    /// stay bodiless.
    static func request(
        _ url: URL,
        method: String,
        headers: [String: String],
        body: [String: Any]? = nil
    ) throws -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = idleTimeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for (field, value) in headers {
            request.setValue(value, forHTTPHeaderField: field)
        }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        return request
    }

    /// Build a request with a pre-encoded body (a multipart upload).
    static func request(
        _ url: URL,
        method: String,
        headers: [String: String],
        rawBody: Data,
        contentType: String
    ) throws -> URLRequest {
        var request = try request(url, method: method, headers: headers)
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        request.httpBody = rawBody
        return request
    }

    /// Perform `request` and return the body of a 2xx reply.
    static func data(for request: URLRequest, session: URLSession) async throws -> Data {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw LLMError.requestFailed(error.localizedDescription)
        }
        try check(response, body: data)
        return data
    }

    /// Open `request` and return its line stream once the server has answered
    /// 2xx. An error reply is small, so it is read in full for its message.
    static func lines(for request: URLRequest, session: URLSession) async throws -> AsyncLineSequence<URLSession.AsyncBytes> {
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: request)
        } catch {
            throw LLMError.requestFailed(error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            var body = Data()
            for try await byte in bytes { body.append(byte) }
            try check(response, body: body)
        }
        return bytes.lines
    }

    /// Throw `LLMError.requestFailed` for a non-2xx status, quoting the
    /// provider's `error.message` when the body has one.
    static func check(_ response: URLResponse, body: Data) throws {
        guard let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) else { return }
        throw LLMError.requestFailed("HTTP \(http.statusCode): \(errorDetail(in: body))")
    }

    /// The provider's error message from a JSON error body (`error.message`,
    /// the shape both OpenAI and Anthropic use), else the first 300 characters.
    static func errorDetail(in body: Data) -> String {
        if let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
           let error = object["error"] as? [String: Any],
           let message = error["message"] as? String, !message.isEmpty {
            return message
        }
        let text = String(decoding: body, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? "no error detail" : String(text.prefix(300))
    }
}
