//
//  AnthropicFiles.swift
//  LLMKit
//
//  The Files API: upload once, reference by id in any number of messages
//  (`LLMAttachment.file`). Uploads are free; the content is billed as input
//  tokens when a message uses it. Files are workspace-wide, so an id is a
//  server-side reference — never accept one from an end user.
//

import Foundation

/// A file in the workspace's store.
public struct AnthropicFile: Sendable, Equatable, Identifiable {
    public let id: String
    public let filename: String
    public let mimeType: String
    public let sizeBytes: Int
    public let createdAt: Date?
    /// Only files the model produced (code execution, skills) can be
    /// downloaded; uploads cannot.
    public let downloadable: Bool
    public let expiresAt: Date?

    static func parse(_ json: JSONValue) throws -> AnthropicFile {
        guard let id = json["id"]?.stringValue else {
            throw LLMError.decodingFailed("no `id` in the file reply")
        }
        return AnthropicFile(
            id: id,
            filename: json["filename"]?.stringValue ?? "",
            mimeType: json["mime_type"]?.stringValue ?? "",
            sizeBytes: json["size_bytes"]?.intValue ?? 0,
            createdAt: ISO8601.date(from: json["created_at"]?.stringValue),
            downloadable: json["downloadable"]?.boolValue ?? false,
            expiresAt: ISO8601.date(from: json["expires_at"]?.stringValue)
        )
    }
}

public extension AnthropicEngine {

    /// Upload a file (PDF, plain text, or an image) and get its id.
    ///
    /// `mimeType` is detected by the server when omitted. `expiresIn` (3,600
    /// to 7,776,000 seconds) makes the file disappear on its own; without it
    /// the file lives until deleted. Up to 500 MB per file.
    func uploadFile(
        _ data: Data,
        filename: String,
        mimeType: String? = nil,
        expiresIn seconds: Int? = nil
    ) async throws -> AnthropicFile {
        let boundary = MultipartForm.makeBoundary()
        var parts: [MultipartForm.Part] = [.file("file", filename: filename, contentType: mimeType, data: data)]
        if let seconds { parts.append(.field("expires_in_seconds", String(seconds))) }
        let request = try HTTPTransport.request(
            baseURL.appendingPathComponent("files"),
            method: "POST",
            headers: headers,
            rawBody: MultipartForm.body(parts: parts, boundary: boundary),
            contentType: MultipartForm.contentType(boundary: boundary)
        )
        let reply = try await HTTPTransport.data(for: request, session: urlSession)
        return try AnthropicFile.parse(JSONValue.parse(reply))
    }

    /// One page of the workspace's files, newest first, with the cursor for
    /// the next page (pass it back as `page`).
    func files(limit: Int = 20, page: String? = nil) async throws -> (files: [AnthropicFile], nextPage: String?) {
        var components = URLComponents(url: baseURL.appendingPathComponent("files"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "limit", value: String(limit))]
            + (page.map { [URLQueryItem(name: "page", value: $0)] } ?? [])
        let request = try HTTPTransport.request(components.url!, method: "GET", headers: headers)
        let reply = try JSONValue.parse(await HTTPTransport.data(for: request, session: urlSession))
        let files = try (reply["data"]?.arrayValue ?? []).map(AnthropicFile.parse)
        return (files, reply["next_page"]?.stringValue)
    }

    /// A file's metadata.
    func fileMetadata(id: String) async throws -> AnthropicFile {
        let request = try HTTPTransport.request(baseURL.appendingPathComponent("files/\(id)"), method: "GET", headers: headers)
        return try AnthropicFile.parse(JSONValue.parse(await HTTPTransport.data(for: request, session: urlSession)))
    }

    /// Delete a file. Cannot be undone.
    func deleteFile(id: String) async throws {
        let request = try HTTPTransport.request(baseURL.appendingPathComponent("files/\(id)"), method: "DELETE", headers: headers)
        _ = try await HTTPTransport.data(for: request, session: urlSession)
    }

    /// The bytes of a file the model produced. Uploads answer 400.
    func downloadFile(id: String) async throws -> Data {
        let request = try HTTPTransport.request(baseURL.appendingPathComponent("files/\(id)/content"), method: "GET", headers: headers)
        return try await HTTPTransport.data(for: request, session: urlSession)
    }
}
