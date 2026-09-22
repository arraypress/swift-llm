//
//  MultipartForm.swift
//  LLMKit
//
//  The `multipart/form-data` encoding a file upload needs. Foundation has no
//  builder for it, and the format is small enough that hand-rolling it beats
//  a dependency: a boundary, one part per field, CRLF line endings.
//

import Foundation

/// A `multipart/form-data` body.
enum MultipartForm {

    /// One field of the form: a plain value or a file with a name and type.
    struct Part: Sendable {
        let name: String
        let filename: String?
        let contentType: String?
        let data: Data

        /// A plain text field.
        static func field(_ name: String, _ value: String) -> Part {
            Part(name: name, filename: nil, contentType: nil, data: Data(value.utf8))
        }

        /// A file field.
        static func file(_ name: String, filename: String, contentType: String?, data: Data) -> Part {
            Part(name: name, filename: filename, contentType: contentType, data: data)
        }
    }

    /// A boundary no sane payload contains.
    static func makeBoundary() -> String {
        "LLMKit-" + UUID().uuidString
    }

    /// The `Content-Type` header value for `boundary`.
    static func contentType(boundary: String) -> String {
        "multipart/form-data; boundary=\(boundary)"
    }

    /// Encode `parts` between `boundary` markers.
    static func body(parts: [Part], boundary: String) -> Data {
        var body = Data()
        for part in parts {
            body.append("--\(boundary)\r\n")
            var disposition = "Content-Disposition: form-data; name=\"\(part.name)\""
            if let filename = part.filename {
                disposition += "; filename=\"\(filename)\""
            }
            body.append("\(disposition)\r\n")
            if let contentType = part.contentType {
                body.append("Content-Type: \(contentType)\r\n")
            }
            body.append("\r\n")
            body.append(part.data)
            body.append("\r\n")
        }
        body.append("--\(boundary)--\r\n")
        return body
    }
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}
