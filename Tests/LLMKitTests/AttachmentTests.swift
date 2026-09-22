//
//  AttachmentTests.swift
//  LLMKitTests
//
//  What each engine does with attachments it can and cannot carry, plus the
//  multipart encoding the Files API upload uses and the file record parse.
//

import XCTest
@testable import LLMKit

final class AttachmentTests: XCTestCase {

    private let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])

    func testMessage_allAttachmentsMergesImagesFirst() {
        let message = LLMMessage.user("x", images: [png], attachments: [.text("t")])
        XCTAssertEqual(message.allAttachments, [.image(png), .text("t")])
        XCTAssertTrue(LLMAttachment.text("t").isText)
        XCTAssertFalse(LLMAttachment.image(png).isText)
    }

    func testRemoteEngine_rendersImagesURLsAndInlinesText() {
        let url = URL(string: "https://example.com/i.png")!
        let body = RemoteEngine.requestBody(
            model: "m",
            messages: [.user("q", attachments: [.imageURL(url), .text("body", title: "Doc")])],
            options: GenerationOptions()
        )
        let content = (body["messages"] as! [[String: Any]])[0]["content"] as! [[String: Any]]
        XCTAssertEqual(content.count, 3)
        XCTAssertEqual((content[0]["image_url"] as? [String: Any])?["url"] as? String, url.absoluteString)
        XCTAssertEqual(content[1]["text"] as? String, "<document title=\"Doc\">\nbody\n</document>")
        XCTAssertEqual(content[2]["text"] as? String, "q")
    }

    func testRemoteEngine_rejectsPDFsAndFiles() {
        XCTAssertThrowsError(try RemoteEngine.checkAttachments(in: [.user("q", attachments: [.pdf(Data(), title: nil)])]))
        XCTAssertThrowsError(try RemoteEngine.checkAttachments(in: [.user("q", attachments: [.file(id: "f", kind: .image)])]))
        XCTAssertNoThrow(try RemoteEngine.checkAttachments(in: [.user("q", images: [png], attachments: [.text("t")])]))
    }

    func testMultipartForm_encoding() {
        let body = MultipartForm.body(
            parts: [.file("file", filename: "a.txt", contentType: "text/plain", data: Data("hi".utf8)),
                    .field("expires_in_seconds", "3600")],
            boundary: "B"
        )
        let text = String(decoding: body, as: UTF8.self)
        XCTAssertEqual(text, """
        --B\r
        Content-Disposition: form-data; name="file"; filename="a.txt"\r
        Content-Type: text/plain\r
        \r
        hi\r
        --B\r
        Content-Disposition: form-data; name="expires_in_seconds"\r
        \r
        3600\r
        --B--\r

        """)
        XCTAssertEqual(MultipartForm.contentType(boundary: "B"), "multipart/form-data; boundary=B")
    }

    func testFileRecordParse() throws {
        let json: JSONValue = ["id": "file_1", "type": "file", "filename": "d.pdf", "mime_type": "application/pdf",
                               "size_bytes": 1024, "created_at": "2025-01-01T00:00:00Z", "downloadable": false, "expires_at": nil]
        let file = try AnthropicFile.parse(json)
        XCTAssertEqual(file.id, "file_1")
        XCTAssertEqual(file.mimeType, "application/pdf")
        XCTAssertEqual(file.sizeBytes, 1024)
        XCTAssertNotNil(file.createdAt)
        XCTAssertFalse(file.downloadable)
        XCTAssertNil(file.expiresAt)
        XCTAssertThrowsError(try AnthropicFile.parse(["type": "file"]))
    }
}
