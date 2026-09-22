//
//  ImageData.swift
//  LLMKit
//
//  `LLMMessage.images` carries encoded bytes without saying which encoding.
//  Providers need a media type on every image block, and a PNG labelled
//  `image/jpeg` is rejected by Anthropic, so the type is read from the magic
//  bytes rather than assumed.
//

import Foundation

/// Sniff the media type of encoded image bytes.
enum ImageData {

    /// `image/png`, `image/gif` or `image/webp` from the file signature;
    /// anything else is reported as `image/jpeg`, the format callers were
    /// already assumed to send.
    static func mediaType(of data: Data) -> String {
        let head = [UInt8](data.prefix(12))
        if head.starts(with: [0x89, 0x50, 0x4E, 0x47]) { return "image/png" }
        if head.starts(with: [0x47, 0x49, 0x46, 0x38]) { return "image/gif" }
        if head.count >= 12, head.starts(with: [0x52, 0x49, 0x46, 0x46]),
           Array(head[8..<12]) == [0x57, 0x45, 0x42, 0x50] {
            return "image/webp"
        }
        return "image/jpeg"
    }
}
