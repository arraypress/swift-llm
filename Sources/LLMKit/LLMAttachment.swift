//
//  LLMAttachment.swift
//  LLMKit
//
//  What a message can carry besides text. Each engine renders these into its
//  own content blocks, and throws `LLMError.unavailable` for the kinds its
//  provider has no block for, rather than silently dropping them.
//

import Foundation

/// A non-text part of a message.
public enum LLMAttachment: Sendable, Equatable {

    /// An encoded image (JPEG, PNG, GIF or WebP), sent inline as base64.
    /// Every engine with vision accepts this.
    case image(Data)

    /// An image the provider fetches from a public URL. Anthropic and
    /// OpenAI-compatible endpoints.
    case imageURL(URL)

    /// A PDF, sent inline as base64. Anthropic only.
    case pdf(Data, title: String? = nil)

    /// A PDF the provider fetches from a public URL. Anthropic only.
    case pdfURL(URL, title: String? = nil)

    /// A plain-text document. Anthropic sends it as a `document` block that
    /// can be cited; other engines inline it into the prompt under its title.
    case text(String, title: String? = nil)

    /// A file already uploaded to the provider's Files API, by id. Anthropic
    /// only; `kind` says whether it goes in an `image` or a `document` block.
    case file(id: String, kind: FileKind)

    /// The content block an uploaded file belongs in.
    public enum FileKind: String, Sendable {
        case image
        case document
    }

    /// Whether this attachment is a plain-text document, the one kind every
    /// engine can carry (inline if nothing better).
    public var isText: Bool {
        if case .text = self { return true }
        return false
    }
}
