//
//  LLMMessage.swift
//  LLMKit
//
//  A chat message — the common currency across every engine (Apple, MLX, cloud).
//  Images are carried as encoded data (JPEG/PNG) so the core stays cross-platform;
//  richer attachments (URLs, PDFs, documents, uploaded files) ride alongside as
//  `LLMAttachment`s. Each engine renders them into its own format and refuses
//  the kinds it cannot carry.
//

import Foundation

/// A single message in a conversation.
public struct LLMMessage: Sendable, Equatable {

    /// Who authored the message.
    public enum Role: String, Sendable, Codable {
        case system
        case user
        case assistant
    }

    /// The message role.
    public var role: Role

    /// The text content.
    public var text: String

    /// Optional images (encoded JPEG/PNG data) for vision-capable models.
    /// Equivalent to `LLMAttachment.image`; kept for the engines and callers
    /// that only know images.
    public var images: [Data]

    /// Other attachments: images by URL, PDFs, text documents, uploaded files.
    public var attachments: [LLMAttachment]

    /// Create a message.
    public init(role: Role, text: String, images: [Data] = [], attachments: [LLMAttachment] = []) {
        self.role = role
        self.text = text
        self.images = images
        self.attachments = attachments
    }

    /// Every attachment, `images` first as `.image`, then `attachments`.
    public var allAttachments: [LLMAttachment] {
        images.map(LLMAttachment.image) + attachments
    }

    /// A system / instructions message.
    public static func system(_ text: String) -> LLMMessage {
        LLMMessage(role: .system, text: text)
    }

    /// A user message, optionally with images and other attachments.
    public static func user(_ text: String, images: [Data] = [], attachments: [LLMAttachment] = []) -> LLMMessage {
        LLMMessage(role: .user, text: text, images: images, attachments: attachments)
    }

    /// An assistant message.
    public static func assistant(_ text: String) -> LLMMessage {
        LLMMessage(role: .assistant, text: text)
    }
}
