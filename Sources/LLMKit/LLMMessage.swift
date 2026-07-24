//
//  LLMMessage.swift
//  LLMKit
//
//  A chat message — the common currency across every engine (Apple, MLX, cloud).
//  Images are carried as encoded data (JPEG/PNG) so the core stays cross-platform;
//  each engine renders them into its own format (Apple `.image()`, OpenAI
//  `image_url`, MLX pixel input).
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
    public var images: [Data]

    /// Create a message.
    public init(role: Role, text: String, images: [Data] = []) {
        self.role = role
        self.text = text
        self.images = images
    }

    /// A system / instructions message.
    public static func system(_ text: String) -> LLMMessage {
        LLMMessage(role: .system, text: text)
    }

    /// A user message, optionally with images.
    public static func user(_ text: String, images: [Data] = []) -> LLMMessage {
        LLMMessage(role: .user, text: text, images: images)
    }

    /// An assistant message.
    public static func assistant(_ text: String) -> LLMMessage {
        LLMMessage(role: .assistant, text: text)
    }
}
