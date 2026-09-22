//
//  ISO8601.swift
//  LLMKit
//
//  Providers stamp times as RFC 3339, sometimes with fractional seconds and
//  sometimes without, in the same API (Anthropic's file records carry
//  microseconds; its model records don't). `ISO8601DateFormatter` parses
//  exactly one of those shapes per instance, so a date with `.350741Z`
//  silently became `nil`. This tries both. Formatters are made per call:
//  the type isn't `Sendable`, and a handful of dates per reply is cheap.
//

import Foundation

/// Lenient RFC 3339 parsing.
enum ISO8601 {

    /// The date in `string`, with or without fractional seconds; `nil` when
    /// it is neither.
    static func date(from string: String?) -> Date? {
        guard let string else { return nil }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: string) { return date }
        let whole = ISO8601DateFormatter()
        whole.formatOptions = [.withInternetDateTime]
        return whole.date(from: string)
    }
}
