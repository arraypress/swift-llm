//
//  AnthropicLiveTests.swift
//  LLMKitTests
//
//  The real API, on the cheapest model, one short request per feature. These
//  skip unless a key is present — `ANTHROPIC_API_KEY` in the environment, or
//  `credentials.anthropic.apiKey` in `~/.config/arraypress/credentials.json`
//  — so `swift test` stays free and offline by default. Run them on purpose:
//
//      swift test --filter AnthropicLiveTests
//
//  Each run costs well under a cent at Haiku 4.5 rates.
//

import XCTest
@testable import LLMKit
#if canImport(CoreGraphics)
import CoreGraphics
import CoreText
#endif

final class AnthropicLiveTests: XCTestCase {

    private static let model = "claude-haiku-4-5"

    /// The key, or `nil` to skip.
    private static let apiKey: String? = {
        if let key = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"], !key.isEmpty { return key }
        let url = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".config/arraypress/credentials.json")
        guard let data = try? Data(contentsOf: url), let json = try? JSONValue.parse(data) else { return nil }
        return json["credentials"]?["anthropic"]?["apiKey"]?.stringValue
    }()

    private func engine(_ configure: (inout AnthropicConfiguration) -> Void = { _ in }) throws -> AnthropicEngine {
        guard let key = Self.apiKey else { throw XCTSkip("no Anthropic key available") }
        var configuration = AnthropicConfiguration()
        configure(&configuration)
        return AnthropicEngine(model: Self.model, apiKey: key, configuration: configuration)
    }

    private let brief = GenerationOptions(temperature: 0, maxTokens: 200)

    /// A 1×1 red PNG.
    private let redPixel = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")!

    // MARK: - Models

    func testListModels() async throws {
        let models = try await engine().availableModels()
        XCTAssertFalse(models.isEmpty)
        // The listing carries dated ids (`claude-haiku-4-5-20251001`); the
        // undated alias is what requests use.
        XCTAssertTrue(models.contains { $0.id.hasPrefix(Self.model) }, "expected \(Self.model)* in \(models.map(\.id))")
        XCTAssertNotNil(models.first?.created)
    }

    // MARK: - Text

    func testRespond_andUsage() async throws {
        let response = try await engine().send([.system("Answer with one word."), .user("What colour is the sky on a clear day?")], options: brief)
        XCTAssertEqual(response.stopReason, .endTurn)
        XCTAssertTrue(response.text.lowercased().contains("blue"), response.text)
        XCTAssertGreaterThan(response.usage.inputTokens, 0)
        XCTAssertGreaterThan(response.usage.outputTokens, 0)
        XCTAssertEqual(response.model.hasPrefix("claude-haiku-4-5"), true)
    }

    func testStream_deltasAssembleToTheReply() async throws {
        var deltas: [String] = []
        var completed: AnthropicResponse?
        for try await event in try engine().streamEvents([.user("Count from 1 to 5, digits separated by spaces, nothing else.")], options: brief) {
            switch event {
            case .textDelta(let text): deltas.append(text)
            case .completed(let response): completed = response
            default: break
            }
        }
        XCTAssertGreaterThan(deltas.count, 1, "expected several chunks, got \(deltas)")
        XCTAssertEqual(deltas.joined(), completed?.text)
        XCTAssertEqual(completed?.stopReason, .endTurn)
        XCTAssertGreaterThan(completed?.usage.outputTokens ?? 0, 0)
    }

    func testFacadeStream_yieldsText() async throws {
        var text = ""
        for try await chunk in LLM(try engine()).stream("Say hello in exactly two words.", options: brief) { text += chunk }
        XCTAssertFalse(text.isEmpty)
    }

    func testStopSequence() async throws {
        let response = try await engine { $0.stopSequences = ["4"] }
            .send([.user("Count from 1 to 9, digits separated by spaces, nothing else.")], options: brief)
        XCTAssertEqual(response.stopReason, .stopSequence)
        XCTAssertEqual(response.stopSequence, "4")
        XCTAssertFalse(response.text.contains("5"))
    }

    // MARK: - Attachments

    func testImage_inlineBase64() async throws {
        let response = try await engine().send([.user("What colour is this image? One word.", images: [redPixel])], options: brief)
        XCTAssertTrue(response.text.lowercased().contains("red"), response.text)
    }

    func testTextDocument_withCitations() async throws {
        let doc = "The Scaffolder project was founded in Reykjavik. Its mascot is a puffin named Bramble."
        let response = try await engine { $0.citations = true }
            .send([.user("What is the mascot's name? Answer briefly.", attachments: [.text(doc, title: "Facts")])], options: brief)
        XCTAssertTrue(response.text.contains("Bramble"), response.text)
        XCTAssertFalse(response.citations.isEmpty, "expected at least one citation")
        XCTAssertEqual(response.citations.first?.documentTitle, "Facts")
    }

    #if canImport(CoreGraphics)
    func testPDF_inlineBase64() async throws {
        let pdf = Self.makePDF(text: "The secret word is marmalade.")
        let response = try await engine().send([.user("What is the secret word in this document? One word.", attachments: [.pdf(pdf, title: "Secret")])], options: brief)
        XCTAssertTrue(response.text.lowercased().contains("marmalade"), response.text)
    }

    func testFilesAPI_uploadUseDelete() async throws {
        let engine = try engine()
        let pdf = Self.makePDF(text: "The password is saffron.")
        let file = try await engine.uploadFile(pdf, filename: "live-test.pdf", mimeType: "application/pdf", expiresIn: 3600)
        XCTAssertTrue(file.id.hasPrefix("file_"))
        XCTAssertEqual(file.mimeType, "application/pdf")
        XCTAssertNotNil(file.expiresAt)
        defer { Task { try? await engine.deleteFile(id: file.id) } }

        let response = try await engine.send([.user("What is the password in this document? One word.", attachments: [.file(id: file.id, kind: .document)])], options: brief)
        XCTAssertTrue(response.text.lowercased().contains("saffron"), response.text)

        let metadata = try await engine.fileMetadata(id: file.id)
        XCTAssertEqual(metadata.id, file.id)
        let listed = try await engine.files(limit: 5)
        XCTAssertNotNil(listed.files)
        try await engine.deleteFile(id: file.id)
    }
    #endif

    // MARK: - Tools, structure, counting

    func testToolLoop_runsHandlerAndAnswers() async throws {
        let tool = AnthropicTool(
            name: "lookup_capital",
            description: "Call this to find the capital city of a country. Always use it rather than answering from memory.",
            inputSchema: ["type": "object", "properties": ["country": ["type": "string"]], "required": ["country"], "additionalProperties": false],
            strict: true
        ) { input in
            input["country"]?.stringValue?.lowercased() == "zembla" ? "Onhava" : "unknown"
        }
        let response = try await engine { $0.tools = [tool] }
            .run([.user("Use the tool to find the capital of Zembla, then reply with just the city name.")], options: GenerationOptions(temperature: 0, maxTokens: 300))
        XCTAssertEqual(response.stopReason, .endTurn)
        XCTAssertTrue(response.text.contains("Onhava"), response.text)
    }

    func testStructuredOutput_matchesSchema() async throws {
        struct Answer: Decodable { let city: String; let population_millions: Double }
        let response = try await engine {
            $0.outputSchema = ["type": "object",
                               "properties": ["city": ["type": "string"], "population_millions": ["type": "number"]],
                               "required": ["city", "population_millions"], "additionalProperties": false]
        }.send([.user("Give the largest city in Japan and its rough population in millions.")], options: brief)
        let answer = try JSONDecoder().decode(Answer.self, from: Data(response.text.utf8))
        XCTAssertEqual(answer.city.lowercased(), "tokyo")
        XCTAssertGreaterThan(answer.population_millions, 5)
    }

    func testCountTokens() async throws {
        let count = try await engine().countTokens([.system("Be brief."), .user("Hello there, how are you today?")])
        XCTAssertGreaterThan(count.inputTokens, 5)
        XCTAssertLessThan(count.inputTokens, 60)
    }

    func testThinkingBudget_onHaiku() async throws {
        let response = try await engine { $0.thinking = .budget(tokens: 1024) }
            .send([.user("What is 17 × 23? Reply with the number only.")], options: GenerationOptions(maxTokens: 2048))
        XCTAssertTrue(response.text.contains("391"), response.text)
        XCTAssertTrue(response.content.contains { if case .thinking = $0 { return true }; return false }, "expected a thinking block")
    }

    func testBadKey_surfacesProviderMessage() async {
        let engine = AnthropicEngine(model: Self.model, apiKey: "sk-ant-invalid")
        do {
            _ = try await engine.send([.user("hi")], options: brief)
            XCTFail("expected an error")
        } catch let error as LLMError {
            guard case .requestFailed(let detail) = error else { return XCTFail("unexpected \(error)") }
            XCTAssertTrue(detail.hasPrefix("HTTP 401"), detail)
        } catch {
            XCTFail("unexpected \(error)")
        }
    }

    // MARK: - Fixtures

    #if canImport(CoreGraphics)
    /// A one-page PDF with `text` drawn on it, made with CoreGraphics so it is
    /// a real PDF rather than a hand-typed one a strict parser might reject.
    private static func makePDF(text: String) -> Data {
        let data = NSMutableData()
        var mediaBox = CGRect(x: 0, y: 0, width: 400, height: 200)
        let consumer = CGDataConsumer(data: data as CFMutableData)!
        let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)!
        context.beginPDFPage(nil)
        let font = CTFontCreateWithName("Helvetica" as CFString, 18, nil)
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: [.font: font]))
        context.textPosition = CGPoint(x: 20, y: 100)
        CTLineDraw(line, context)
        context.endPDFPage()
        context.closePDF()
        return data as Data
    }
    #endif
}
