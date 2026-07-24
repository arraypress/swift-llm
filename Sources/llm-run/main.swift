//
//  main.swift — llm-run
//
//  On-device smoke test for the MLX engine: download a small model and run one
//  generation, proving LLMKitMLX works end-to-end on the Metal GPU.
//
//  Usage: llm-run [modelID] [prompt]
//    modelID defaults to qwen3-0.6b (smallest, ~450MB, Apache-2.0).
//
//  NOTE: mlx-swift needs a Metal library that only ships in Xcode app builds.
//  To run headless, colocate `mlx.metallib` next to the binary first (see
//  CLAUDE.md).
//

import Foundation
import LLMKit
import LLMKitMLX

func eprint(_ s: String) { FileHandle.standardError.write(Data(s.utf8)) }

let args = CommandLine.arguments
let modelID = args.count > 1 ? args[1] : "qwen3-0.6b"
let prompt  = args.count > 2 ? args[2] : "In one sentence, what is a haiku?"

guard let model = MLXModel.all.first(where: { $0.id == modelID }) else {
    let ids = MLXModel.all.map(\.id).joined(separator: ", ")
    eprint("unknown model '\(modelID)'. Options: \(ids)\n")
    exit(2)
}

do {
    let engine = MLXLLMEngine(model)
    print("• model:  \(model.displayName)  [\(model.repoID)]  ~\(model.approximateSizeMB) MB  \(model.license)")
    print("• preparing (downloads on first run, then loads on GPU)…")
    let t0 = Date()
    try await engine.prepare { p in
        // Inlined so nothing non-Sendable is captured by this @Sendable closure.
        FileHandle.standardError.write(Data("\r  \(Int(p * 100))%   ".utf8))
    }
    eprint("\r")
    print(String(format: "• ready in %.1fs", -t0.timeIntervalSinceNow))

    print("• prompt: \(prompt)")
    var msgs: [LLMMessage] = [.user(prompt)]
    if args.count > 3, let d = try? Data(contentsOf: URL(fileURLWithPath: args[3])) {
        msgs = [.user(prompt, images: [d])]   // attach an image (VLM test)
        print("• attached image: \(args[3]) (\(d.count) bytes)")
    }
    print("• streaming:\n")
    let t1 = Date()
    var full = ""
    for try await chunk in engine.streamResponse(to: msgs, options: GenerationOptions(temperature: 0.7)) {
        full += chunk
        FileHandle.standardOutput.write(Data(chunk.utf8))   // live tokens
    }
    let split = full.splitReasoning()
    print(String(format: "\n\n• generated in %.1fs", -t1.timeIntervalSinceNow))
    if let reasoning = split.reasoning {
        print("• (\(reasoning.count) chars of <think> reasoning split out)")
        print("• answer:\n\(split.answer)")
    }
} catch {
    eprint("FAILED: \(error)\n")
    exit(1)
}
