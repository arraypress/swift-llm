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

    let llm = LLM(engine)
    print("• prompt: \(prompt)")
    let t1 = Date()
    let reply = try await llm.generate(prompt)
    print(String(format: "• generated in %.1fs:\n", -t1.timeIntervalSinceNow))
    print(reply)
} catch {
    eprint("FAILED: \(error)\n")
    exit(1)
}
