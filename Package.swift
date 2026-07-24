// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LLMKit",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        // Core: engine protocol + shared types + the zero-dependency engines
        // (Apple FoundationModels + any OpenAI-compatible cloud endpoint).
        .library(name: "LLMKit", targets: ["LLMKit"]),
        // Local downloaded MLX models (Qwen3 / Phi-4 / SmolLM3 / …) — opt-in.
        .library(name: "LLMKitMLX", targets: ["LLMKitMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "3.31.4"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMajor(from: "0.8.1")),
    ],
    targets: [
        .target(
            name: "LLMKit",
            dependencies: []
        ),
        .target(
            name: "LLMKitMLX",
            dependencies: [
                "LLMKit",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "LLMKitTests",
            dependencies: ["LLMKit"]
        ),
    ]
)
