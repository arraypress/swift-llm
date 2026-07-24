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
    ],
    targets: [
        .target(
            name: "LLMKit",
            dependencies: []
        ),
        .testTarget(
            name: "LLMKitTests",
            dependencies: ["LLMKit"]
        ),
    ]
)
