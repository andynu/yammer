fn main() {
    #[cfg(target_os = "macos")]
    build_swift_bridge();
}

#[cfg(target_os = "macos")]
fn build_swift_bridge() {
    use std::process::Command;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let swift_src = format!("{}/swift/SpeechBridge.swift", manifest_dir);
    let lib_path = format!("{}/libSpeechBridge.a", out_dir);

    // Compile Swift source into a static library
    let status = Command::new("swiftc")
        .args([
            "-emit-library",
            "-static",
            "-O",
            "-module-name", "SpeechBridge",
            "-o", &lib_path,
            &swift_src,
        ])
        .status()
        .expect("Failed to run swiftc — is Xcode installed?");

    if !status.success() {
        panic!("swiftc compilation failed");
    }

    // Link the static library and required frameworks
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=SpeechBridge");

    // Link Swift standard library search path
    let toolchain_lib = String::from_utf8(
        Command::new("swift")
            .args(["-print-target-info"])
            .output()
            .expect("swift -print-target-info failed")
            .stdout,
    )
    .unwrap();

    // Parse the runtimeLibraryPaths from swift -print-target-info JSON
    if let Ok(info) = serde_json::from_str::<serde_json::Value>(&toolchain_lib) {
        if let Some(paths) = info["paths"]["runtimeLibraryPaths"].as_array() {
            for path in paths {
                if let Some(p) = path.as_str() {
                    println!("cargo:rustc-link-search=native={}", p);
                }
            }
        }
    }

    // macOS frameworks needed by Speech + AVFoundation
    println!("cargo:rustc-link-lib=framework=Speech");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Re-run if Swift source changes
    println!("cargo:rerun-if-changed=swift/SpeechBridge.swift");
    println!("cargo:rerun-if-changed=include/speech_bridge.h");
}
