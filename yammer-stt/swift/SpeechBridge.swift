// SpeechBridge.swift — C-callable wrapper around Apple SpeechAnalyzer
//
// macOS 26 (Tahoe) introduces SpeechAnalyzer in the Speech framework,
// providing Whisper-class streaming STT on the Neural Engine.
//
// Architecture: All async work runs via DispatchQueue.global().async +
// Task {} rather than Task.detached, and results are drained with a
// bounded timeout. The key challenge is that SpeechAnalyzer requires
// Swift async/await, but the C FFI callers are synchronous. We use
// a GCD-initiated Task pattern where the Task is spawned from within
// a DispatchQueue.async block, keeping the cooperative pool usage
// transient. The drain task is replaced by periodic polling from the
// Rust side via apple_stt_poll_results.

import Foundation
import Speech
import AVFoundation

// MARK: - Thread-safe primitives

private final class ResultQueue: @unchecked Sendable {
    private var queue: [String] = []
    private let lock = NSLock()

    func push(_ text: String) {
        lock.lock()
        defer { lock.unlock() }
        queue.append(text)
    }

    func popAll() -> String {
        lock.lock()
        defer { lock.unlock() }
        let joined = queue.joined(separator: " ")
        queue.removeAll()
        return joined
    }
}

private final class AtomicInt32: @unchecked Sendable {
    private var value: Int32
    private let lock = NSLock()

    init(_ v: Int32) { value = v }

    func load() -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func store(_ v: Int32) {
        lock.lock()
        defer { lock.unlock() }
        value = v
    }
}

// MARK: - Global state

private let gResults = ResultQueue()
private var gSampleRate: UInt32 = 16000
private var gChannels: UInt32 = 1
private var gInputContinuation: AsyncStream<AnalyzerInput>.Continuation? = nil

/// Init state: 0 = pending, 1 = success, negative = error
private let gInitState = AtomicInt32(0)

/// Whether the result drain loop has been started (to avoid double-start).
private var gDrainStarted = false

// MARK: - Dedicated queue for all Speech work
private let gSpeechQueue = DispatchQueue(label: "yammer.speech-bridge", qos: .userInitiated)

// MARK: - C API

@_cdecl("apple_stt_init")
public func apple_stt_init(_ locale: UnsafePointer<CChar>?) -> Int32 {
    let localeStr = locale.map { String(cString: $0) } ?? "en-US"
    let nsLocale = Locale(identifier: localeStr)

    gInitState.store(0)
    gDrainStarted = false

    // Launch async init from our dedicated GCD queue.
    // The Task spawned inside runs on the cooperative pool, but
    // it's short-lived (just init + start, then returns).
    gSpeechQueue.async {
        Task {
            do {
                let transcriber = SpeechTranscriber(
                    locale: nsLocale,
                    preset: .progressiveTranscription
                )

                if let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) {
                    gSampleRate = UInt32(format.sampleRate)
                    gChannels = UInt32(format.channelCount)
                } else {
                    gSampleRate = 16000
                    gChannels = 1
                }

                let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
                gInputContinuation = continuation

                let analyzer = SpeechAnalyzer(modules: [transcriber])

                try await analyzer.start(inputSequence: stream)

                // Start draining results in a fire-and-forget manner.
                // This task will run on the cooperative pool but is bounded
                // by the lifetime of the result stream.
                if !gDrainStarted {
                    gDrainStarted = true
                    Task.detached { [weak transcriber] in
                        guard let transcriber = transcriber else { return }
                        do {
                            for try await result in transcriber.results {
                                let text = String(result.text.characters)
                                if !text.isEmpty {
                                    gResults.push(text)
                                }
                            }
                        } catch {
                            // Expected when input ends
                        }
                    }
                }

                gInitState.store(1) // success

                // Keep analyzer alive by storing globally.
                // Using withExtendedLifetime to prevent optimizer from
                // releasing it before we're done.
                withExtendedLifetime(analyzer) {}

            } catch {
                NSLog("[SpeechBridge] init failed: \(error)")
                gInitState.store(-2)
            }
        }
    }

    // Spin-poll from the calling (Rust) thread.
    let deadline = Date().addingTimeInterval(10.0)
    while gInitState.load() == 0 && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.005)
    }

    let state = gInitState.load()
    if state == 0 {
        NSLog("[SpeechBridge] init timed out")
        return -3
    }
    return state == 1 ? 0 : state
}

@_cdecl("apple_stt_get_format")
public func apple_stt_get_format(
    _ rate: UnsafeMutablePointer<UInt32>?,
    _ channels: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    rate?.pointee = gSampleRate
    channels?.pointee = gChannels
    return 0
}

@_cdecl("apple_stt_feed_pcm")
public func apple_stt_feed_pcm(
    _ samples: UnsafePointer<Float>?,
    _ count: UInt32
) -> Int32 {
    guard let samples = samples, count > 0, let continuation = gInputContinuation else {
        return -1
    }

    guard let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(gSampleRate),
        channels: AVAudioChannelCount(gChannels),
        interleaved: false
    ) else {
        return -2
    }

    guard let pcmBuffer = AVAudioPCMBuffer(
        pcmFormat: format,
        frameCapacity: AVAudioFrameCount(count)
    ) else {
        return -3
    }

    pcmBuffer.frameLength = AVAudioFrameCount(count)

    if let channelData = pcmBuffer.floatChannelData {
        channelData[0].update(from: samples, count: Int(count))
    }

    continuation.yield(AnalyzerInput(buffer: pcmBuffer))
    return 0
}

@_cdecl("apple_stt_poll_results")
public func apple_stt_poll_results(
    _ buf: UnsafeMutablePointer<UInt8>?,
    _ len: UInt32
) -> Int32 {
    guard let buf = buf else { return -1 }

    let text = gResults.popAll()
    if text.isEmpty { return 0 }

    let data = Array(text.utf8)
    let copyLen = min(data.count, Int(len))
    data.withUnsafeBufferPointer { src in
        buf.update(from: src.baseAddress!, count: copyLen)
    }
    return Int32(copyLen)
}

@_cdecl("apple_stt_stop")
public func apple_stt_stop() -> Int32 {
    // Finish the input stream. Do NOT deallocate anything —
    // the cooperative pool may still have references. The OS
    // reclaims everything on process exit.
    gInputContinuation?.finish()
    gInputContinuation = nil
    return 0
}
