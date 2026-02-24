// SpeechBridge.swift — C-callable wrapper around Apple SpeechAnalyzer
//
// macOS 26 (Tahoe) introduces SpeechAnalyzer in the Speech framework,
// providing Whisper-class streaming STT on the Neural Engine.

import Foundation
import Speech
import AVFoundation

// MARK: - Thread-safe result queue

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

// MARK: - Global state

private var gAnalyzer: SpeechAnalyzer? = nil
private var gTranscriber: SpeechTranscriber? = nil
private var gResults = ResultQueue()
private var gSampleRate: UInt32 = 16000
private var gChannels: UInt32 = 1
private var gInputContinuation: AsyncStream<AnalyzerInput>.Continuation? = nil
private var gDrainTask: Task<Void, Never>? = nil
private var gFinished = false

// MARK: - C API

@_cdecl("apple_stt_init")
public func apple_stt_init(_ locale: UnsafePointer<CChar>?) -> Int32 {
    let localeStr = locale.map { String(cString: $0) } ?? "en-US"
    let nsLocale = Locale(identifier: localeStr)

    // Request speech recognition permission (blocks on first call)
    let semaphore = DispatchSemaphore(value: 0)
    var authStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined
    SFSpeechRecognizer.requestAuthorization { status in
        authStatus = status
        semaphore.signal()
    }
    semaphore.wait()

    if authStatus != .authorized {
        NSLog("[SpeechBridge] Speech recognition not authorized: \(authStatus.rawValue)")
        return -1
    }

    // Use a blocking semaphore to bridge async init into the sync C function.
    let initSema = DispatchSemaphore(value: 0)
    var initError: Int32 = 0

    Task {
        do {
            let transcriber = SpeechTranscriber(
                locale: nsLocale,
                preset: .progressiveTranscription
            )

            // Query the preferred audio format
            if let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) {
                gSampleRate = UInt32(format.sampleRate)
                gChannels = UInt32(format.channelCount)
            } else {
                // Fallback: 16kHz mono
                gSampleRate = 16000
                gChannels = 1
            }

            // Build an async stream that we control from the sync feed_pcm calls
            let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
            gInputContinuation = continuation

            let analyzer = SpeechAnalyzer(modules: [transcriber])
            gTranscriber = transcriber
            gAnalyzer = analyzer

            // Spawn a background task to drain results
            gFinished = false
            gDrainTask = Task {
                do {
                    for try await result in transcriber.results {
                        let text = String(result.text.characters)
                        if !text.isEmpty {
                            gResults.push(text)
                        }
                    }
                } catch {
                    NSLog("[SpeechBridge] result drain error: \(error)")
                }
                gFinished = true
            }

            // Start the analyzer with our input stream
            try await analyzer.start(inputSequence: stream)

            initError = 0
        } catch {
            NSLog("[SpeechBridge] init failed: \(error)")
            initError = -2
        }
        initSema.signal()
    }
    initSema.wait()
    return initError
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

    // Create an AVAudioPCMBuffer matching the analyzer's expected format
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

    // Copy samples into the buffer's float channel data
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
    // Signal end of audio input
    gInputContinuation?.finish()
    gInputContinuation = nil

    // Wait briefly for the drain task to finish
    let deadline = Date().addingTimeInterval(3.0)
    while !gFinished && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.01)
    }

    gDrainTask?.cancel()
    gDrainTask = nil
    gTranscriber = nil
    gAnalyzer = nil

    return 0
}
