#ifndef SPEECH_BRIDGE_H
#define SPEECH_BRIDGE_H

#include <stdint.h>

/// Initialise the SpeechAnalyzer pipeline for the given locale (e.g. "en-US").
/// Returns 0 on success.
int32_t apple_stt_init(const char *locale);

/// Query the audio format expected by the recogniser.
/// Writes sample rate and channel count into the provided pointers.
/// Returns 0 on success.
int32_t apple_stt_get_format(uint32_t *rate, uint32_t *channels);

/// Feed PCM f32 samples to the recogniser.
/// Returns 0 on success.
int32_t apple_stt_feed_pcm(const float *samples, uint32_t count);

/// Poll accumulated transcription text.
/// Copies up to `len` bytes of UTF-8 text into `buf`.
/// Returns the number of bytes written, or 0 if no new text, or -1 on error.
int32_t apple_stt_poll_results(uint8_t *buf, uint32_t len);

/// Stop the recogniser session and clean up.
/// Returns 0 on success.
int32_t apple_stt_stop(void);

#endif /* SPEECH_BRIDGE_H */
