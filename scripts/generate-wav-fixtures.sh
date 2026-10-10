#!/bin/bash
# Regenerate the synthetic WAV fixtures used by yammer-stt/tests/wav_regression.rs.
#
# Each clip is spoken by espeak and then converted with ffmpeg into a different
# sample rate / sample format / channel layout, so the regression run exercises
# every branch of load_wav_24k (int and float decode, stereo mixdown, resampling
# and the 24 kHz pass-through). Every clip gets 0.5 s of lead-in and 1.5 s of
# trailing silence, as a real recording would have around the speech.
#
# espeak output differs between versions, so regenerating can change which
# words the model hears. Rerun the regression test after regenerating and adjust
# the expected keywords there if needed. The committed WAVs are the reference.
#
# armstrong_11k_u8.wav is not generated here: it is NASA's public-domain Apollo
# 11 recording, downloaded unmodified from
# https://commons.wikimedia.org/wiki/File:Neil_Armstrong_small_step.wav
# (sha256 9c2146638c3acba0c05643d18ee3a03fd2d5c167f89cf160579f97266b368cd2).
#
# Requires: espeak (or espeak-ng), ffmpeg

set -euo pipefail

OUT="$(cd "$(dirname "$0")/.." && pwd)/yammer-stt/tests/fixtures"
mkdir -p "$OUT"

ESPEAK="$(command -v espeak-ng || command -v espeak || true)"
if [ -z "$ESPEAK" ]; then
    echo "espeak or espeak-ng is required" >&2
    exit 1
fi
command -v ffmpeg > /dev/null || { echo "ffmpeg is required" >&2; exit 1; }

# make_clip NAME "TEXT" FFMPEG_OUTPUT_ARGS...
make_clip() {
    local name="$1" text="$2"
    shift 2
    "$ESPEAK" -v en-us -s 150 --stdout "$text" |
    ffmpeg -loglevel error -y -f wav -i pipe:0 \
        -af "adelay=500,apad=pad_dur=1.5" -map_metadata -1 -fflags +bitexact \
        "$@" "$OUT/$name.wav"
    echo "wrote $OUT/$name.wav"
}

make_clip meeting_22k_i16 \
    "Please schedule a meeting with the engineering team on Thursday afternoon." \
    -ar 22050 -ac 1 -c:a pcm_s16le

make_clip grocery_48k_stereo_i16 \
    "Remember to buy coffee, bread, and oranges from the grocery store." \
    -ar 48000 -ac 2 -c:a pcm_s16le

make_clip weather_24k_f32 \
    "The weather today is sunny with a gentle breeze." \
    -ar 24000 -ac 1 -c:a pcm_f32le

make_clip email_16k_i16 \
    "Send an email to the support team about the broken printer." \
    -ar 16000 -ac 1 -c:a pcm_s16le
