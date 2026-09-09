"""
Full-duplex local speech-to-speech agent — instrumented.

Per-turn timing printed inline:
    stt         transcription
    llm_ttft    time to first LLM token  <- often the biggest slice
    llm_total   full generation
    tts_synth   synthesis per chunk
    tts_rtf     synth_time / audio_duration   (<1.0 = faster than realtime)
    ttfa        end-of-speech to first speaker sample

Code lives in src/asr_test/. To swap an engine (VAD/STT/LLM/TTS), pass a
different implementation of its interface (src/asr_test/interfaces/) into
Agent(...) — see src/asr_test/vad, stt, tts, llm for the built-in ones.
Pick a TTS engine at launch with --tts; VAD/STT/LLM currently have a
single built-in implementation each, so there's nothing to select yet.

Install:
    uv sync

Usage:
    uv run main.py                              # default: kokoro, af_bella
    uv run main.py --tts supertonic             # swap TTS engine
    uv run main.py --tts supertonic --voice M1  # + pick its voice

    # Trigger word (off by default): only text transcribed from your speech
    # that LEADS WITH this phrase (tolerant of a couple filler words, e.g.
    # "uh computer ...") is sent to the LLM; anything else is silently
    # ignored. No separate audio model - this runs on the normal STT
    # output, after VAD segments your speech as usual.
    uv run main.py --trigger-word "computer"

    # Pick an input device (index or a substring of its name):
    uv run main.py --list-mics          # print available input devices, then exit
    uv run main.py --mic "USB"

    # Mute: once running, press Enter in this terminal to toggle
    # muting the mic (frames are dropped before VAD ever sees them, so
    # the agent stays idle) — press Enter again to unmute.
"""

import argparse

from asr_test.agent import Agent
from asr_test.tts import KokoroTts, SupertonicTts
from asr_test.utils import list_input_devices, resolve_input_device, start_mute_toggle_listener

_TTS_ENGINES = {
    "kokoro": KokoroTts,
    "supertonic": SupertonicTts,
}


def main():
    parser = argparse.ArgumentParser(description="Full-duplex local speech-to-speech agent")
    parser.add_argument(
        "--tts", choices=sorted(_TTS_ENGINES), default="kokoro",
        help="TTS engine to use (default: kokoro)",
    )
    parser.add_argument(
        "--voice", default=None,
        help="Voice name, engine-specific (default: kokoro=af_bella, supertonic=F5)",
    )
    parser.add_argument(
        "--trigger-word", default=None, metavar="PHRASE",
        help="Only respond to speech whose transcript leads with this "
             "phrase (case-insensitive, tolerant of a couple leading "
             "filler words); everything else is ignored. Off by default "
             "(respond to everything, current behavior).",
    )
    parser.add_argument(
        "--mic", default=None, metavar="DEVICE",
        help="Input device to use: an index or a substring of its name "
             "(see --list-mics). Default: system default input device.",
    )
    parser.add_argument(
        "--list-mics", action="store_true",
        help="Print available input devices and exit.",
    )
    parser.add_argument(
        "--echo-mode", choices=("headphones", "duck", "aec"), default=None,
        help="Echo handling (default: config.ECHO_MODE, 'headphones'). Use "
             "'duck' or 'aec' if you're on speakers instead of real headphones.",
    )
    args = parser.parse_args()

    if args.list_mics:
        for line in list_input_devices():
            print(line)
        return

    device = resolve_input_device(args.mic)

    tts_kwargs = {"voice": args.voice} if args.voice else {}
    tts = _TTS_ENGINES[args.tts](**tts_kwargs)

    agent = Agent(tts=tts, trigger_word=args.trigger_word, echo_mode=args.echo_mode)
    start_mute_toggle_listener(agent.muted)
    agent.run(device=device)


if __name__ == "__main__":
    main()
