# voice_assistant.py
from __future__ import annotations
import os
import json
import queue
import re
import time
import webbrowser
import subprocess
from difflib import SequenceMatcher
from dataclasses import dataclass

import pyttsx3
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from actions import Action, ActionBus

@dataclass
class VoiceConfig:
    # Auto-discovery checks multiple common paths when model_path is None.
    model_path: str | None = None
    # If None, infer from selected input device.
    samplerate: int | None = None
    device: int | None = None  # set if you have multiple mics

class VoiceAssistant:
    def __init__(self, bus: ActionBus, cfg: VoiceConfig = VoiceConfig()) -> None:
        self.bus = bus
        self.cfg = cfg
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 175)
        self.tts.setProperty("volume", 1.0)
        self.use_windows_tts = os.name == "nt"

        # Resolve input device and sample rate early so stream and recognizer match.
        self.device = self._resolve_device(cfg.device)
        self.samplerate = self._resolve_samplerate(self.device, cfg.samplerate)
        print(f"[VoiceAssistant] Using mic device={self.device} samplerate={self.samplerate}")

        # Try to load Vosk model, but continue without it if not available.
        self.model = None
        self.rec = None
        self.model_path = self._resolve_model_path(cfg.model_path)
        try:
            self.model = Model(self.model_path)
            self.rec = KaldiRecognizer(self.model, self.samplerate)
            self.rec.SetWords(False)
            print(f"[VoiceAssistant] Vosk model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"[VoiceAssistant] Warning: Vosk model not available ({e})")
            print("[VoiceAssistant] Voice commands disabled. Gesture control only.")

        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._last_partial_dispatch = 0.0
        self._last_partial_text = ""
        self._ignore_audio_until = 0.0

    def _reply(self, text: str) -> None:
        """Speak feedback directly so voice replies don't depend on executor thread."""
        print(f"[Assistant] {text}")
        # Temporarily suppress incoming mic audio while assistant is speaking.
        self._ignore_audio_until = time.time() + 1.8
        spoken = False

        # Prefer Windows built-in System.Speech on Windows for better reliability.
        if self.use_windows_tts:
            try:
                self._speak_windows(text)
                spoken = True
            except Exception as e:
                print(f"[VoiceAssistant] Windows TTS error: {e}")

        if not spoken:
            try:
                self.say(text)
                spoken = True
            except Exception as e:
                print(f"[VoiceAssistant] pyttsx3 TTS error: {e}")

        # Audible fallback if both engines fail.
        if not spoken and os.name == "nt":
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass

    @staticmethod
    def _speak_windows(text: str) -> None:
        safe = text.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Volume = 100; $s.Rate = 0; "
            f"$s.Speak('{safe}')"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=True)

    @staticmethod
    def _resolve_device(configured_device: int | None) -> int | None:
        if configured_device is not None:
            try:
                d = sd.query_devices(configured_device, "input")
                if d.get("max_input_channels", 0) > 0:
                    return configured_device
            except Exception:
                # Fall through to auto-selection.
                pass

        try:
            devices = sd.query_devices()
            best_idx = None
            best_score = -10

            for i, d in enumerate(devices):
                if d.get("max_input_channels", 0) <= 0:
                    continue

                name = str(d.get("name", "")).lower()
                score = 0

                # Prefer explicit microphone devices.
                if "microphone" in name:
                    score += 5
                if "array" in name:
                    score += 2
                if "realtek" in name:
                    score += 1

                # Avoid virtual / loopback / generic mapper devices.
                if "stereo mix" in name:
                    score -= 6
                if "speaker" in name:
                    score -= 5
                if "sound mapper" in name:
                    score -= 4
                if "primary sound capture" in name:
                    score -= 3

                if score > best_score:
                    best_score = score
                    best_idx = i

            return best_idx
        except Exception:
            return None

    @staticmethod
    def _resolve_samplerate(device: int | None, configured_samplerate: int | None) -> int:
        if configured_samplerate is not None:
            return int(configured_samplerate)
        try:
            info = sd.query_devices(device, "input")
            return int(info["default_samplerate"])
        except Exception:
            return 16000

    @staticmethod
    def _resolve_model_path(configured_path: str | None) -> str:
        """Resolve first available Vosk model path from common locations."""
        candidates = []

        env_path = os.getenv("VOSK_MODEL_PATH")
        if env_path:
            candidates.append(env_path)

        if configured_path:
            candidates.append(configured_path)

        candidates.extend(
            [
                "model",
                "models/vosk-model-small-en-us-0.15",
                "models/vosk-model-en-us-0.42-gigaspeech",
            ]
        )

        for path in candidates:
            if path and os.path.isdir(path):
                return path

        # Return preferred default so downstream error message is explicit.
        return candidates[0] if candidates else "model"

    def say(self, text: str) -> None:
        self.tts.say(text)
        self.tts.runAndWait()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # ignore status spam; could log if desired
            pass
        if time.time() < self._ignore_audio_until:
            return
        self._audio_q.put(bytes(indata))

    @staticmethod
    def _similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _normalize_text(text: str) -> str:
        # Keep only letters/spaces and compress long repeated characters.
        t = re.sub(r"[^a-z\s]", " ", text.lower())
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _contains_any_token(tokens: list[str], vocab: set[str]) -> bool:
        return any(tok in vocab for tok in tokens)

    def _handle_text(self, text: str) -> None:
        t = self._normalize_text(text)
        if not t:
            return

        print(f"[VoiceAssistant] Heard: {t}")
        token_list = t.split()
        words = set(token_list)

        def near(target: str, threshold: float = 0.72) -> bool:
            return self._similar(t, target) >= threshold

        def has_word_like(target: str, threshold: float = 0.72) -> bool:
            return any(self._similar(w, target) >= threshold for w in token_list)

        def has_open_like() -> bool:
            return "open" in words or has_word_like("open", 0.70)

        def has_google_like() -> bool:
            google_aliases = {"google", "gogle", "googl", "gooogle", "googel", "gogole"}
            return self._contains_any_token(token_list, google_aliases) or has_word_like("google", 0.62)

        def has_youtube_like() -> bool:
            youtube_aliases = {"youtube", "youtub", "yutube", "yootube", "utube", "ytube", "tube"}
            lead_aliases = {"you", "yu", "yuu", "u"}
            joined = " ".join(token_list)
            pair_match = (
                (self._contains_any_token(token_list, lead_aliases) and "tube" in words)
                or "you tube" in joined
                or "u tube" in joined
            )
            return pair_match or self._contains_any_token(token_list, youtube_aliases) or has_word_like("youtube", 0.60)

        def has_telegram_like() -> bool:
            telegram_aliases = {"telegram", "telegraph", "tele", "gram"}
            joined = " ".join(token_list)
            pair_match = "tele gram" in joined
            return pair_match or self._contains_any_token(token_list, telegram_aliases) or has_word_like("telegram", 0.62)

        # Ignore assistant's own spoken phrases to prevent feedback loops.
        if (
            "sorry i did not understand" in t
            or "i am listening" in t
            or "try commands like" in t
            or "voice assistant started" in t
            or "please say a supported command" in t
            or "please say one clear command" in t
        ):
            return

        # Conversational confirmations (helps users verify mic+STT is working).
        if (
            t in {"hi", "hello", "hey", "voice assistant", "voice assistants"}
            or "what s happening" in t
            or near("what is happening")
            or near("are you there")
        ):
            self._reply("I am listening. Try commands like start gestures, click, or open google.")
            return

        # Quit
        if (
            t in {"quit", "exit", "stop program", "close program", "quick"}
            or near("quit")
            or near("exit")
        ):
            self._reply("Quitting.")
            self.bus.put(Action(type="QUIT"))
            return

        # Toggle gestures
        if (
            "stop gestures" in t
            or "disable gestures" in t
            or ("stop" in words and "gesture" in t)
            or near("stop gestures")
        ):
            self.bus.put(Action(type="TOGGLE_GESTURES", enabled=False))
            self._reply("Gestures disabled.")
            return

        if (
            "start gestures" in t
            or "enable gestures" in t
            or ("start" in words and "gesture" in t)
            or near("start gestures")
            or near("enable gestures")
        ):
            self.bus.put(Action(type="TOGGLE_GESTURES", enabled=True))
            self._reply("Gestures enabled.")
            return

        # Mouse actions
        if "double click" in t or ("double" in words and "click" in words):
            self.bus.put(Action(type="DOUBLE_CLICK"))
            return

        if t in {"click", "mouse click"} or near("click"):
            self.bus.put(Action(type="CLICK"))
            return

        if "scroll up" in t or ("scroll" in words and "up" in words):
            self.bus.put(Action(type="SCROLL", amount=400))
            return

        if "scroll down" in t or ("scroll" in words and "down" in words):
            self.bus.put(Action(type="SCROLL", amount=-400))
            return

        # Type command: "type hello world"
        m = re.match(r"^(type|write)\s+(.*)$", t)
        if m:
            self.bus.put(Action(type="TYPE_TEXT", text=m.group(2)))
            return

        # Open sites
        if (
            "open youtube" in t
            or (has_open_like() and has_youtube_like())
            or (has_youtube_like() and len(token_list) <= 3)
        ):
            self.bus.put(Action(type="OPEN_URL", url="https://www.youtube.com"))
            self._reply("Opening YouTube.")
            return

        if (
            "open google" in t
            or (has_open_like() and has_google_like())
            or (has_google_like() and len(token_list) <= 3)
        ):
            self.bus.put(Action(type="OPEN_URL", url="https://www.google.com"))
            self._reply("Opening Google.")
            return

        if (
            "open telegram" in t
            or (has_open_like() and has_telegram_like())
            or (has_telegram_like() and len(token_list) <= 3)
        ):
            self.bus.put(Action(type="OPEN_URL", url="https://web.telegram.org"))
            self._reply("Opening Telegram Web.")
            return

        if "time" in t:
            now = time.strftime("%I:%M %p")
            self._reply(f"It is {now}.")
            return

        # Ignore non-command speech to avoid noisy loops.
        return

    def run(self) -> None:
        if self.model is None or self.rec is None:
            print("[VoiceAssistant] Voice recognition not available, voice thread exiting")
            return
        
        self._reply("Voice assistant started.")

        try:
            stream = sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=8000,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=self._callback,
            )
        except Exception as e:
            print(f"[VoiceAssistant] Input stream failed on device={self.device}: {e}")
            print("[VoiceAssistant] Retrying with default microphone device.")
            self.device = None
            stream = sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=8000,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=self._callback,
            )

        with stream:
            while True:
                data = self._audio_q.get()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "")
                    self._handle_text(text)
                else:
                    partial = json.loads(self.rec.PartialResult()).get("partial", "")
                    if partial:
                        print(f"[VoiceAssistant] Listening: {partial}")
                        p = partial.strip().lower()
                        has_command_token = any(
                            token in p
                            for token in [
                                "click",
                                "double",
                                "scroll",
                                "start",
                                "stop",
                                "open",
                                "google",
                                "youtube",
                                "quit",
                                "exit",
                                "time",
                                "type",
                                "write",
                            ]
                        )

                        # Promote stable partial command text to handler for faster responses.
                        now = time.time()
                        if has_command_token and p != self._last_partial_text and (now - self._last_partial_dispatch) > 1.0:
                            self._last_partial_text = p
                            self._last_partial_dispatch = now
                            self._handle_text(p)