# main.py
from __future__ import annotations
import argparse
import threading
import time
import webbrowser

import pyautogui

from actions import ActionBus, Action
from gesture_controller import GestureController
from voice_assistant import VoiceAssistant, VoiceConfig

pyautogui.FAILSAFE = True  # move mouse to top-left corner to stop PyAutoGUI

def executor_loop(bus: ActionBus, gesture: GestureController, voice: VoiceAssistant | None) -> None:
    screen_w, screen_h = pyautogui.size()
    mouse_is_down = False

    while True:
        action = bus.get()

        # Keep logs compact by skipping high-frequency cursor logs.
        if action.type != "MOUSE_MOVE":
            print(f"[Executor] Action: {action.type}")

        if action.type == "QUIT":
            break

        try:
            if action.type == "SAY" and action.text:
                print(f"[Assistant] {action.text}")
                if voice is not None:
                    voice.say(action.text)

            elif action.type == "TOGGLE_GESTURES" and action.enabled is not None:
                gesture.set_enabled(action.enabled)

            elif action.type == "OPEN_URL" and action.url:
                webbrowser.open(action.url)

            elif action.type == "MOUSE_MOVE":
                # action.x/action.y are 0..10000 normalized ints
                if action.x is None or action.y is None:
                    continue
                nx = max(0, min(10000, action.x)) / 10000.0
                ny = max(0, min(10000, action.y)) / 10000.0
                try:
                    pyautogui.moveTo(int(nx * screen_w), int(ny * screen_h), _pause=False)
                except pyautogui.FailSafeException:
                    # User moved mouse to top-left corner to trigger emergency stop.
                    bus.put(Action(type="QUIT"))
                    break

            elif action.type == "MOUSE_DOWN":
                if not mouse_is_down:
                    pyautogui.mouseDown()
                    mouse_is_down = True

            elif action.type == "MOUSE_UP":
                if mouse_is_down:
                    pyautogui.mouseUp()
                    mouse_is_down = False

            elif action.type == "CLICK":
                pyautogui.click()

            elif action.type == "DOUBLE_CLICK":
                pyautogui.doubleClick()

            elif action.type == "SCROLL" and action.amount is not None:
                pyautogui.scroll(int(action.amount))

            elif action.type == "TYPE_TEXT" and action.text:
                pyautogui.write(action.text, interval=0.01)
        except pyautogui.FailSafeException:
            print("[Executor] PyAutoGUI failsafe triggered. Move mouse away from top-left and restart.")
            bus.put(Action(type="QUIT"))
            break
        except Exception as e:
            print(f"[Executor] Error while handling {action.type}: {e}")

        time.sleep(0.001)

def _safe_worker(name: str, fn, bus: ActionBus) -> None:
    try:
        fn()
    except Exception as e:
        print(f"[Main] {name} thread crashed: {e}")
        bus.put(Action(type="QUIT"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture-based laptop controller")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice assistant")
    parser.add_argument("--voice-model", type=str, default=None, help="Path to Vosk model directory")
    parser.add_argument("--mic-device", type=int, default=None, help="Input microphone device index")
    parser.add_argument("--mic-samplerate", type=int, default=None, help="Microphone samplerate override")
    args = parser.parse_args()

    bus = ActionBus()
    gesture = GestureController(bus)
    # Start with both gesture and voice enabled for simultaneous operation.
    gesture.set_enabled(True)
    print("[Main] Gestures and voice are both active. Say 'stop gestures' to disable gestures.")
    
    # Voice assistant is optional - gesture control will work without it
    voice = None
    if not args.no_voice:
        try:
            vcfg = VoiceConfig(
                model_path=args.voice_model,
                device=args.mic_device,
                samplerate=args.mic_samplerate,
            )
            voice = VoiceAssistant(bus, cfg=vcfg)
        except Exception as e:
            print(f"[Main] Voice assistant initialization failed: {e}")
            print("[Main] Continuing with gesture control only")

    t_exec = threading.Thread(target=executor_loop, args=(bus, gesture, voice), daemon=True)
    t_gest = threading.Thread(target=_safe_worker, args=("Gesture", gesture.run, bus), daemon=True)
    
    t_exec.start()
    t_gest.start()
    
    # Only start voice thread if voice assistant is available
    t_voice = None
    if voice is not None:
        t_voice = threading.Thread(target=_safe_worker, args=("Voice", voice.run, bus), daemon=True)
        t_voice.start()

    # Keep main thread alive and ensure background worker failures terminate cleanly.
    gesture_stopped_notified = False
    while t_exec.is_alive():
        if not t_gest.is_alive() and not gesture_stopped_notified:
            print("[Main] Gesture thread stopped. Continuing in voice-only mode.")
            gesture_stopped_notified = True
        if t_voice is not None and not t_voice.is_alive() and voice is not None and voice.model is not None:
            print("[Main] Voice thread stopped unexpectedly. Exiting.")
            bus.put(Action(type="QUIT"))
            break
        time.sleep(0.25)

    t_exec.join()

if __name__ == "__main__":
    main()