import sys
import json
import os
from pathlib import Path
from elevenlabs.client import ElevenLabs

BASE_DIR = Path(__file__).resolve().parent
AUDIO_FILE_PATH = BASE_DIR / "transcribe.flac"
OUTPUT_JSON_PATH = BASE_DIR / "transcript.json"

def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")

    # If not set in environment, try loading from .env file.
    # Prefer python-dotenv if available, otherwise fall back to a simple parser.
    if not api_key:
        try:
            # prefer python-dotenv when installed
            from dotenv import load_dotenv
            load_dotenv(str(BASE_DIR / ".env"))
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                print("Loaded ELEVENLABS_API_KEY from .env (python-dotenv)")
        except Exception:
            env_path = BASE_DIR / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r", encoding="utf-8") as ef:
                        for line in ef:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('\"\'')
                            if k == "ELEVENLABS_API_KEY" and v:
                                os.environ[k] = v
                                api_key = v
                                print("Loaded ELEVENLABS_API_KEY from .env (manual)")
                                break
                except Exception:
                    # ignore parsing errors and continue to final check
                    pass

    if not api_key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY. Set it as an environment variable or add 'ELEVENLABS_API_KEY=your_key' to a .env file in the project folder.")

    print("Python:", sys.executable)
    print("Audio exists:", AUDIO_FILE_PATH.exists())

    client = ElevenLabs(api_key=api_key)

    with open(AUDIO_FILE_PATH, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v2",
            diarize=True,
            tag_audio_events=True,
            language_code="uzb",   # <- force Uzbek
        )


    # result is a model object, convert to plain dict first
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    elif hasattr(result, "dict"):
        payload = result.dict()
    else:
        payload = result  # fallback (in case it's already dict)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)


    print("Done:", OUTPUT_JSON_PATH)

if __name__ == "__main__":
    main()
