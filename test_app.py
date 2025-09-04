import unittest
import os
import shutil
import subprocess
import time
from app import (
    generate_tts_audio,
)
from openai import OpenAI

if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set, skipping TTS integration test.")
class TestTTSGenerationIntegration(unittest.TestCase):
    def test_generate_tts_audio_mp3_real(self):
        client = OpenAI()  # uses env OPENAI_API_KEY
        tmp_path = None
        try:
            start = time.perf_counter()
            tmp_path, media_type = generate_tts_audio(
                text="This is a short devotional audio test.",
                language="en",
                voice=None,
                fmt="mp3",
                openai_client=client,
            )
            elapsed = time.perf_counter() - start
            self.assertTrue(os.path.exists(tmp_path), "Temp audio file should exist")
            self.assertEqual(media_type, "audio/mpeg")
            self.assertGreater(os.path.getsize(tmp_path), 128, "MP3 file should have content")
            print(f"TTS MP3 generation took {elapsed:.2f}s")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                # play the audio with mpg123
                try:
                    if shutil.which("mpg123"):
                        subprocess.run(["mpg123", "-q", tmp_path], check=False, timeout=5)
                    else:
                        print("mpg123 not found; skipping playback")
                except Exception as e:
                    print(f"mpg123 playback skipped: {e}")

    def test_generate_tts_audio_wav_real(self):
        client = OpenAI()
        tmp_path = None
        try:
            tmp_path, media_type = generate_tts_audio(
                text="这是一个中文语音测试。愿你平安。",
                language="zh",
                voice=None,
                fmt="wav",
                openai_client=client,
            )
            self.assertTrue(os.path.exists(tmp_path))
            self.assertEqual(media_type, "audio/wav")
            self.assertGreater(os.path.getsize(tmp_path), 128, "WAV file should have content")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                print(tmp_path)
                # os.remove(tmp_path)


 
