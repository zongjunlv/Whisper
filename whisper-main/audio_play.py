import sys
import json
import wave
import pyaudio
import asyncio

import numpy as np
import sounddevice as sd

# async def play_audio(file_path):
#     chunk = 1024

#     wf = wave.open(file_path, 'rb')
#     p = pyaudio.PyAudio()

#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True)

#     data = wf.readframes(chunk)

#     while data:
#         stream.write(data)
#         data = wf.readframes(chunk)

#     stream.stop_stream()
#     stream.close()

#     p.terminate()


async def main():
    #audio_file = "playaudio.wav"
    audio_str = sys.stdin.read()
    # 将数据转换回numpy数组
    audio = np.asarray(json.loads(audio_str))
    sample_rate = 44100  # 采样率（每秒样本数）

    # 播放音频数据
    sd.play(audio, sample_rate,blocking=False)
    sd.wait()

asyncio.run(main())