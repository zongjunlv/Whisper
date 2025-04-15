import uvicorn
import pyaudio
import subprocess

from pydantic import BaseModel
from pydub import AudioSegment
from pydub.playback import play 
from fastapi import FastAPI, File, UploadFile


app = FastAPI()
stop_flag = False


@app.post("/audio")
async def save_audio(audio: UploadFile = File(...)):
    # 获取音频文件的内容
    audio_content = await audio.read()

    # 保存音频文件到磁盘
    with open(f"audio/{audio.filename}", "wb") as file:
        file.write(audio_content)


    audio = AudioSegment.from_file(f"audio/{audio.filename}", format="wav")

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio.sample_width),
                channels=audio.channels,
                rate=audio.frame_rate,
                output=True)

    global stop_flag
    while len(audio) > 0 and not stop_flag:
        # 获取当前时间戳
        #print(len(audio))
        stream.start_stream()
        current_time = audio[:1000].duration_seconds
        stream.write(audio_data.tobytes())
        # 播放 1 秒的音频
        stream.write(audio[:1000]._data)
        audio = audio[1000:]
        with open('./stop.txt', 'r') as f:
            my_flag = f.read().strip()
        if my_flag == "True" :
            stop_flag = True

    with open('./stop.txt', 'w') as file:
        file.write("False")
    stop_flag = False
    stream.stop_stream()
        
    return {"message": "Audio saved successfully"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8017)