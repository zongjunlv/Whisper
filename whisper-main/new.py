import io
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import wave  # 使用wave库可读、写wav类型的音频文件
import json
import re
import asyncio
import queue
import torch
import zhconv
import whisper
import pyaudio
import requests
import threading
import websockets
#import subprocess
import numpy as np
import difflib as df
import sounddevice as sd
import pandas.io.clipboard as cb

from httpx import post
from pypinyin import pinyin
from pydub import AudioSegment
from pydub.playback import play


# 定义数据流块
temp = 20
CHUNK = 1024  # 音频帧率（也就是每次读取的数据是多少，默认1024）
FORMAT = pyaudio.paInt16  # 采样时生成wav文件正常格式
CHANNELS = 1  # 音轨数（每条音轨定义了该条音轨的属性,如音轨的音色、音色库、通道数、输入/输出端口、音量等。可以多个音轨，不唯一）
RATE = 16000  # 采样率（即每秒采样多少数据）
mindb = 500 # 最小声音，大于则开始录音，否则结束
delayTime = 0.4
#RECORD_SECONDS = time  # 录音时间
play_thread = None
stop_flag = False
WAVE_OUTPUT_FILENAME = "./output.wav"  # 保存音频路径

p = pyaudio.PyAudio()  # 创建PyAudio对象
stream = p.open(format=FORMAT,  # 采样生成wav文件的正常格式
                channels=CHANNELS,  # 音轨数
                rate=RATE,  # 采样率
                input=True,  # Ture代表这是一条输入流，False代表这不是输入流
                frames_per_buffer=CHUNK)  # 每个缓冲多少帧

#读取行动指令文件
f_command = open("./command.txt", "r", encoding='utf-8')
command_lists = f_command.readlines()

commandset_e = []
commandset_c = []
for command in command_lists:
    data1 = command.strip('\n')  # 去掉开头和结尾的换行符
    data2 = data1.split("_")
    commandset_c.append(data2[0])
    commandset_e.append(data2[1])

#读取唤醒文件   
f_wake = open("./wake.txt", "r", encoding='utf-8')
wake_lists = f_wake.readlines()
wakeset_e = []
wakeset_c = []
for wake in wake_lists:
    wakedata1 = wake.strip('\n')  # 去掉开头和结尾的换行符
    wakedata2 = wakedata1.split("_")
    wakeset_c.append(wakedata2[0])
    wakeset_e.append(wakedata2[1])

#读取停止文件
with open('./pause.txt', 'r') as f:
    match_stop = f.read().strip()
#切分句子函数
def split_sentences(text):
    sentences = re.split(r"\.|\！|\？|\n|\,|\。|\?|\!|\，", text)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences
#收音函数
def chat_to_glm():
    frames = []  # 定义frames为一个空列表
    audio_frames = []
    flag = False # 开始录音的节点
    stat = True
    stat2 = False
    tempnum = 0
    tempnum2 = 0
    print("user:")  # 开始录音标志
    while(stat):
        data = stream.read(CHUNK,exception_on_overflow = False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.short)
        audio_data = audio_data.astype(np.float32)
        audio_frames.append(audio_data)
        temp = np.max(audio_data) #计算音频的峰值
        if temp > mindb and flag==False:
            flag =True # 开始录音
            tempnum2=tempnum
        if flag:
            if(temp < mindb and stat2==False): # 是否达到录音结束的条件
                stat2 = True
                tempnum2 = tempnum # 记录声音变小的节点
            if(temp > mindb): # 重新变大
                stat2 =False
                tempnum2 = tempnum # 记录重新变大的节点
                # 刷新tempnum2,继续录音
            if(tempnum > tempnum2 + delayTime*10 and stat2==True):
                #间隔0.4s后开始检测是否还是小声
                if(stat2 and temp < mindb):
                    stat = False
                    #停止录音
                else:
                    stat2 = False

        #print(str(temp)  +  "      " +  str(tempnum))
        tempnum = tempnum + 1
        if tempnum > 100:                #超时直接退出
            stat = False
    #print("录音结束")
    #stream.stop_stream()  # 停止输入流
    #stream.close()  # 关闭输入流
    #wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')  # 以’wb‘二进制流写的方式打开一个文件
    #wf.setnchannels(CHANNELS)  # 设置音轨数
    #wf.setsampwidth(p.get_sample_size(FORMAT))  # 设置采样点数据的格式，和FOMART保持一致
    #wf.setframerate(RATE)  # 设置采样率与RATE要一致
    #wf.writeframes(b''.join(frames))  # 将声音数据写入文件
    #wf.close()  # 数据流保存完，关闭文件
    audio_array = np.concatenate(audio_frames, axis=0)
    audio_mono = audio_array.flatten()
    # 归一化音频数据到[-1, 1]范围内
    normalized_audio = audio_mono / np.max(np.abs(audio_mono))

    audio_array = np.array(normalized_audio)
    # audio_tensor = torch.from_numpy(audio_array)
    # 打印转换后的数组形状
    # print(audio_array.shape)
    # print(audio_tensor.shape)
    result = model.transcribe(audio_array,language='Chinese',fp16 = True)
    s0 = result["text"]
    s2 = zhconv.convert(s0, 'zh-cn')
    #stream.start_stream()
    return s2

# 对收录的声音做处理
def recording(): 
    segment_id = 0 
    while True:
        s1 = chat_to_glm()    
        if s1 != "":
            print(s1)
            match_word = df.get_close_matches(s1, commandset_c, 3, cutoff=0.65)
            match_wake = df.get_close_matches(s1, wakeset_c, 3, cutoff=0.65)
            with open('./pause.txt', 'r') as f:
                match_stop = f.read().strip()
            
            if match_word:
                segment_id += 1
                res = match_word[0]
                match_index = commandset_c.index(res)
                #copyStr = str(segment_id) + " " + str(match_index + 1)
                #cb.copy(copyStr)
                print(str(segment_id) + ":" + commandset_c[match_index])
   
            elif s1 == match_stop :
                #print(s1)            
                global stop_flag
                stop_flag = True
                #process.kill()
                #process = None
                #subprocess.Popen(["pkill", "-f", "audio_play.py"])
                #with open('./stop.txt', 'w') as file:
                    #file.write("True")
            
            elif match_wake :
                #match_stop = ''
                #print(s1)
                audio_zaide = AudioSegment.from_file("./zaide.wav", format="wav")
                play(audio_zaide)
                #s3=''
                #while not match_stop :
                s3 = chat_to_glm()
                print(s3)
                #match_stop = df.get_close_matches(s3, wakeset_c, 3, cutoff=0.65)                    
                if s3 :
                    chat_messages[1]["content"] = s3
                    contents = create_chat_completion("chatglm3-6b", chat_messages)
                    print(contents)
                    sentences=split_sentences(contents)
                    for sentence in sentences:
                        audio_queue = queue.Queue()
                    # OpenVoice.demopart3.texts={ 'ZH': content }
                    # OpenVoice.demopart3.synthesis()
                    #textapi = str(content)
                    #payload = {"text": contents}
                        payload = {"text": sentence}
                    #headers = {"Content-Type":"application/json"}
                    
                        text_api_url = f"{text_url}/receive_text"
                        response = requests.post(text_api_url, json = payload)
                    #asyncio.get_event_loop().run_until_complete(client())
                    #data=json.dumps(payload),headers=headers
                    
                        if response.status_code == 200:
                            audio_np= np.asarray(json.loads(response.json()))
                            audio_queue.put(audio_np)
                            if audio_queue.qsize() != 0:
                                audio=audio_queue.get()
                                audio_play(audio)
                                #play_thread = threading.Thread(target=audio_play, args=(audio,))
                                #play_thread.start()
                                #time.sleep(1.5)
                        #process = subprocess.Popen(["python", "audio_play.py"], stdin=subprocess.PIPE)
                        #process.communicate(input=audio_str.encode())
                    else:
                        print("请求失败。")
                    
    #print("* done recording")  # 结束录音标志
'''
async def client():
    response = requests.get(f"{text_url}/receive_text", stream=True)
    audio_queue = queue.Queue()  # 创建音频片段队列
    for audio in response.iter_lines():
        if audio:
            print("Received data")
            audio_queue.put(audio)
    
        if audio_queue.qsize() == 1:
            await play_audio(audio_queue)
            
    #await asyncio.sleep(60)  # 持续监听 1 分钟

async def play_audio(audio_queue):
    while not audio_queue.empty():
        segment = audio_queue.get()
        audio_play(segment)
        audio_queue.task_done()
   '''




#分块前播放音频
def audio_play(audio:np.ndarray):

    CHUNK = 1024  # 缓冲区大小
    RATE = 44100  # 采样率

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True)

    duration = 1  # 每次播放的时长（秒）
    start_index = 0
    global stop_flag
    while start_index < len(audio) and not stop_flag:
        end_index = start_index + duration * RATE  # 计算结束索引
        segment = audio[start_index:end_index]  # 提取当前秒的音频段
        stream.write(segment.astype(np.float32).tobytes())  # 写入并播放音频段
        start_index = end_index  # 更新开始索引

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    if stop_flag:
        stop_flag = False

#和大模型通信
def create_chat_completion(model, messages):
    data = {
        "model": model, # 模型名称
        "messages": messages, # 会话历史
        "max_tokens": 500, # 最多生成字数
        "temperature": 0.8, # 温度
        "top_p": 0.8, # 采样概率
    }
    response = requests.post(f"{base_url}/v1/chat/completions", json=data)
    decoded_line = response.json()
    content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
    return content


if __name__ == '__main__':
    # 配置你的 ChatGLM 服务器地址和端口
    base_url = "http://localhost:8000" # 前面本地启动的API服务地址
    text_url = "http://127.0.0.1:8088"

    chat_messages = [
        {
            "role": "system",
            "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
        },
        {
            "role": "user",
            "content": "input_text"
        }
    ]

    devices = sd.query_devices()
    print(devices)
    # default_input_device_idx = sd.default.device[0]
    # print(f'Use default device: {devices[default_input_device_idx]["name"]}')
    input_device_idx = int(input("请选择您的麦克风设备（输入对应数字）:"))
    print(f'用户设备: {devices[input_device_idx]["name"]}')
    
    if torch.cuda.is_available():
        device_gpu =torch.device("cuda")
    else:
        device_gpu =torch.device("cpu")
        print("当前使用CPU，请更换GPU")

    model = whisper.load_model("medium",device=device_gpu)

    try:
        recording()
    except KeyboardInterrupt:
        print("* done recording")

