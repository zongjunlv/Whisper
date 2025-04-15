#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   common.py
@Time    :   2024/06/19 11:11:31
@Author  :   Meantao
@Desc    :   None
'''
import os
import re
import sys
import json
import time
import wave
import queue
import asyncio
import pyaudio
import requests
import threading
import pyperclip

import numpy as np
import difflib as df
from pypinyin import pinyin
from pydub import AudioSegment
from pydub.playback import play
from requests.exceptions import ReadTimeout
from ollama import Client

audio_thread = None
play_thread = None
output_thread = None
stop_flag = False
empty_q = True
stop_tts = False
lock = threading.Lock()  # 锁，用于线程同步
audio_queue = queue.Queue()
segment_id = 0
prompt = ""


def chat_to_glm(
        stream,
        mindb,
        delayTime,
        CHUNK,
):
    """
    录音函数
    """
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
        temp = np.max(audio_data) # 计算音频的峰值
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
                # 间隔0.4s后开始检测是否还是小声
                if(stat2 and temp < mindb):
                    # 停止录音
                    stat = False
                else:
                    stat2 = False
        tempnum = tempnum + 1
        if tempnum > 100:  # 超时直接退出
            stat = False

    audio_array = np.concatenate(audio_frames, axis=0)
    audio_mono = audio_array.flatten()
    # 归一化音频数据到[-1, 1]范围内
    normalized_audio = audio_mono / np.max(np.abs(audio_mono))
    audio_array = np.array(normalized_audio)

    return audio_array.tolist()

'''
def create_chat_completion(model, messages, base_url,max_tokens=800,
                           temperature=0.3, top_p=0.8):
    """
    与大模型通信
    """
    data = {
        "model": model, # 模型名称
        "messages": messages, # 会话历史
        "max_tokens": max_tokens, # 最多生成字数
        "temperature": temperature, # 温度
        "top_p": top_p, # 采样概率
        "stream": True
    }
    sentence = ""
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=True)
        for line in response.iter_lines():
            if line and "DONE" not in line.decode('utf-8'):
                tmp_line = bytearray(line)
                tmp_line = tmp_line.decode("utf-8")
                tmp_line = tmp_line[len("data: "):]
                decoded_line = json.loads(tmp_line)
                content = decoded_line["choices"][0]["delta"]["content"]
                sentence += content
                if content in r"\.\！\？\n\。\?\!":  # 检查是否为句号、问号或感叹号
                    complete_sentence = sentence.strip()
                    sentence = ""  # 重置句子
                    # print(complete_sentence)
                    yield complete_sentence
        if sentence != "":
            yield sentence
    except Exception as e:
        raise e


def split_sentences(text):
    """
    (deprecated)
    切分句子函数
    """
    sentences = re.split(r"\.|\！|\？|\n|\。|\?|\!", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()!='']
    return sentences
'''


def stream_generate_text(prompt, model="deepseek-r1:8b", host="http://localhost:11434"):
    url = f"{host}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
       
    }
    response = requests.post(url, json=data, stream=True)
    sentence = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if "response" in decoded_line:
                content = json.loads(decoded_line)["response"]

                if "think"  in content:  # 过滤包含 <think> 标签的内容
                    continue
                sentence += content
                if content.endswith((".", "。", "！", "!", "？", "?", "\n")):
                    complete_sentence = sentence.strip()
                    sentence = ""  # 重置句子
                    print(complete_sentence)
                    yield complete_sentence
    if sentence != "":
            yield sentence
            #print(content, end="", flush=True)


def play_audio(text_url,prompt):
    """
    1. 向大模型发送请求，并且得到字符流;
    2. 分别按序播放每句话.
    """
    global empty_q, segment_id, stop_tts, audio_thread
    empty_q = True

    stop_tts = False

    while not audio_queue.empty():
        audio_queue.get()
    audio_thread = threading.Thread(target=play_audio_data)  # 创建音频播放线程
    audio_thread.daemon = True
    audio_thread.start()

    play_flag = False
    
    for sentence in stream_generate_text(prompt, model="deepseek-r1:8b", host="http://localhost:11434"):
        if sentence == "":
            continue
        if stop_tts:
            print("stop_tts = True")
            break
        if not play_flag:
            segment_id += 1
            copyStr = str(segment_id) + " 100" 
            if sys.platform != "linux":
                pyperclip.copy(copyStr)
            print(copyStr)
            play_flag = True
        print(sentence)
        payload = {"text": sentence}
        text_api_url = f"{text_url}/receive_text"
        try:
            response = requests.post(text_api_url, json = payload)
        except Exception as e:
            # 打印异常信息
            raise e

        if response.status_code == 200:
            audio_np= np.asarray(json.loads(response.json()))
            audio_queue.put(audio_np)
            empty_q = False
        else:
            print("请求失败。")


def play_audio_data(CHUNK: int = 1024, RATE: int = 22050):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True)
    global stop_flag,empty_q
    global segment_id
    
    while (not audio_queue.empty() and not stop_flag) or  empty_q :
            audio_data = audio_queue.get()
            start_index = 0
            duration = 1  # 每次播放的时长（秒）
            
            while start_index < len(audio_data) and not stop_flag :
                end_index = int(start_index + duration * RATE)  # 计算结束索引
                end_index = min(end_index,len(audio_data))
                segment = audio_data[start_index:end_index]  # 提取当前秒的音频段
                stream.write(segment.astype(np.float32).tobytes())  # 写入并播放音频段
                start_index = end_index  # 更新开始索引
            audio_queue.task_done()
    segment_id += 1
    copyStr = str(segment_id) + " 101" 
    if sys.platform != "linux":
        pyperclip.copy(copyStr)
    print(copyStr)
    stream.stop_stream()
    stream.close()
    p.terminate()
    if stop_flag:
        stop_flag = False
        print("stopflag = false")


def audio2chars(
        s1_middle: list,
        whis_url: str,
        api: str = "receive_audio_strming"
) -> str:
    """
    将音频流发送至服务器，获取其对应文字。
    :param: s1_middle: 音频流 (转为了list的numpy数组)
    :param: whis_url: 服务器url
    :param: api: 服务器上对应的接口名
    :reutrn: 转换后的简体中文
    """
    payload = {"audio_steam": s1_middle}
    try:
        response_whis = requests.post(f"{whis_url}/{api}", json = payload)
    except ReadTimeout:
        #如果请求超时报错，返回空
        return ""

    except Exception as e:
        # 打印异常信息
        raise e
    
    if response_whis.status_code == 200:
        s1 = json.loads(response_whis.text)["converted_text"]
    else:
        raise ValueError(f"请求失败！状态码: {response_whis.status_code}")

    return s1


def output_audio(root_path,answer_index):
    chunk = 1024
    global stop_flag,segment_id
    audiofile_path = os.path.join(root_path, f"assets/output_audio/output_audio_{answer_index + 1}.wav")
    audio_play = AudioSegment.from_file(audiofile_path, format="wav")
    wf = wave.open(audiofile_path, 'rb')
    
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    
    # 打开流
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    # 读取和播放音频数据
    data = wf.readframes(chunk)
    segment_id += 1
    copyStr = str(segment_id) + " 100" 
    if sys.platform != "linux":
        pyperclip.copy(copyStr)
    print(copyStr)

    while data:
        if stop_flag :
            break
        stream.write(data)
        data = wf.readframes(chunk)
        
    if stop_flag:
        stop_flag = False
    segment_id += 1
    copyStr = str(segment_id) + " 101" 
    if sys.platform != "linux":
        pyperclip.copy(copyStr)
    print(copyStr)

    # 结束播放
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    

def recording(
        root_path, base_url, text_url, whis_url, stream, mindb, delayTime, CHUNK, 
        chat_messages, max_tokens, temperature, top_p
):
    #读取行动指令文件
    f_command = open(os.path.join(root_path, "assets/command.txt"), "r", encoding='utf-8')
    command_lists = f_command.readlines()
    commandset_e = []
    commandset_c = []
    for command in command_lists:
        data1 = command.strip('\n')  # 去掉开头和结尾的换行符
        data2 = data1.split("_")
        commandset_c.append(data2[0])
        commandset_e.append(data2[1])

    #读取指定问题文件
    f_answer = open(os.path.join(root_path, "assets/answer.txt"), "r", encoding='utf-8')
    answer_lists = f_answer.readlines()
    answer_e = []
    answer_c = []
    for answer in answer_lists:
        answerdata1 = answer.strip('\n')  # 去掉开头和结尾的换行符
        answerdata2 = answerdata1.split("_")
        answer_c.append(answerdata2[0])
        answer_e.append(answerdata2[1])

    #读取唤醒文件   
    f_wake = open(os.path.join(root_path, "assets/wake.txt"), 'r', encoding='utf-8')
    wake_lists = f_wake.readlines()
    wakeset_e = []
    wakeset_c = []
    for wake in wake_lists:
        wakedata1 = wake.strip('\n')  # 去掉开头和结尾的换行符
        wakedata2 = wakedata1.split("_")
        wakeset_c.append(wakedata2[0])
        wakeset_e.append(wakedata2[1])

    #读取停止文件
    with open(os.path.join(root_path, "assets/pause.txt"), 'r', encoding='utf-8') as f:
        match_stop = f.read().strip()

    while True:
        # 获得音频流
        s1_middle = chat_to_glm(
            stream,
            mindb,
            delayTime,
            CHUNK,
        )
        # 发送至whisper服务，获取中文文字
        s1 = audio2chars(s1_middle, whis_url)
        global segment_id, stop_flag,output_thread,play_thread,stop_tts,audio_thread
        # 文字转语音流程
        if s1 != "":
            print(s1)
            resultArray = pinyin(s1, 0)
            result_pinyin = ''
            for r in resultArray:
                result_pinyin = ''.join([result_pinyin, r[0]])
            
            match_word = df.get_close_matches(result_pinyin, commandset_e, 3, cutoff=0.80)
            match_wake = df.get_close_matches(result_pinyin, wakeset_e, 3, cutoff=0.80)
            match_answer = df.get_close_matches(result_pinyin, answer_e, 3, cutoff=0.80)
            
            if match_word:
                segment_id += 1
                res1 = match_word[0]
                match_index = commandset_e.index(res1)
                copyStr = str(segment_id) + " " + str(match_index + 1)
                if sys.platform != "linux":
                    pyperclip.copy(copyStr)
                print(str(segment_id) + ":" + commandset_c[match_index])


            elif match_stop in s1:
                if (output_thread != None and output_thread.is_alive()) or (play_thread != None and play_thread.is_alive()) or (audio_thread != None and audio_thread.is_alive()):
                    stop_flag = True
                    stop_tts = True 
            
            elif match_answer:
                segment_id += 1
                res2 = match_answer[0]
                answer_index = answer_e.index(res2)
                print(str(segment_id) + ":" + answer_c[answer_index])
                if (output_thread != None and output_thread.is_alive()) or (play_thread != None and play_thread.is_alive()) or (audio_thread != None and audio_thread.is_alive()):
                    stop_flag = True
                    stop_tts = True   

                time.sleep(0.2)
                output_thread = threading.Thread(target=output_audio,args=(root_path, answer_index))
                output_thread.daemon = True
                output_thread.start()

            elif match_wake :
                if (output_thread != None and output_thread.is_alive()) or (play_thread != None and play_thread.is_alive()) or (audio_thread != None and audio_thread.is_alive()):
                    stop_flag = True
                    stop_tts = True
                
                time.sleep(0.2)
                audio_zaide = AudioSegment.from_file(os.path.join(root_path, "assets/zaide.wav"), format="wav")
                play(audio_zaide)
                s3_middle = chat_to_glm(
                    stream,
                    mindb,
                    delayTime,
                    CHUNK,
                )
                s3 = audio2chars(s3_middle, whis_url)
                print(s3)
                
                if s3 :
                    global prompt
                    prompt = s3
                    '''
                    chat_messages[1]["content"] = s3
                    client = Client(
                    host='http://127.0.0.1:11434',
                    )
                    response = client.chat(model='deepseek-r1:8b', stream=True,messages=[
                         {           
                            'role': 'system',
                            'content': '你是一个乐于助人的助手',
                        },
                        {
                             'role': 'user',
                            'content': chat_messages,
                        },
                    ])

                    for chunk in response:
                        #print(chunk['message']['content'],end='',flush=True)

                    '''
                    play_thread = threading.Thread(target=play_audio,args=(text_url,prompt))
                    play_thread.daemon = True
                    play_thread.start()
                    