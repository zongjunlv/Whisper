#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/06/19 11:00:00
@Author  :   Anonymity
@Desc    :   拾音->发送音频流->接收回答音频流->播放音频并将对应的指令ID写入剪贴板
'''
import os
import time
import yaml
import pyaudio
import logging
import traceback
import sounddevice as sd
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from utils.common import recording
from get_root_path import get_root_path



def log_args(log_dir):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    today_date = datetime.now().strftime(r'%Y-%m-%d')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir,f'{today_date}.log')

    # args FileHandler to save log file
    fh = TimedRotatingFileHandler(log_file,when='midnight',interval=1,backupCount=7)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def delete_old_logs(log_dir):
    for file_name in os.listdir(log_dir):
        if file_name.endswith(".log"):
            file_path = os.path.join(log_dir,file_name)
            if os.path.isfile(file_path) and os.stat(file_path).st_mtime<(time.time()-7*24*60*60):
                os.remove(file_path)

if __name__ == '__main__':
    root_path = get_root_path()
    # 定义数据流块
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
    log_args(log_dir)
    delete_old_logs(log_dir)
    
    with open(os.path.join(root_path, 'assets/value.yml'), 'r', encoding='utf-8') as stream:
        try:
            value = yaml.safe_load(stream)
            temp = value['temp']
            CHUNK = value['CHUNK']
            CHANNELS = value['CHANNELS']
            RATE = value['RATE']
            mindb = value['mindb']
            delayTime = value['delayTime']
            max_tokens = value['max_tokens']
            temperature = value['temperature']
            top_p = value['top_p']
            # 使用变量的代码
            logging.info(f"RATE: {RATE}, mindb: {mindb}")
        except yaml.YAMLError as ex:
            logging.error(f"YAML Error:{ex}")
            logging.error(traceback.format_exc())
    #  读取自定义prompt块
    with open(os.path.join(root_path, 'assets/chat_messages.yml'), 'r', encoding="utf-8") as stream:
        try:
            chat_messages = yaml.safe_load(stream)
        except yaml.YAMLError as ex:
            logging.error(f"YAML Error:{ex}")
            logging.error(traceback.format_exc())
    '''        
    temp = 20
    CHUNK = 1024  # 音频帧率（也就是每次读取的数据是多少，默认1024）
    FORMAT = pyaudio.paInt16  # 采样时生成wav文件正常格式
    CHANNELS = 1  # 音轨数（每条音轨定义了该条音轨的属性,如音轨的音色、音色库、通道数、输入/输出端口、音量等。可以多个音轨，不唯一）
    RATE = 16000  # 采样率（即每秒采样多少数据）
    mindb = 600 # 最小声音，大于则开始录音，否则结束
    delayTime = 0.4
    '''
    FORMAT = pyaudio.paInt16  # 采样时生成wav文件正常格式
    p = pyaudio.PyAudio()  # 创建PyAudio对象
    stream = p.open(format=FORMAT,  # 采样生成wav文件的正常格式
                    channels=CHANNELS,  # 音轨数
                    rate=RATE,  # 采样率
                    input=True,  # Ture代表这是一条输入流，False代表这不是输入流
                    frames_per_buffer=CHUNK)  # 每个缓冲多少帧

    # 打印并选择当前音频设备
    # 打印并选择当前音频设备
    devices = sd.query_devices()
    defalt_device = sd.default.device
    print(devices)
    #input_device_idx = int(input("请选择您的麦克风设备（输入对应数字）:"))
    print(f'用户设备: {defalt_device}')

    server_url = "10.10.113.121"
    
    try:
        recording(
            root_path=root_path,
            base_url="http://127.0.0.1:11434", #http://127.0.0.1:8000
            text_url = "http://127.0.0.1:8088",
            whis_url = "http://127.0.0.1:8090",
            stream=stream,
            mindb=mindb,
            delayTime=delayTime,
            CHUNK=CHUNK,
            chat_messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
    except KeyboardInterrupt:
        print("* done recording")
    except Exception as e:
        logging.error(f"Error during initialization:{e}")
        logging.error(traceback.format_exc())
