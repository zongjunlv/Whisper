#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   whisper_api.py
@Time    :   2024/06/19 13:41:14
@Author  :   None
@Desc    :   None
'''
import os
import sys
import time
import uvicorn
import argparse
import logging
from typing import Optional
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import torch
import zhconv
import numpy as np
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile

import whisper
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

def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=Optional[str], default=None, help="项目根目录的绝对路径")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"],help="模型推理时使用的硬件设备")

    return parser.parse_args()


class AudioData(BaseModel):
    audio_steam: list
    language: str = "Chinese"

app = FastAPI()
@app.post('/receive_audio_strming')
async def receive_audio_strming(
    audata: AudioData,
):
    audio_array = np.array(audata.audio_steam, dtype=np.float32)
    language = audata.language
    result = model.transcribe(audio_array,language=language, fp16=True)
    s0 = result["text"]
    s2: str = zhconv.convert(s0, 'zh-cn')
    return {"converted_text": s2}


if __name__ == "__main__":
    args = get_parser()

    root_path = get_root_path() if args.root_path is None else args.root_path

    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
    # 记录正常的 print 信息
    stdout = log_args(log_dir)
    # 记录 traceback 异常信息
    stderr = log_args(log_dir)
    delete_old_logs(log_dir)

    if torch.cuda.is_available():
        device =torch.device(args.device)
        print("当前推理设备为: GPU")
    else:
        device =torch.device(args.device)
        print("当前推理设备为: CPU")
    model = whisper.load_model("medium",device=device, download_root=os.path.join(root_path, "weights"))
    print("Whisper model loaded.")

    uvicorn.run(app, host='0.0.0.0', port=8090)
