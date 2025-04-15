import os
import time
import json
import argparse
import warnings
from typing import Optional
warnings.filterwarnings("ignore")

import yaml
import torch
import uvicorn
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from demopart3 import synthesis_toncvtr, load_model
from logging.handlers import TimedRotatingFileHandler

from openvoice import se_extractor
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
    parser.add_argument("--cc", "--ckpt_converter", type=str, default="checkpoints_v2/converter", help="音色抽取和合成模型的权重文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"],help="模型推理时使用的硬件设备")
    parser.add_argument("--output_dir", type=str, default="outputs_v2", help="模型生成的语音文件的保存路径")

    return parser.parse_args()

app = FastAPI()

class TextData(BaseModel):
    text: str


@app.post('/receive_text')
async def receive_text(text: TextData):
    # 指定保存文件的路径
    save_path = "text_file"
    data = text.text
    start = time.perf_counter()

    res_audio = synthesis_toncvtr(
        model=tts_model,
        tone_color_converter=tone_color_converter,
        source_se=source_se,
        target_se=target_se,
        texts=data
    )

    end = time.perf_counter()
    print("Time: {0}".format(end-start))

    return json.dumps(res_audio.tolist())


if __name__ == '__main__':
    args = get_parser()
    
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
    # 记录正常的 print 信息
    stdout = log_args(log_dir)
    # 记录 traceback 异常信息
    stderr = log_args(log_dir)
    delete_old_logs(log_dir)


    # 项目根目录
    root_path = get_root_path() if args.root_path is None else args.root_path

    device = args.device

    print("加载模型中......")
    wav_relative_p = "./models--M4869--WavMark/snapshots/0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef/step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl"
    tone_color_converter, tts_model = load_model(
        os.path.join(root_path, args.cc), device=device,
        local_dir=os.path.join(root_path, ".cache"), wav_dir=os.path.join(root_path, ".cache", wav_relative_p),
    )
    print("成功加载.")

    with open(os.path.join(root_path, '.cache/speakerpath.yaml'), 'r') as yaml_file:
        speakerpath = yaml.safe_load(yaml_file)

    reference_speaker =speakerpath['Paths']['reference_speaker']  # This is the voice you want to clone
    target_se, audio_name = se_extractor.get_se(os.path.join(root_path, reference_speaker), tone_color_converter, vad=False, cache_dir=os.path.join(root_path, ".cache"))
    source_se = torch.load(os.path.join(get_root_path(), f'checkpoints_v2/base_speakers/ses/zh.pth'), map_location=device)
    print("Real voice loaded.")

    synthesis_toncvtr(
        model=tts_model,
        tone_color_converter=tone_color_converter,
        source_se=source_se,
        target_se=target_se,
        texts="hello.",
    )
    print("TTS model loaded.")

    uvicorn.run(app, host='0.0.0.0', port=8088)