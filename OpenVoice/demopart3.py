import os
import re
import time
import argparse
from typing import Optional, Any

import torch
import torch.nn as nn
from MeloTTS.melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter


def readtxt():
    with open(os.path.join('text_file/receive_text.txt'), 'r', encoding='utf-8') as f:
        content = f.read()

    str_txt=str(content)

    return str_txt


def split_sentences(text):
    sentences = re.split(r"\.|\！|\？|\n|\,|\。|\?|\!|\，", text)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def load_model(
    ckpt_converter: str,
    device: torch.device,
    language: str = "ZH",
    local_dir=None,
    wav_dir=None,
):
    """
    加载模型: 1. 音色抽取和合成模型; 2. 文字转语音模型
    转音色的功能已注释，如需要则取消注释
    :param: ckpt_converter: 音色抽取和合成模型的权重文件
    :param: device: 模型推理时使用的硬件设备
    :param: language: 设定合成音频时的语言
    """
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device, wav_dir=wav_dir)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    tts_model = TTS(language, device=device, local_dir=local_dir, use_hf=False)

    return tone_color_converter, tts_model


def synthesis_without_toncvtr(
    model: Optional[torch.nn.Module],
    texts: str,
    speed: float = 1.12
):
    # for language, text in texts.items():
    speaker_ids = model.hps.data.spk2id
    
    assert len(speaker_ids.keys()) == 1, \
        "Now we only translate zh! But keys contain: {0}".format(speaker_ids.keys()) 
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')

    audio = model.tts_to_file(texts, list(speaker_ids.values())[0], speed=speed)

    return audio


def synthesis_toncvtr(
    model: Optional[TTS],
    tone_color_converter: Optional[ToneColorConverter],
    source_se: Any, 
    target_se: Any,
    texts: str,
    speed: float = 1.12
):
    speaker_ids = model.hps.data.spk2id

    audio = model.tts_to_file(texts, list(speaker_ids.values())[0], speed=speed)

    # Run the tone color converter
    encode_message = "@MyShell"
    audio = tone_color_converter.convert(
        audio_src_path=audio,
        src_se=source_se,
        tgt_se=target_se,
        message=encode_message,
        orig_sr=model.hps.data.sampling_rate,
        # output_path=r"F:\OpenVoice\outputs_v2\测试.wav"
    )

    return audio


if __name__ == '__main__':
    # 项目根目录 
    # root_path = get_root_path() if args.root_path is None else args.root_path

    # device = args.device

    # ckpt_converter = os.path.join(root_path, "checkpoints_v2/converter")
    # output_dir = 'outputs_v2'

    # tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device)
    # tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    # _, tts_model = load_model(
    #     os.path.join(root_path, args.cc), device=device
    # )
    # print("TTS model loaded.")

    # os.makedirs(os.path.join(root_path, args.output_dir), exist_ok=True)

    # reference_speaker = os.path.join(root_path, "resources/example_reference.mp3") # This is the voice you want to clone
    # target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

    # source_se = torch.load(os.path.join(get_root_path(), f'checkpoints_v2/base_speakers/ses/zh.pth'), map_location=device)
    # print("Real voice loaded.")

    # output_text=str() # 区分output.txt历史和现在
    # # str_txt=str()#全局变量，给zh赋值
    # src_path = f'{args.output_dir}/tmp.wav'
    # speed = 1.12

    # 原代码可以合成不同语言的语音
    # texts = {
    #     'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    #     'EN': "Did you ever hear a folk tale about a giant turtle?",
    #     'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    #     'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    #     'ZH':  ""
    #     'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    #     'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
    # }

    # uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    # audio_url = "http://10.10.115.11:8017"
    #while(1):
    # textread = readtxt()
        #print(textread)
    # if(textread!=''):
    #         #if(textread != output_text):
    #             #sentences = split_sentences(textread)
    #             #for sentence in sentences: 
    #              #print(sentence)
    #     start = time.perf_counter()
    #     synthesis(
    #         model=tts_model,
    #         texts=textread,
    #     )
    #     end = time.perf_counter()
    #     print(end-start)
                #simpleaudio.stop_all()
                #audio_filename = 'outputs_v2/tmp.wav'
                 # 调用音频处理API
                #audio_api_url = f"{audio_url}/audio"
                #files = {"audio": open(audio_filename, "rb")}  
                #response = requests.post(audio_api_url, files=files)
                #output_text=textread
    
    #with open('./receive_text.txt', 'w') as file:
                    #file.write("")
    pass
