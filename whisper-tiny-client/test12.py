import pyaudio
import numpy as np
import difflib as df
from requests.exceptions import ReadTimeout


def chat_to_glm(
):
    """
    录音函数
    """
    temp = 20
    CHUNK = 1024
    CHANNELS = 1
    RATE = 16000
    mindb = 900
    delayTime = 0.4
    frames = []  # 定义frames为一个空列表
    audio_frames = []
    flag = False # 开始录音的节点
    stat = True
    stat2 = False
    tempnum = 0
    tempnum2 = 0

    FORMAT = pyaudio.paInt16  # 采样时生成wav文件正常格式
    p = pyaudio.PyAudio()  # 创建PyAudio对象
    stream = p.open(format=FORMAT,  # 采样生成wav文件的正常格式
                    channels=CHANNELS,  # 音轨数
                    rate=RATE,  # 采样率
                    input=True,  # Ture代表这是一条输入流，False代表这不是输入流
                    frames_per_buffer=CHUNK)  # 每个缓冲多少帧
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


if __name__ == "__main__":
    chat_to_glm()