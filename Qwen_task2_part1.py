from pathlib import Path
import librosa
import numpy as np
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# 1. 加载模型和 processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

# 2. 设定音频文件路径
audio_folder = Path("./ESC-50-master/audio")
audio_files = list(audio_folder.glob("*.wav"))

if not audio_files:
    print("未找到任何 .wav 文件！")
    exit()

# 3. 设定特征存储路径
output_folder = Path("./new_Feature")
output_folder.mkdir(parents=True, exist_ok=True)

# 4. 生成对话内容（注意这里构造了包含所有音频的文本）
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "<|AUDIO|>",
                "local_path": str(audio_file)
            } for audio_file in audio_files
        ]
    }
]

# 5. 生成带 <|AUDIO|> 占位符的文本
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

# 6. 加载音频
def load_local_audio(path: str) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"音频文件 {path} 不存在或为空")

    audio, sr = librosa.load(
        path,
        sr=processor.feature_extractor.sampling_rate,
        mono=True
    )
    return audio

# 7. 读取所有音频
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele.get("type") == "audio":
                try:
                    audio_data = load_local_audio(ele["local_path"])
                    audios.append(audio_data)
                    print(f"成功加载：{ele['local_path']}")
                except Exception as e:
                    print(f"加载失败：{ele['local_path']}，错误：{str(e)}")
                    audios.append(np.zeros(1))  # 失败时填充零数组

# 8. 遍历每个音频，逐个处理
for i, audio in enumerate(audios):
    # 更新 text，仅包含当前音频的 <|AUDIO|> 占位符
    single_text = processor.apply_chat_template(
        conversation=[{
            "role": "user",
            "content": [{
                "type": "audio",
                "audio_url": "<|AUDIO|>",
                "local_path": str(audio_files[i])
            }]
        }],
        tokenize=False,
        add_generation_prompt=True
    )

    # 使用 processor 处理当前音频和更新后的文本
    single_input = processor(text=single_text, audios=[audio], return_tensors="pt", padding=True)
    single_input = {k: v.to(next(model.parameters()).device) for k, v in single_input.items()}  # 迁移到同一设备

    model.eval()
    with torch.no_grad():
        # 获取模型输出
        outputs = model(**single_input, output_hidden_states=True)

        # 获取最后一层 hidden state
        hidden_states = outputs.hidden_states[-1].cpu().numpy()

        # 保存特征
        original_name = Path(audio_files[i]).stem  # 获取音频文件名
        output_path = output_folder / f"{original_name}.npy"
        np.save(output_path, hidden_states)
        print(f"已保存特征文件：{output_path}，维度：{hidden_states.shape}")
