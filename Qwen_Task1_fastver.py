from pathlib import Path
import librosa
import csv
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioCSVGenerator:
    def __init__(self,
                 model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
                 output_csv: str = "audio_results.csv",
                 batch_size: int = 4,
                 max_audio_length: float = 30.0):
        """
        初始化音频处理管道
        Args:
            model_name: 模型名称或路径
            output_csv: 输出CSV文件路径
            batch_size: 批处理大小（根据GPU显存调整）
            max_audio_length: 最大音频长度（秒）
        """
        try:
            # 自动选择设备和精度
            if torch.cuda.is_available():
                self.device = "cuda"
                self.dtype = torch.float16
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.dtype = torch.float16
            else:
                self.device = "cpu"
                self.dtype = torch.float32

            # 加载模型
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True
            ).eval()

            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # 配置参数
            self.output_csv = Path(output_csv)
            self.batch_size = batch_size
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
            self.max_samples = int(max_audio_length * self.sampling_rate)

            logger.info(f"✅ 初始化完成 | 设备: {self.device} | 精度: {self.dtype}")

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    def _process_audio_batch(self, audio_paths: list) -> list:
        """处理音频批次并返回响应"""
        try:
            # 1. 加载并预处理音频
            audios = []
            for path in audio_paths:
                audio, _ = librosa.load(
                    path,
                    sr=self.sampling_rate,
                    duration=self.max_samples / self.sampling_rate
                )
                audios.append(audio)

            # 2. 构建对话模板
            conversations = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "audio", "audio_path": str(path)},
                        {"type": "text", "text": "<|AUDIO|> Please describe this sound."},
                    ]}
                ] for path in audio_paths
            ]

            # 3. 处理输入
            inputs = self.processor(
                text=[self.processor.apply_chat_template(c, tokenize=False) for c in conversations],
                audio=audios,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_samples,
                sampling_rate=self.sampling_rate
            ).to(self.model.device)

            # 4. 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=3,
                    temperature=0.7
                )

            # 5. 解码结果
            return self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

        except Exception as e:
            logger.error(f"处理批次失败: {str(e)}")
            return ["Error"] * len(audio_paths)

    def process_folder(self, input_folder: str):
        """处理整个音频文件夹"""
        input_path = Path(input_folder)
        audio_paths = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3"))

        # 准备CSV写入
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Description"])

            # 进度条
            pbar = tqdm(total=len(audio_paths), desc="Processing Audio")

            # 分批处理
            for i in range(0, len(audio_paths), self.batch_size):
                batch_paths = audio_paths[i:i + self.batch_size]

                try:
                    responses = self._process_audio_batch(batch_paths)

                    # 写入结果
                    for path, resp in zip(batch_paths, responses):
                        writer.writerow([path.name, resp])
                        pbar.update(1)
                        pbar.set_postfix_str(f"Processing: {path.name}")

                    # 立即写入磁盘
                    f.flush()

                except Exception as e:
                    logger.error(f"跳过批次 {i // self.batch_size}：{str(e)}")
                    continue


if __name__ == "__main__":
    # 使用示例
    processor = AudioCSVGenerator(
        output_csv="./audio_results.csv",
        batch_size=10,  # RTX 3090推荐值
        max_audio_length=30  # 截断长音频
    )

    processor.process_folder("./ESC-50-master/audio")