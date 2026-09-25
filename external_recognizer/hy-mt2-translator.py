#!/usr/bin/env python3
#
# Copyright (c) 2025 Xiaomi Corporation

"""
Hy-MT2-1.8B Translator for TMSpeech
Combines FunASR-Nano (ASR) + Hy-MT2-1.8B (Translation)
Outputs results in TMSpeech CommandRecognizer protocol format.
"""

import argparse
import multiprocessing
import queue
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_audio_utils import (  # noqa: E402
    MyPrinter,
    cleanup_recording_process,
    get_audio_devices,
    np,
    pyaudio,
    sample_rate,
    select_input_device,
    sherpa_onnx,
    start_recording,
)


# ============================================================
# Translation Backend Selection
# ============================================================

def create_translator(args):
    """
    Create translator instance based on the selected backend.
    Supports: transformers (default), gguf (llama.cpp)
    """
    if args.translator_backend == "gguf":
        return create_gguf_translator(args)
    else:
        return create_transformers_translator(args)


def create_transformers_translator(args):
    """Create Hy-MT2 translator using HuggingFace transformers."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print(
            "缺少 transformers 或 torch 依赖。\n"
            "请运行: pip install transformers torch\n"
            "或使用 install-hymt2.bat 安装",
            file=sys.stderr,
        )
        sys.exit(-1)

    model_path = args.hy_mt2_model
    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"Hy-MT2 模型目录不存在: {model_path}\n"
            f"请从 https://huggingface.co/tencent/Hy-MT2-1.8B 下载模型"
        )

    print(f"正在加载 Hy-MT2 模型: {model_path}", file=sys.stderr)
    print(f"推理引擎: transformers, 设备: {args.device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device if args.device == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    return TransformersTranslator(model, tokenizer, args)


def create_gguf_translator(args):
    """Create Hy-MT2 translator using llama.cpp GGUF."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "缺少 llama-cpp-python 依赖。\n"
            "请运行: pip install llama-cpp-python\n"
            "或使用 install-hymt2.bat 安装",
            file=sys.stderr,
        )
        sys.exit(-1)

    model_path = args.hy_mt2_model
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Hy-MT2 GGUF 模型文件不存在: {model_path}\n"
            f"请从 https://huggingface.co/tencent/Hy-MT2-1.8B-1.25bit-GGUF 下载"
        )

    print(f"正在加载 Hy-MT2 GGUF 模型: {model_path}", file=sys.stderr)
    print(f"推理引擎: llama.cpp, 线程数: {args.num_threads}", file=sys.stderr)

    n_gpu_layers = -1 if args.device == "cuda" else 0

    llm = Llama(
        model_path=str(model_path),
        n_ctx=8192,
        n_threads=args.num_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return GGUFTranslator(llm, args)


# ============================================================
# Translator Implementations
# ============================================================

class TransformersTranslator:
    """Hy-MT2 translator using HuggingFace transformers."""

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.source_lang_name = args.source_lang_name
        self.target_lang_name = args.target_lang_name
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def translate(self, text: str) -> str:
        """Translate text from source language to target language."""
        if not text or not text.strip():
            return text

        # Build prompt according to Hy-MT2 instruction format
        prompt = (
            f"将以下文本翻译为 {self.target_lang_name}，注意"
            f"只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        )

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False
        )

        input_ids = self.tokenizer.encode(inputs, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
            )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return result.strip()


class GGUFTranslator:
    """Hy-MT2 translator using llama.cpp GGUF."""

    def __init__(self, llm, args):
        self.llm = llm
        self.args = args
        self.source_lang_name = args.source_lang_name
        self.target_lang_name = args.target_lang_name
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def translate(self, text: str) -> str:
        """Translate text from source language to target language."""
        if not text or not text.strip():
            return text

        prompt = (
            f"将以下文本翻译为 {self.target_lang_name}，注意"
            f"只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        )

        output = self.llm(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repetition_penalty,
            stop=["\n\n"],  # Stop at double newline
        )

        result = output["choices"][0]["text"].strip()
        return result


# ============================================================
# Main Recognition Pipeline
# ============================================================

def validate_args(args):
    """Validate command line arguments and model paths."""
    # Check ASR model paths
    required_asr_files = (
        args.silero_vad_model,
        args.encoder_adaptor,
        args.llm,
        args.embedding,
    )
    missing = [str(p) for p in required_asr_files if not Path(p).is_file()]
    if not Path(args.tokenizer).is_dir():
        missing.append(str(args.tokenizer))

    if missing:
        details = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "缺少 Fun-ASR-Nano 运行文件:\n"
            f"{details}\n"
            f"Nano 模型: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2\n"
            f"Silero VAD: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
        )

    # Check translator model path
    if args.translator_backend == "gguf":
        if not Path(args.hy_mt2_model).is_file():
            raise FileNotFoundError(
                f"Hy-MT2 GGUF 模型文件不存在: {args.hy_mt2_model}\n"
                f"请从 https://huggingface.co/tencent/Hy-MT2-1.8B-1.25bit-GGUF 下载"
            )
    else:
        if not Path(args.hy_mt2_model).is_dir():
            raise FileNotFoundError(
                f"Hy-MT2 模型目录不存在: {args.hy_mt2_model}\n"
                f"请从 https://huggingface.co/tencent/Hy-MT2-1.8B 下载模型"
            )

    if args.num_threads <= 0:
        raise ValueError("--num-threads 必须大于 0")


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Hy-MT2 Translator: ASR + Translation for TMSpeech"
    )

    # --- Audio / ASR arguments ---
    parser.add_argument("--device", type=int, default=-1, help="输入设备编号")
    parser.add_argument(
        "--silero-vad-model",
        default=str(SCRIPT_DIR / "silero_vad.onnx"),
        help="silero_vad.onnx 文件路径",
    )
    parser.add_argument(
        "--encoder-adaptor",
        default=str(SCRIPT_DIR / "sherpa-onnx-funasr-nano-int8-2025-12-30" / "encoder_adaptor.int8.onnx"),
        help="Fun-ASR-Nano encoder adaptor ONNX 文件",
    )
    parser.add_argument(
        "--llm",
        default=str(SCRIPT_DIR / "sherpa-onnx-funasr-nano-int8-2025-12-30" / "llm.int8.onnx"),
        help="Fun-ASR-Nano LLM ONNX 文件",
    )
    parser.add_argument(
        "--embedding",
        default=str(SCRIPT_DIR / "sherpa-onnx-funasr-nano-int8-2025-12-30" / "embedding.int8.onnx"),
        help="Fun-ASR-Nano embedding ONNX 文件",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(SCRIPT_DIR / "sherpa-onnx-funasr-nano-int8-2025-12-30" / "Qwen3-0.6B"),
        help="Qwen3 tokenizer 目录",
    )
    parser.add_argument("--num-threads", type=int, default=4, help="ASR 推理线程数")

    # --- Translator arguments ---
    parser.add_argument(
        "--translator-backend",
        choices=("transformers", "gguf"),
        default="transformers",
        help="翻译推理引擎",
    )
    parser.add_argument(
        "--hy-mt2-model",
        default=str(SCRIPT_DIR / "hy-mt2-1.8b"),
        help="Hy-MT2 模型路径 (transformers: 目录; gguf: .gguf 文件)",
    )
    parser.add_argument(
        "--source-lang-name",
        default="中文",
        help="源语言名称（用于构建翻译 prompt）",
    )
    parser.add_argument(
        "--target-lang-name",
        default="英语",
        help="目标语言名称（用于构建翻译 prompt）",
    )
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda"),
        default="cpu",
        help="ASR 推理后端 (cuda 仅影响 FunASR-Nano)",
    )
    parser.add_argument(
        "--device-translator",
        choices=("cpu", "cuda"),
        default="cpu",
        help="翻译推理后端",
    )

    # --- ASR generation parameters ---
    parser.add_argument("--max-new-tokens", type=int, default=512, help="ASR 单段最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=1e-6, help="ASR temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="ASR top-p")
    parser.add_argument("--seed", type=int, default=42, help="ASR 随机种子")
    parser.add_argument("--language", default="", help="ASR 语言（留空使用默认）")
    parser.add_argument("--itn", action="store_true", default=True, help="启用 ITN")
    parser.add_argument("--no-itn", action="store_false", dest="itn")
    parser.add_argument("--hotwords", default="", help="ASR 热词（逗号分隔）")
    parser.add_argument("--debug", action="store_true", help="输出 ASR 调试信息")
    parser.add_argument(
        "--vad-threshold", type=float, default=0.5, help="Silero VAD 阈值"
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float, default=0.5,
        help="VAD 最短静音秒数",
    )
    parser.add_argument(
        "--min-speech-duration",
        type=float, default=0.25,
        help="VAD 最短语音段秒数",
    )
    parser.add_argument(
        "--max-speech-duration",
        type=float, default=20.0,
        help="VAD 最长语音段秒数",
    )
    parser.add_argument(
        "--mix-mode",
        choices=("average", "add"),
        default="average",
        help="多设备混音模式",
    )
    parser.add_argument(
        "--debug-save-audio",
        default="",
        help="将混音音频保存到 WAV 文件（调试用）",
    )

    # --- Translation generation parameters ---
    parser.add_argument("--max-new-tokens-translator", type=int, default=4096,
                        help="翻译最多生成 token 数")
    parser.add_argument("--temperature-translator", type=float, default=0.7,
                        help="翻译 temperature")
    parser.add_argument("--top-p-translator", type=float, default=0.6,
                        help="翻译 top-p")
    parser.add_argument("--top-k", type=int, default=20,
                        help="翻译 top-k")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="翻译重复惩罚")

    return parser.parse_args(argv)


def create_recognizer(args):
    """Create FunASR-Nano ASR recognizer."""
    return sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        encoder_adaptor=args.encoder_adaptor,
        llm=args.llm,
        embedding=args.embedding,
        tokenizer=args.tokenizer,
        num_threads=args.num_threads,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=args.debug,
        provider=args.provider,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        language=args.language,
        itn=args.itn,
        hotwords=args.hotwords,
    )


def create_vad(args):
    """Create Silero VAD."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.silero_vad.threshold = args.vad_threshold
    config.silero_vad.min_silence_duration = args.min_silence_duration
    config.silero_vad.min_speech_duration = args.min_speech_duration
    config.silero_vad.max_speech_duration = args.max_speech_duration
    config.sample_rate = sample_rate
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    return vad


def decode_asr(recognizer, samples):
    """Run ASR decoding on audio samples."""
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def select_devices(args):
    """Select audio input devices."""
    audio = pyaudio.PyAudio()
    try:
        devices = get_audio_devices(audio)
        if not devices:
            raise RuntimeError("没有任何输入设备")

        if args.device >= 0:
            selected = [args.device]
        else:
            selected = select_input_device(devices, audio)
            if not selected:
                selected = [audio.get_default_input_device_info()["index"]]
        return devices, selected
    finally:
        audio.terminate()


def run_pipeline(args, recognizer, vad, translator, window_size, selected_devices):
    """
    Main pipeline: Audio → ASR → Translation → stdout (TMSpeech protocol)

    Protocol:
        - print(result, end='\n')  → TMSpeech interim result (single \n)
        - print("\n", end='')      → TMSpeech sentence done (second \n)
    """
    samples_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    recording_process = multiprocessing.Process(
        target=start_recording,
        args=(
            selected_devices,
            samples_queue,
            stop_event,
            args.mix_mode,
            args.debug_save_audio,
        ),
    )
    recording_process.start()
    printer = MyPrinter()
    buffered = np.empty(0, dtype=np.float32)

    print(f"ASR + 翻译已启动 ({args.source_lang_name} → {args.target_lang_name})", file=sys.stderr)
    print(f"翻译引擎: {args.translator_backend}", file=sys.stderr)

    try:
        while True:
            try:
                samples = _read_audio_batch(samples_queue)
            except queue.Empty:
                continue

            if buffered.size:
                samples = np.concatenate((buffered, samples))

            # VAD processing
            complete_length = (len(samples) // window_size) * window_size
            for offset in range(0, complete_length, window_size):
                vad.accept_waveform(samples[offset: offset + window_size])
            buffered = samples[complete_length:]
            _drain_vad_segments(vad, recognizer, translator, printer)
    finally:
        cleanup_recording_process(stop_event, recording_process)


def _read_audio_batch(samples_queue):
    """Read all available audio batches from queue."""
    chunks = [samples_queue.get(timeout=0.5)]
    while True:
        try:
            chunks.append(samples_queue.get_nowait())
        except queue.Empty:
            break
    if len(chunks) == 1:
        return np.asarray(chunks[0], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _drain_vad_segments(vad, recognizer, translator, printer):
    """Process all completed VAD segments: ASR → Translation → Output."""
    while not vad.empty():
        samples = vad.front.samples
        vad.pop()

        # Step 1: ASR
        asr_text = decode_asr(recognizer, samples)
        if not asr_text:
            continue

        # Step 2: Translation
        try:
            translated = translator.translate(asr_text)
        except Exception as e:
            print(f"翻译失败: {e}", file=sys.stderr)
            translated = asr_text  # Fallback to original text

        # Step 3: Output via TMSpeech protocol
        printer.do_print(translated)
        printer.on_endpoint()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = get_args(argv)
    validate_args(args)

    devices, selected_devices = select_devices(args)

    names = {index: device["name"] for index, device in devices}
    print(
        "使用输入设备: " + ", ".join(
            f"{index}: {names.get(index, '未知设备')}"
            for index in selected_devices
        ),
        file=sys.stderr,
    )

    # Initialize ASR
    print(f"正在加载 Fun-ASR-Nano ({args.provider}, {args.num_threads} threads)", file=sys.stderr)
    recognizer = create_recognizer(args)
    vad = create_vad(args)

    # Initialize Translator
    print(f"正在加载 Hy-MT2 翻译器 ({args.translator_backend})", file=sys.stderr)
    translator = create_translator(args)

    print("ASR + 翻译已启动，请说话", file=sys.stderr)
    run_pipeline(args, recognizer, vad, translator, 512, selected_devices)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在退出", file=sys.stderr)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"启动失败: {error}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as error:
        print(f"未知错误: {error}", file=sys.stderr)
        raise SystemExit(1)
