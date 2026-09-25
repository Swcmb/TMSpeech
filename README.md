# TMSpeech

(旧版)视频演示：https://www.bilibili.com/video/BV1rX4y1p7Nx/

关键词：语音转文字，实时字幕，会议语音识别，歌词字幕展示，识别历史记录查看

`TMSpeech` 是一个Windows下的中文实时语音字幕，通过WASAPI的CaptureLoopback捕获电脑声音（录内音），将语音实时转文字，并以歌词字幕的形式展示。即使完全关闭电脑声音也能使用。

你可以：
- 开会时更放心地走神，突然被喊到的时候不会那么不知所措，只需要看一看识别的历史记录。（本项目的名字来源于此）
- 会议实时转录，自动生成会议纪要，并保存到文件。默认会将识别结果按日期保存到“我的文档”的`TMSpeechLogs`文件夹中

基于[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/)项目的语音识别框架和模型二次开发。实测在我的AMD 5800u的笔记本上CPU占用不到5%。

## 展示

无边框窗口，可任意拖动和调整大小

![正常识别窗口](imgs/main.png)

历史记录页面（可右键或者Ctrl-C复制）：

![历史记录页面](imgs/history.png)

设置页面：

![设置页面](imgs/settings.png)

切换为命令行识别器：

![设置页面](imgs/setting-change-recognizer.png)

## 使用

在[Release](https://github.com/jxlpzqc/TMSpeech/releases)页面中下载最新的release解压，运行`TMSpeech.exe`即可。在桌面创建快捷方式，使用起来更加方便。出现问题后可以运行重置配置的bat脚本，删除现有配置文件。

## 基于自定义外部命令的识别

在设置中选用“命令行识别器”。它基于程序和参数，启动子进程，并将标准输出（stdout）作为字幕格式识别，将标准错误输出（stderr）作为日志文件记录（都使用UTF-8编码）。

使用单个换行（'\n'）更新当前句子，使用多个换行（'\n\n'）表示当前行识别结束，样例输出如下：

```
一二
一二三四
一二三四五六七

七六
七六五四
七六五四三二一

```

参考python代码如下：

```diff
+ class MyPrinter:
+     def __init__(self):
+         self.prev_result = ""
+ 
+     def do_print(self, result):
+         if result and self.prev_result != result:
+             self.prev_result = result
+             print(result, end='\n', flush=True)
+ 
+     def on_endpoint(self):
+         print("\n", end="", flush=True)
+ 
+     printer = MyPrinter()
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate, device=device) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            is_endpoint = recognizer.is_endpoint(stream)
            result = recognizer.get_result(stream)

+           printer.do_print(result)
            if is_endpoint:
+               if result:
+                   printer.on_endpoint()
                recognizer.reset(stream)
```

注意事项：

1. 单个换行结尾的行是临时结果，只有多个换行结尾的行才会被存储到历史记录中，这种方式允许模型在后面纠正前面的识别结果。
1. 基于该方式需要子进程独立获取语音源。在设置中切换语音源将不会生效。
1. 程序接受多个参数时，使用空格分割，如果参数本身包含空格，比如带有空格的路径，则可能会出现问题，需要通过双引号转义。详见[这里](https://stackoverflow.com/questions/15061854/how-to-pass-multiple-arguments-in-processstartinfo)和[这里](https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.processstartinfo.arguments?view=net-10.0)
1. 程序指定为批处理脚本（'.bat'）时，记得前面加上@隐藏命令显示，同时不要在结尾加入`pause`这种命令（无法检测命令的退出）。

    ```bat
    @python ./speech-recognition-from-microphone-with-endpoint-detection.py
    ```

### 使用 Fun-ASR-Nano

Fun-ASR-Nano 是离线自回归模型，不能直接替换内置 C# 流式识别器。仓库中的
`external_recognizer/simulate-streaming-funasr-nano.py` 会复用现有的多设备录音和
Silero VAD：检测到完整语音段后调用 Nano，并按命令行识别器协议输出最终字幕。

先在仓库根目录安装支持 Nano API 的 sherpa-onnx：

```powershell
python -m pip install --upgrade PyAudioWPatch sherpa_onnx==1.13.4 scipy
```

下载 [Fun-ASR-Nano INT8 模型](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2)
（压缩包约 842 MB）和 [Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)，按下面的默认目录放置：

```text
external_recognizer/
├── silero_vad.onnx
└── sherpa-onnx-funasr-nano-int8-2025-12-30/
    ├── encoder_adaptor.int8.onnx
    ├── llm.int8.onnx
    ├── embedding.int8.onnx
    └── Qwen3-0.6B/
```

可以先在命令行用 CPU 启动：

```powershell
python .\external_recognizer\simulate-streaming-funasr-nano.py --provider cpu --num-threads 4
```

确认运行正常后，在 TMSpeech 设置中选择“命令行识别器”，填写：

```text
命令路径：C:\Path\to\python.exe
命令参数：simulate-streaming-funasr-nano.py --provider cpu --num-threads 4
工作目录：C:\Path\to\TMSpeech\external_recognizer
```

需要热词时可以追加 `--hotwords "TMSpeech,FunASR"`。`--language` 默认留空；可用值和
识别范围以下载的 checkpoint 说明为准。`--provider cuda` 会把模型请求转给 CUDA，
但需要安装带 CUDA execution provider 的 sherpa-onnx 构建，普通 CPU wheel 不会因此
自动获得 GPU 支持。

这个脚本只解码 VAD 已完成的语音段，不会每 200 ms 重跑一次自回归模型。因此字幕会在
说完并出现短暂停顿后一次性提交；这是为了避免 Nano 在 CPU 上持续重复推理。可以通过
`--min-silence-duration` 和 `--max-speech-duration` 调整分段延迟与最长句长。

### 使用 Hy-MT2-1.8B 实时翻译

Hy-MT2-1.8B 是腾讯混元的轻量多语言翻译模型，可与 Fun-ASR-Nano 组合使用：Nano 负责语音识别（ASR），Hy-MT2 负责实时翻译，实现「语音→原文→译文」的一体化字幕。

#### 安装依赖

双击运行或手动执行：

```powershell
.\external_recognizer\install-hymt2.bat
```

或手动安装：

```powershell
# 方式一：transformers（推荐 GPU 用户）
pip install transformers torch

# 方式二：GGUF + llama.cpp（推荐 CPU 用户，模型仅约 440 MB）
pip install llama-cpp-python
```

#### 下载模型

| 模型 | 大小 | 适合场景 |
|:---|:---|:---|
| [Hy-MT2-1.8B (full)](https://huggingface.co/tencent/Hy-MT2-1.8B) | ~3.6 GB | GPU 推理 |
| [Hy-MT2-1.8B-1.25bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25bit-GGUF) | ~440 MB | **CPU 推理（推荐）** |

下载后放入 `external_recognizer/hy-mt2-1.8b/` 目录（transformers 格式）或任意路径（GGUF 格式）。

#### 配置 TMSpeech

在设置中选择 **"命令行识别器"**，填写：

```text
命令路径：C:\Path\to\python.exe
命令参数：hy-mt2-translator.py --source-lang-name 中文 --target-lang-name 英语 --translator-backend transformers
工作目录：C:\Path\to\TMSpeech\external_recognizer
stderr保存：C:\Path\to\TMSpeechLogs\hymt2-error.log
```

GGUF 方式（CPU 推荐）：

```text
命令路径：C:\Path\to\python.exe
命令参数：hy-mt2-translator.py --translator-backend gguf --hy-mt2-model hy-mt2-1.8b.gguf --source-lang-name 中文 --target-lang-name 英语
工作目录：C:\Path\to\TMSpeech\external_recognizer
```

#### 支持的语言

Hy-MT2 支持 33 种语言互译，包括中文、英语、日语、韩语、法语、德语、俄语等。
使用 `--source-lang-name` 和 `--target-lang-name` 参数指定语言名称（如 `中文`、`英语`、`日本語`、`Français`、`Deutsch` 等），与模型训练时的语言名称保持一致。

#### 命令行参数

| 参数 | 说明 | 默认值 |
|:---|:---|:---|
| `--source-lang-name` | 源语言名称 | `中文` |
| `--target-lang-name` | 目标语言名称 | `英语` |
| `--translator-backend` | 翻译引擎 `transformers` / `gguf` | `transformers` |
| `--hy-mt2-model` | 模型路径（transformers: 目录；gguf: .gguf 文件） | `hy-mt2-1.8b/` |
| `--num-threads` | ASR 推理线程数 | `4` |
| `--provider` | ASR 推理后端 `cpu` / `cuda` | `cpu` |
| `--device-translator` | 翻译推理后端 `cpu` / `cuda` | `cpu` |
| `--max-speech-duration` | 最长语音段（秒），影响翻译延迟 | `20.0` |
| `--min-silence-duration` | 句子结束所需最短静音（秒） | `0.5` |

#### 注意事项

- **延迟**：翻译在 VAD 语音段结束后执行，因此字幕会在说完并停顿后出现。可通过 `--max-speech-duration` 调整。
- **stderr 日志**：翻译模型的加载进度、错误信息会写入 stderr，可在 TMSpeech 配置的 LogFile 中查看。
- **不打包模型**：Hy-MT2 模型文件不包含在发布包中，需用户自行下载。
- **协议**：与 Fun-ASR-Nano 脚本相同，单换行更新临时结果，双换行表示句子完成。


## 我们需要你的反馈

觉得很有用？但是还有不完美的地方？欢迎点击这里[创建Discussion](https://github.com/jxlpzqc/TMSpeech/discussions/new)、提出反馈！

- 识别准确率不高？
    - 这可能需要更好的模型。当前我们支持sherpa-onnx的流式模型，可以在[这里](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)下载其他模型，并在设置中修改模型路径。
    - [想要用自己的模型？](https://github.com/jxlpzqc/TMSpeech/issues/6) 如果你发现了效果更好的开源模型，也欢迎推荐给我们！
- 还需要更多功能？
    - 请点击这里[创建issue](https://github.com/jxlpzqc/TMSpeech/issues/new)告诉我们！
    - 如果你懂Windows/C#开发，欢迎提交pull request，开发的过程中遇到任何问题可以创建issue和我们讨论。

## 带模型的Release打包流程

- 在github actions中下载构建好的安装包
- 放入正确的模型文件夹。放入正确的default_config.json，对应上当前的识别器。
- 打包为zip文件，在开发电脑，和另外一台电脑上测试各种功能。
