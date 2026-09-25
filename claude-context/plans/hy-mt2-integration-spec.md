# Hy-MT2-1.8B 适配 TMSpeech — 规格文档

## 1. 目的与范围

### 目的
将腾讯混元 Hy-MT2-1.8B 多语言翻译模型集成到 TMSpeech 项目中，实现以下两个目标：

1. **实时字幕翻译**：将 ASR 识别出的字幕实时翻译为目标语言
2. **ASR + 翻译一体化部署**：通过 Python 后端同时提供语音识别和翻译能力

### 边界
- TMSpeech 项目地址：`D:\TMSpeech`
- Hy-MT2 是翻译模型（不是 ASR 模型），需搭配 ASR 使用
- 集成方式：Python 后端 + TMSpeech 内置「命令行识别器」插件
- 本次不包含 Hy-MT2 模型文件打包进发布包（模型由用户自行部署）

## 2. 技术方案

### 2.1 架构设计

```
用户语音 → Windows 音频采集(NAudio) → 外部Python进程
                                    ├── FunASR-Nano (ASR: 语音→中文文字)
                                    ├── Hy-MT2-1.8B (翻译: 中文→目标语言)
                                    └── 按命令行识别器协议输出结果
                                        ↓
                                   TMSpeech 读取 stdout
                                        ↓
                                   实时字幕展示
```

### 2.2 Python 后端脚本

在 `external_recognizer/` 下创建 `hy-mt2-translator.py`：

**功能**：
1. 复用现有 `common_audio_utils.py` 的音频采集
2. 集成 FunASR-Nano 进行 ASR（语音→文字）
3. 集成 Hy-MT2-1.8B 进行实时翻译
4. 支持可配置的源语言和目标语言
5. **严格按 CommandRecognizer.cs 的协议输出**：
   - `\r`（回车）→ 更新当前行临时结果
   - `\n`（换行）→ 句子完成

> ⚠️ **协议说明**：`CommandRecognizer.cs`（第212-250行）使用逐字节读取：
> - 遇到 `\r` 时跳过（不输出，仅用于分隔临时结果）
> - 遇到 `\n` 时触发事件处理
> - 累计2个换行表示句子完成，1个换行表示临时结果更新
> 因此 Python 脚本必须输出 `text + \r + text + \r + text + \n` 格式

**推理引擎选项**：
- **transformers**（推荐，部署简单）：`pip install transformers>=5.6.0 torch`
- **vllm**（高性能）：需从源码构建
- **GGUF + llama.cpp**（轻量，CPU友好）：1.25-bit 量化仅 440MB

### 2.3 TMSpeech 配置适配

利用已有的「命令行识别器」插件，无需修改 C# 核心代码：
- 命令路径：Python 可执行文件路径
- 命令参数：`hy-mt2-translator.py --source-lang zh --target-lang en ...`
- 工作目录：`external_recognizer/`

### 2.4 模型部署

**Hy-MT2-1.8B 下载**：
- HuggingFace: `https://huggingface.co/tencent/Hy-MT2-1.8B`
- ModelScope: `https://www.modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B`
- 量化版：`Hy-MT2-1.8B-1.25bit-GGUF`（440MB，推荐用于 CPU）

**目录结构**：
```
external_recognizer/
├── hy-mt2-translator.py          # 新增：Hy-MT2 翻译脚本
├── common_audio_utils.py          # 已有：音频工具
├── simulate-streaming-funasr-nano.py  # 已有
├── silero_vad.onnx                # 已有
└── hy-mt2-1.8b/                   # 新增：模型目录
    ├── config.json
    ├── tokenizer/
    └── model-*.safetensors
```

## 3. 具体实施步骤

## 3. 具体实施步骤

### Step 1：创建 Hy-MT2 翻译脚本
- 文件：`external_recognizer/hy-mt2-translator.py`
- 功能：音频采集 → ASR（FunASR-Nano）→ 翻译（Hy-MT2）→ 按协议输出
- 严格按 CommandRecognizer.cs 协议：`text\r\n`（临时结果）+ `\n`（句子完成）
- 支持 `--provider cpu/cuda`、`--source-lang`、`--target-lang` 参数
- 支持 transformers、GGUF+llama.cpp 两种推理引擎

### Step 2：模型目录初始化
- 创建 `external_recognizer/hy-mt2-1.8b/` 目录占位符
- 添加 `.gitkeep` 文件（防止空目录被 git 忽略）
- 模型由用户自行下载，不打包进发布

### Step 3：编写依赖安装脚本
- `external_recognizer/install-hymt2.bat`
- 检测 Python 环境（PATH 中是否有 python/python3）
- 自动安装 transformers + torch（transformers 方式）或 llama-cpp-python（GGUF 方式）
- 显示下载 Hy-MT2 模型的指引链接

### Step 4：错误处理规范
- Python 进程崩溃时：TMSpeech 检测到 stdin 关闭，抛出异常并通知用户
- 模型文件缺失：启动时校验所有必需文件，缺失则报错并提示下载地址
- stderr 日志：通过 TMSpeech 命令行识别器的 LogFile 配置写入文件
- 冷启动时间：首次加载模型约 10-30 秒（transformers 方式）或 3-5 秒（GGUF 方式）

### Step 5：更新 README.md
- 新增 Hy-MT2 集成章节
- 包含完整的配置示例和参数说明
- 添加模型下载指引

### Step 6：测试打包
- 构建项目验证无编译错误
- 测试 Python 脚本独立运行（使用测试音频验证 ASR+翻译流程）
- 在 TMSpeech 中配置命令行识别器，验证完整流程

## 4. 成功标准

- [x] Python 脚本能独立运行，完成 ASR+翻译流程
- [x] TMSpeech 配置「命令行识别器」后能正常加载 Hy-MT2 翻译结果
- [x] 项目能正常构建（Build 通过，无编译错误）
- [x] README 包含完整的使用文档

## 5. 风险评估

| 风险 | 影响 | 缓解措施 |
|:---|:---|:---|
| Hy-MT2 推理速度慢 | 字幕延迟高 | 推荐使用 1.25-bit GGUF + llama.cpp |
| 模型文件太大 | 发布包体积膨胀 | 本次不打包模型，用户自行下载 |
| GPU 内存不足 | 无法运行 | 提供 CPU 推理选项 |
| Python 环境依赖 | 用户需手动安装 | 提供 install-hymt2.bat 一键安装 |
