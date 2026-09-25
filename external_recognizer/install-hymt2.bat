@echo off
chcp 65001 >nul
echo ============================================
echo   Hy-MT2-1.8B TMSpeech 翻译器 环境配置脚本
echo ============================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    python3 --version >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [错误] 未检测到 Python，请先安装 Python 3.8+
        echo 下载地址: https://www.python.org/downloads/
        pause
        exit /b 1
    ) else (
        set PYTHON=python3
    )
) else (
    set PYTHON=python
)

echo [检测] Python 版本:
%PYTHON% --version
echo.

:: Ask which backend to install
echo 请选择翻译推理引擎:
echo   1) transformers  (推荐，GPU 支持更好)
echo   2) llama.cpp     (轻量，仅需约 440MB 模型文件)
set /p BACKEND="请输入选项 (1或2，默认1): "

if "%BACKEND%"=="" set BACKEND=1
if "%BACKEND%"=="2" (
    echo.
    echo 正在安装 llama-cpp-python ...
    pip install llama-cpp-python
) else (
    echo.
    echo 正在安装 transformers + torch ...
    pip install transformers torch
)

echo.
echo ============================================
echo   安装完成!
echo   下一步: 下载 Hy-MT2-1.8B 模型
echo   Transformers 版: https://huggingface.co/tencent/Hy-MT2-1.8B
echo   GGUF 版:         https://huggingface.co/tencent/Hy-MT2-1.8B-1.25bit-GGUF
echo ============================================
echo.
pause
