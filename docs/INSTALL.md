# 安装与环境

已验证环境为 Linux、Python 3.10.21、PyTorch 2.4.1+cu118。PMF 支持 CPU；Windows 的持续集成配置已提供，实际通过状态以 GitHub Actions 运行结果为准。

创建独立环境，例如 `conda create -n pmf python=3.10`，然后 `conda activate pmf`。以下均在仓库根目录执行。

## CPU

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints/py310-tested.txt -e .
```

## NVIDIA GPU：与当前服务器一致的 CUDA 11.8 构建

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -c constraints/py310-tested.txt -e .
```

CPU/CUDA 安装索引依据 [PyTorch 2.4.1 官方安装说明](https://pytorch.org/get-started/previous-versions/#v241)。PMF 不需要 torchvision 或 torchaudio。
已有同版本 PyTorch 时，不必重复安装。其他 GPU 构建按官方说明选择；本项目尚未逐一验证其他组合。

核心版本见 `constraints/py310-tested.txt`。包元数据限制 PyTorch 2.4 系列与 Transformers 4.46 系列，避免上游旧模型导入接口发生不兼容变化。
虽然 PMF 不运行 BERT 编码器，完整 MMSA 的模块导入仍需要 transformers、pytorch-transformers、einops 等依赖，安装声明已包含它们。

普通 PMF 训练和 test2 推理无需下载 BERT，也无需安装 test2 专用组件。test2 是由队友单独共享的现成 pkl，通过 --test2 路径读取；数据和参考标签不进入 Git。

检查环境：

```bash
python -c "import MMSA, torch; print(MMSA.__file__); print(torch.__version__)"
python -m unittest discover -s tests -v
```

打包安装：`python -m pip wheel --no-deps --no-build-isolation . -w dist`。wheel 包含 MMSA Python 模块和内置 JSON 配置；团队首次共享推荐完整源码目录，包含运行脚本、测试和文档。
