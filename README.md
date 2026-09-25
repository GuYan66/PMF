# MMSA + PMF

基于 [THUIAR/MMSA](https://github.com/thuiar/MMSA) 的团队代码版本，新增最简基线 **PMF（Pooling-based Multimodal Fusion）**。原始 MMSA 模型保留上游实现；本项目的双头训练仅用于 PMF。

PMF 使用已提取的对齐特征，不微调特征提取器：

```text
文本 50×768 ─ 掩码均值池化 ─ 训练集标准化 ─ Linear→128 ─ ReLU ┐
音频 50×74  ─ 掩码均值池化 ─ 训练集标准化 ─ Linear→128 ─ ReLU ┼ 拼接384
视觉 50×35  ─ 掩码均值池化 ─ 训练集标准化 ─ Linear→128 ─ ReLU ┘
       → Linear(384,128) → ReLU → Dropout(0.2)
       → 极性头 Linear(128,3) / 幅度头 3×sigmoid(Linear(128,1))
```

默认可训练参数 **162,436**。普通交叉熵 + 真实非中性样本的幅度 L1；预测中性时最终强度为 0。

## 1. 安装

建议使用独立的 **Python 3.10** 环境。在本仓库根目录执行：

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints/py310-tested.txt -e .
```

上面安装 CPU 版。GPU 安装步骤、环境范围见 [安装说明](docs/INSTALL.md)。约束文件记录已测试的核心依赖，不是全量依赖锁。

## 2. 准备数据

数据和模型产物单独共享，不放入 Git。最低需要附件 2 的 `aligned_50.pkl`，例如放在 `data/aligned_50.pkl`。
可选的附件 3 `test2_aligned_50.pkl` 放在 `data/attachment3/`。

输入字段和 test2 使用方式见 [数据说明](data/README.md)。只有附件 2 就可以训练，test2 是可选项。

## 3. 检查和训练

先校验数据与前向流程，不训练、不保存产物：

```bash
python scripts/train_pmf.py --data data/aligned_50.pkl --device cpu --dry-run
```

正式训练：

```bash
python scripts/train_pmf.py --data data/aligned_50.pkl --config configs/pmf.example.json --device cpu --seeds 1111 1112 1113
```

GPU 使用 `--device cuda:0`，也可使用默认的 `auto`。启用附件 3 时追加：

```text
--test2 data/attachment3/test2_aligned_50.pkl
```

用 `--output-dir` 指定产物目录，默认 `outputs/`。所有输入路径相对于当前工作目录。

## 4. 输出与评测

```text
outputs/
  saved_models/<run_id>/pmf-mosei-seed1111.pth
  saved_models/<run_id>/pmf-mosei-seed1111.json
  results/normal/mosei_polmag_v1.csv
  results/normal/mosei_polmag_v1_summary.csv
  logs/pmf-mosei.log
```

每个种子保留自己的权重和完整配置；不同运行使用不同 run_id。标准化统计量保存在权重中。

- `acc3`、`macro_f1`：负向/中性/正向三分类；`mae`、`corr`：全部样本的最终有符号强度。
- CSV 保存原始数值，不乘 100；未定义的 Corr 留空，汇总表用 corr_n 标识有效种子数。
- 原 train 训练，valid MacroF1 选择检查点；test、test2 在加载最佳检查点后评测。选择规则保持四位小数 MacroF1 严格提升。
- test2 为公开来源参考标签评测，单独记为 `split=test2`，不用于训练或选模型。它与原 test 有 4 条相同片段，不能将两组当作完全独立证据。
- 多种子汇总的标准差使用 ddof=0；已有 CSV 表头不兼容时在训练前报错。

## 5. 推理和测试

```bash
python scripts/predict_pmf.py --config outputs/saved_models/RUN_ID/pmf-mosei-seed1111.json --weights outputs/saved_models/RUN_ID/pmf-mosei-seed1111.pth --features data/attachment3/test2_aligned_50.pkl --output outputs/predictions.json
python -m unittest discover -s tests -v
```

将 RUN_ID 换为实际目录名。推理也接受包含 text、text_bert、audio、vision 的单样本平铺 pickle，输出极性类别、幅度和最终强度，不读取标签来生成预测。

## 文档与来源

- [PMF 结构和扩展约定](docs/PMF.md)
- [团队共享与上传 Git](docs/SHARING.md)
- [改动记录](CHANGELOG.md)
- [上游版本与许可证](UPSTREAM.md)，[上游 README 原文](docs/UPSTREAM_README.md)

沿用上游 MIT 许可证，详见 [LICENSE](LICENSE)。自动化工作流只运行测试，不自动发布 PyPI。CI 配置覆盖 Linux/Windows；发布前的本地检查范围记录在 [验证记录](docs/VERIFICATION.md)。
