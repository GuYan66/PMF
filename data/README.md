# 数据接口

本目录只跟踪此说明，数据、参考标签和生成特征通过团队已有渠道单独共享。

## 附件 2

`aligned_50.pkl` 顶层是 train、valid、test。每个划分包含：

| 字段 | 形状/类型 |
| --- | --- |
| text | N×50×768 浮点特征 |
| text_bert | N×3×50；第 1 通道为 attention mask |
| audio | N×50×74 |
| vision | N×50×35 |
| raw_text、id | 长度 N |
| regression_labels | N，范围 [-3,3] |

当前竞赛数据为 train 3395、valid 728、test 727。PMF 不使用原文重新提取训练文本特征。
标准化从 train 的池化向量拟合，统计量随 checkpoint 保存。输入不是三个独立 feature_T/A/V 覆盖文件，掩码必须与三模态来自同一个特征文件。

## 附件 3 test2

可以直接使用已经构建好的 `test2_aligned_50.pkl`，顶层为 test2，字段与附件 2 的单个划分相同。30 条参考标签为负向 5、中性 9、正向 16。

队友直接共享现成的 `test2_aligned_50.pkl`，不需要重建，也不需要安装额外组件或下载 BERT。

```bash
python scripts/train_pmf.py --data data/aligned_50.pkl --test2 /path/to/test2_aligned_50.pkl --device cpu
```

该文件保留附件 3 的缺失音视频与词元输入，并已经包含由缺失词元生成的 text 特征。训练代码直接读取，不使用恢复的完整媒体或原文填补模型输入。

当前 test2 与 train、valid 无相同片段或原视频；与 test 有 4 条相同片段、21 条来自相同原视频。单独报告 test2，不能把它当作完全独立于 test 的新证据。
参考标签来自公开源匹配，并未独立取得比赛隐藏答案进行核验。test2 只在训练完成、最佳检查点加载后评测。

## 单样本推理

平铺 pickle 需要 text[50,768]、text_bert[3,50]、audio[50,74]、vision[50,35]，id 可选，标签不需要。
`predict_pmf.py` 也接受整个生成的 test2 pickle，输出的预测不读取 regression_labels。
