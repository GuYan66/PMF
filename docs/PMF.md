# PMF 结构与实现约定

PMF = Pooling-based Multimodal Fusion。默认共 162,436 个可训练参数，使用已提取的对齐特征。

## 前向流程

1. 文本 768 维、语音 74 维、视觉 35 维，各自在 50 个位置上进行掩码均值池化。
2. 掩码为 text_bert[:,1,:]，1 是有效位置，0 是 padding。不从全零行或 UNK 推断 padding；实际缺失位置仍保留在池化分母中。
3. 分别标准化池化后的特征维度。均值、总体标准差只用训练集计算。标准差小于 1e-6 的维度缩放因子取 1。
4. 各模态使用 Linear(input_dim,128) + ReLU，拼接为 384 维。
5. Linear(384,128) + ReLU + Dropout(0.2)。
6. 极性头 Linear(128,3)；幅度头 Linear(128,1) 后接 3×sigmoid。

模型的 feature_mean、feature_std、normalization_fitted 是注册 buffer，随权重保存。新样本不重新拟合统计量。
Feature_t、Feature_a、Feature_v、Feature_f 输出分别对应投影后模态表示和融合表示，便于后续消融与解释。

## 训练与预测

普通三分类交叉熵 + 非中性真实标签上的绝对幅度 L1，权重均为 1。全中性批次幅度损失为 0。
极性取 argmax，类别为负向 0、中性 1、正向 2。中性强度输出 0，负向取负幅度，正向取正幅度。
默认 Adam，learning_rate=0.001，weight_decay=0，batch_size=64，early_stop=8。
检查点沿用四位小数 MacroF1 严格提升的规则；test2 和 MAE 不用于检查点选择。

## 实现位置

- src/MMSA/models/singleTask/PMF.py：网络与标准化 buffer。
- src/MMSA/models/subNets/PolMagHead.py：双头、标签映射、强度合成。
- src/MMSA/trains/singleTask/POLMAG.py：训练、训练集统计量拟合、评测。
- src/MMSA/data_loader.py：显式 padding mask 和可选 test2。
- src/MMSA/utils/polmag_results.py：每种子明细、独立汇总。
- src/MMSA/config/config_regression.json：PMF 默认配置。

AMIO 的 PMF 前向需要显式 padding_mask。MMSA_test 和独立推理脚本从 text_bert 取得该掩码。
已有 MMSA 模型代码保留原版；不再给其他模型开启 polmag。PMF 当前支持 MOSEI 对齐数据的普通训练模式。
完整操作步骤见根 README，测试见 tests/。
