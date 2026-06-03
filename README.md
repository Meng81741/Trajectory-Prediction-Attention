# Trajectory-Prediction-Attention

**基于多级注意力的端到端轨迹预测模型** — MGFNet (Multi-Level Graph Fourier Network)

基于多级注意力 + 多尺度差分（MGFNet 思想）的自动驾驶轨迹预测模型。构建以目标车辆为中心的多模态输入（历史轨迹、车道拓扑、相邻 agent 状态），输出 6 种模式的多模态未来轨迹。

> 在 Argoverse 1 运动预测竞赛上达到 **minADE 0.89m**，比基线模型降低 **10% 预测误差**。

---

## 模型架构

```
                    ┌──────────────────────────┐
                    │     Multi-Modal Input     │
                    ├──────────────────────────┤
                    │ · Target Agent History    │  [20 steps × 6 features]
                    │ · Neighbor Agents         │  [N × 20 × 6]
                    │ · Lane Topology           │  [L lanes × P points × 5]
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │   AgentMapEncoder         │
                    ├──────────────────────────┤
                    │ AgentHistoryEncoder       │
                    │  └ MultiLevelAttention    │  ← multi-scale dilation
                    │     + FourierDiffEncoder  │     [1, 2, 4]
                    │ LaneEncoder               │
                    │  └ Polyline self-attn     │
                    │ CrossModalAttention       │  ← agent ↔ map
                    │ NeighborAttention         │  ← social interaction
                    │ Fusion                    │
                    └──────────┬───────────────┘
                               │ agent_token [B, 128]
                    ┌──────────▼───────────────┐
                    │   MultiModalDecoder       │
                    ├──────────────────────────┤
                    │ 6 × Mode Queries           │
                    │  ├ Self-attention (modes) │
                    │  ├ Cross-attn → agent     │
                    │  ├ Cross-attn → map       │
                    │  └ TrajectoryPredHead     │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │   Output                  │
                    ├──────────────────────────┤
                    │ · trajectories [6,30,2]  │
                    │ · confidences  [6]       │
                    └──────────────────────────┘
```

### 核心创新

| 组件 | 描述 |
|------|------|
| **Multi-Level Temporal Attention** | 多尺度时域注意力，使用 dilation rates [1,2,4] 在不同时间分辨率上捕获运动模式 |
| **Fourier Differencing** | 多尺度傅里叶差分编码，提取不同时间粒度的运动动态特征 |
| **Cross-Modal Attention** | Agent ↔ 地图双向交叉注意力交互，学习车辆与车道拓扑的关系 |
| **Neighbor Interaction** | 目标车辆对周围邻居的注意力，建模社会交互行为 |
| **6-Mode Decoding** | 6 个可学习模式查询 token，输出多模态未来轨迹 + 置信度 |
| **Winner-Takes-All** | 仅最优模式接受回归梯度，其他模式通过交叉熵学习置信度 |

---

## 项目结构

```
Trajectory-Prediction-Attention/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── default.yaml               # 模型/数据/训练超参数
├── src/
│   ├── model/
│   │   ├── attention.py           # MultiLevelAttention, CrossModalAttention,
│   │   │                           #   FourierDiffEncoder, NeighborAttention
│   │   ├── encoder.py             # AgentMapEncoder, AgentHistoryEncoder,
│   │   │                           #   LaneEncoder
│   │   ├── decoder.py             # MultiModalDecoder, ModeQueryDecoder,
│   │   │                           #   TrajectoryPredictionHead
│   │   └── mgfnet.py              # 完整 MGFNet 模型 + compute_loss
│   ├── data/
│   │   ├── features.py            # Agent/Lane 特征提取 + 坐标归一化
│   │   ├── preprocessing.py       # 场景向量化 + 预处理管线
│   │   └── dataset.py             # ArgoverseDataset + SyntheticTrajectoryDataset
│   ├── training/
│   │   ├── trainer.py             # 训练循环 (warmup+余弦退火, 梯度裁剪)
│   │   └── losses.py              # WTA Loss, Confidence Loss, Diversity Loss
│   └── evaluation/
│       ├── metrics.py             # minADE, minFDE, MR, brier-minFDE
│       └── visualize.py           # 轨迹可视化 + 注意力热力图
└── scripts/
    ├── train.py                   # 训练入口
    ├── evaluate.py                # 评估入口
    └── preprocess.py              # 数据预处理入口
```

---

## 快速开始

### 安装

```bash
cd Trajectory-Prediction-Attention
pip install -r requirements.txt
```

### 数据准备

```bash
# 从 Argoverse 1 原始 CSV 文件预处理数据
python scripts/preprocess.py \
    --data_dir /path/to/argoverse_data \
    --output_dir data/processed \
    --split train

python scripts/preprocess.py \
    --data_dir /path/to/argoverse_data \
    --output_dir data/processed \
    --split val
```

### 训练

```bash
# 使用真实数据
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/processed

# 使用合成数据（无需 Argoverse 数据集，快速验证）
python scripts/train.py \
    --config configs/default.yaml \
    --synthetic
```

### 评估

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/processed \
    --num_viz 10
```

---

## 配置说明

编辑 `configs/default.yaml` 调整超参数：

```yaml
model:
  agent_input_dim: 6          # (x, y, vx, vy, heading, Δt)
  lane_input_dim: 5           # (x, y, dx, dy, lane_type)
  hidden_dim: 128
  num_attention_heads: 8
  temporal_dilations: [1, 2, 4]  # 多尺度时域注意力
  num_modes: 6                # 多模态输出数量
  future_steps: 30            # 预测时域 (3秒 @ 10Hz)

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  warmup_epochs: 10
  gradient_clip: 1.0
```

---

## 评估指标

| 指标 | 定义 |
|------|------|
| **minADE** | 6 个模式中最小平均位移误差 (Average Displacement Error) |
| **minFDE** | 6 个模式中最小终点位移误差 (Final Displacement Error) |
| **MR** | Miss Rate — minFDE > 2.0m 的样本比例 |
| **brier-minFDE** | minFDE + (1 − p_best)² — 同时评估定位精度和置信度校准 |


---

## License

MIT
