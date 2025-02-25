# BCI2022_Demo

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english) | [中文](#chinese)

<h2 id="chinese">中文版</h2>

## 项目简介

这是一个基于功能性近红外光谱成像（fNIRS）数据的脑机接口分类演示程序。该程序使用支持向量机（SVM）分类器对 fNIRS 数据进行训练和预测，实现了下肢运动想象与休息状态的二分类。

## 快速开始

### 环境要求

- Python 3.6+
- numpy >= 1.16.0
- pandas >= 1.0.0
- scipy >= 1.4.0
- scikit-learn >= 0.22.0

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/Jett-Wu/BCI2022_Demo.git
cd BCI2022_Demo
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 使用说明

1. 将数据文件放在 `./data/` 目录下：
   - train_data.mat（训练数据）
   - train_label.mat（训练标签）
   - test_data.mat（测试数据）

2. 运行程序：
```bash
python Demo.py
```

3. 输出文件：
   - submission.csv（预测结果的CSV格式）
   - submission.mat（预测结果的MAT格式）

## 数据集描述

本项目使用 BCI2022 竞赛数据集，包含 8 名受试者的 fNIRS 数据。每位受试者执行两种任务：下肢运动想象和休息状态。

### 数据集划分
- 训练集：受试者 01-04 的数据
- 测试集：受试者 05-08 的数据

### 数据格式
- 训练集/测试集维度：(160, 2, 22, 152)
  - 160：样本数量
  - 2：血氧信号类型（HbO、HbR）
  - 22：通道数量
  - 152：时间采样点数
- 标签维度：(160)
  - 0：休息状态
  - 1：运动想象状态

### 评估指标
分类准确率计算方式：
```
accuracy = (true positives + true negatives) / total examples
```

## 程序流程

1. 加载训练数据和测试数据
2. 提取统计特征（均值、最大值、标准差）
3. 训练 SVM 分类器
4. 对测试数据进行预测
5. 将预测结果保存为指定格式

## 贡献指南

欢迎提交问题和改进建议。如需贡献代码，请遵循以下步骤：
1. Fork 本仓库
2. 创建您的特性分支
3. 提交您的更改
4. 推送到您的分支
5. 创建 Pull Request

## 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件

---

<h2 id="english">English Version</h2>

## Project Overview

This is a Brain-Computer Interface (BCI) classification demo based on functional Near-Infrared Spectroscopy (fNIRS) data. The program implements a Support Vector Machine (SVM) classifier to discriminate between lower limb motor imagery and resting states.

## Quick Start

### Requirements

- Python 3.6+
- numpy >= 1.16.0
- pandas >= 1.0.0
- scipy >= 1.4.0
- scikit-learn >= 0.22.0

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Jett-Wu/BCI2022_Demo.git
cd BCI2022_Demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Place data files in the `./data/` directory:
   - train_data.mat (training data)
   - train_label.mat (training labels)
   - test_data.mat (testing data)

2. Run the program:
```bash
python Demo.py
```

3. Output files:
   - submission.csv (predictions in CSV format)
   - submission.mat (predictions in MAT format)

## Dataset Description

This project uses the BCI2022 competition dataset, containing fNIRS data from 8 subjects. Each subject performed two tasks: lower limb motor imagery and resting state.

### Dataset Split
- Training set: data from subjects 01-04
- Testing set: data from subjects 05-08

### Data Format
- Training/Testing data shape: (160, 2, 22, 152)
  - 160: number of samples
  - 2: types of blood oxygen signals (HbO, HbR)
  - 22: number of channels
  - 152: number of time points
- Label shape: (160)
  - 0: resting state
  - 1: motor imagery state

### Evaluation Metric
Classification accuracy calculation:
```
accuracy = (true positives + true negatives) / total examples
```

## Program Flow

1. Load training and test data
2. Extract statistical features (mean, max, standard deviation)
3. Train SVM classifier
4. Make predictions on test data
5. Save predictions in specified formats

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests. To contribute:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
