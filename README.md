# 区块链价格预测

这个项目使用机器学习技术预测加密货币（如比特币和以太坊）的价格走势。通过历史价格数据和技术指标，模型可以预测未来一段时间内的价格变动趋势。

## 功能特点

- 自动获取比特币和以太坊的历史价格数据
- 使用技术指标和机器学习模型进行价格预测
- 生成价格预测图表和CSV报告
- 支持自定义预测时间段

## 安装

1. 克隆此仓库
```bash
git clone https://github.com/yourusername/blockchain-price-prediction.git
cd blockchain-price-prediction
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 比特币价格预测

```bash
python btc_prophecy.py
```

### 以太坊价格预测

```bash
python eth_prophecy.py
```

## 预测结果示例

预测结果会生成图表文件（如`btc_prediction.png`和`eth_prediction.png`）和CSV报告。

## 依赖库

- pandas
- yfinance
- matplotlib
- numpy
- qlib
- scikit-learn

## 注意事项

- 预测结果仅供参考，不构成投资建议
- 加密货币市场波动较大，预测准确性受多种因素影响 