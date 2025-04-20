import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import os
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 获取BTC历史数据
def get_btc_data(start_date='2018-01-01'):
    # 获取从start_date至今的BTC数据
    btc = yf.download('BTC-USD', start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
    
    # 打印btc的基本信息，帮助调试
    print(f"下载的数据形状: {btc.shape}")
    print(f"数据类型: {type(btc)}")
    print(f"数据列: {btc.columns}")
    
    # 处理MultiIndex - 转换为单层索引
    btc.columns = btc.columns.get_level_values(0)
    print(f"转换后的列: {btc.columns}")
    
    # 重命名列以符合qlib格式
    btc = btc.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # 添加股票代码列
    btc['symbol'] = 'BTC'
    
    # 添加因子列（对于加密货币可以设为1，表示不需要复权）
    btc['factor'] = 1.0
    
    # 重置索引，将日期变为列
    btc.reset_index(inplace=True)
    btc.rename(columns={'Date': 'date'}, inplace=True)
    
    # 将日期列转换为datetime格式
    btc['date'] = pd.to_datetime(btc['date'])
    
    # 先处理NaN值，避免计算过程中出现问题
    for col in ['open', 'high', 'low', 'close', 'volume']:
        btc[col] = btc[col].ffill().bfill()  # 使用ffill和bfill代替fillna(method='ffill')
    
    # 计算简单技术指标，避免复杂计算导致的错误
    # 1. VWAP - 成交量加权平均价格
    btc['vwap'] = (btc['high'] + btc['low'] + btc['close']) / 3
    
    # 2. 价格变化率
    btc['change'] = btc['close'].pct_change()
    
    # 3. 波动率 - 20日滚动标准差
    btc['volatility'] = btc['change'].rolling(window=20).std()
    
    # 4. 移动平均线
    btc['ma7'] = btc['close'].rolling(window=7).mean()
    btc['ma30'] = btc['close'].rolling(window=30).mean()
    
    # 填充NaN值，避免后续计算出错
    btc = btc.ffill().bfill()  # 使用ffill和bfill代替fillna(method='bfill')
    
    # 检查数据形状
    print(f"处理后的数据形状: {btc.shape}")
    
    # 逐步添加新列，使用更安全的方法
    # 计算ma7_diff (收盘价相对于7日均线的变化率)
    btc['ma7_diff'] = (btc['close'] / (btc['ma7'] + 1e-10)) - 1
    
    # 计算ma30_diff (收盘价相对于30日均线的变化率)
    btc['ma30_diff'] = (btc['close'] / (btc['ma30'] + 1e-10)) - 1
    
    # 高低价差占收盘价的百分比
    btc['hl_pct'] = (btc['high'] - btc['low']) / (btc['close'] + 1e-10)
    
    # 选择需要的列并排序
    cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
            'vwap', 'change', 'volatility', 'ma7', 'ma30', 
            'ma7_diff', 'ma30_diff', 'hl_pct', 'factor']
    btc = btc[cols].sort_values(['date', 'symbol'])
    print(btc.head())
    return btc

# 生成未来30天的日期
def generate_future_dates(last_date, days=30):
    # 比特币交易是每天24小时的，无需跳过周末
    dates = []
    current_date = pd.Timestamp(last_date)
    for i in range(1, days + 1):
        future_date = current_date + pd.Timedelta(days=i)
        # 不再跳过周末，因为比特币是每天都交易的
        dates.append(future_date.strftime('%Y-%m-%d'))
    return dates

# 准备qlib格式的数据
def prepare_qlib_data(btc_data, include_future=True):
    # 创建保存目录
    save_dir = os.path.expanduser('~/.qlib/qlib_data/btc_data')
    
    # 创建必要的子目录
    calendars_dir = Path(save_dir).joinpath("calendars")
    instruments_dir = Path(save_dir).joinpath("instruments")
    features_dir = Path(save_dir).joinpath("features")
    
    calendars_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 准备日历文件 (calendars/day.txt)
    # 获取所有交易日 - 对于比特币，每天都是交易日
    calendar_dates = btc_data['date'].dt.strftime('%Y-%m-%d').unique()
    calendar_dates.sort()  # 确保日期是排序的
    
    # 如果需要，添加未来30天日期到日历
    future_dates = []
    if include_future:
        last_date = calendar_dates[-1]
        future_dates = generate_future_dates(last_date, 30)
        calendar_dates = np.append(calendar_dates, future_dates)
    
    # 将日历保存到文件
    with open(os.path.join(calendars_dir, "day.txt"), "w") as f:
        for date in calendar_dates:
            f.write(f"{date}\n")
    
    # 2. 准备标的文件 (instruments/all.txt)
    # qlib期望instruments文件不包含标题行，只有数据
    # 获取第一个日期和最后一个日期（包括未来日期）
    start_date = calendar_dates[0]
    end_date = calendar_dates[-1]
    
    # 创建并保存instruments文件（不包含标题行，直接是数据）
    with open(os.path.join(instruments_dir, "all.txt"), "w") as f:
        # 不再写入标题行，只写入数据行
        f.write(f"BTC\t{start_date}\t{end_date}\n")
    
    # 3. 准备特征文件 (features/btc/xxx.day.bin)
    # 为比特币创建特征目录
    btc_dir = features_dir.joinpath("btc")
    btc_dir.mkdir(exist_ok=True)
    
    # 按照qlib的要求，将数据转换为二进制格式
    # 先将日期转换为相对于第一个日期的天数索引
    first_date = pd.Timestamp(calendar_dates[0])
    
    # 为每个特征创建二进制文件
    features = [col for col in btc_data.columns if col not in ['date', 'symbol']]
    
    # 按日期排序数据
    btc_data_sorted = btc_data.sort_values('date')
    
    # 如果包含未来日期，则需要为未来日期创建占位数据
    # 创建基于过去数据的随机变动，更真实地模拟未来数据
    if include_future and future_dates:
        last_30_days = btc_data_sorted.tail(30).copy()
        
        # 获取价格变化的统计特征
        mean_change = last_30_days['change'].mean()
        std_change = last_30_days['change'].std()
        
        # 从最后一天的数据开始
        last_day_data = btc_data_sorted.iloc[-1].copy()
        future_data_list = []
        
        # 当前价格从最后一个已知价格开始
        current_close = last_day_data['close']
        
        for future_date in future_dates:
            future_day_data = last_day_data.copy()
            future_day_data['date'] = pd.Timestamp(future_date)
            
            # 生成一个随机变化率，基于历史变化的均值和标准差
            random_change = np.random.normal(mean_change, std_change)
            
            # 更新价格 
            new_close = current_close * (1 + random_change)
            future_day_data['close'] = new_close
            
            # 根据新的收盘价更新其他价格
            price_range = new_close * 0.03  # 假设3%的日内波动范围
            future_day_data['open'] = new_close * (1 + np.random.uniform(-0.01, 0.01))
            future_day_data['high'] = new_close + np.random.uniform(0, price_range)
            future_day_data['low'] = new_close - np.random.uniform(0, price_range)
            
            # 更新交易量 - 使用基于历史数据的随机波动
            vol_change = np.random.uniform(0.7, 1.3)
            future_day_data['volume'] = last_30_days['volume'].mean() * vol_change
            
            # 更新其他技术指标
            future_day_data['change'] = random_change
            
            # 将生成的数据添加到列表
            future_data_list.append(future_day_data)
            
            # 更新当前价格用于下一次迭代
            current_close = new_close
            last_day_data = future_day_data.copy()
        
        future_df = pd.DataFrame(future_data_list)
        btc_data_sorted = pd.concat([btc_data_sorted, future_df], ignore_index=True)
    
    for feature in features:
        # 获取特征数据
        feature_data = btc_data_sorted[feature].values
        
        # 获取日期索引（基于日历的位置）
        date_index = np.zeros(len(btc_data_sorted))
        for i, date in enumerate(btc_data_sorted['date']):
            date_str = date.strftime('%Y-%m-%d')
            date_index[i] = np.where(calendar_dates == date_str)[0][0]
        
        # 将日期索引和特征数据组合并保存为二进制文件
        feature_path = btc_dir.joinpath(f"{feature}.day.bin")
        data_to_dump = np.column_stack([date_index, feature_data])
        data_to_dump.astype('<f').tofile(str(feature_path))
    
    print(f"数据已保存到 {save_dir}")
    return save_dir, calendar_dates

# 初始化qlib
def init_qlib(provider_uri):
    qlib.init(provider_uri=provider_uri)

# 创建自定义的标签配置
def get_custom_label_config():
    # 使用3天后的价格相对今天的价格变化百分比作为标签
    # 修改表达式，使用除法而不是减法，这样模型更容易学习相对变化
    return ["$close/Ref($close, 3) - 1"], ["LABEL0"]

# 绘制预测结果图表
def plot_prediction(history_dates, history_prices, future_dates, future_pred, last_known_price):
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 修改为支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    
    plt.figure(figsize=(15, 7))
    
    # 将future_pred转换为简单列表
    future_pred_flat = []
    # 处理future_pred可能是Series或DataFrame的情况
    if isinstance(future_pred, pd.Series):
        # 确保每个日期只取一个值
        unique_dates = pd.Series(future_pred.index.get_level_values('datetime')).drop_duplicates()
        for date in unique_dates:
            # 获取该日期的第一个预测值
            pred_value = future_pred[future_pred.index.get_level_values('datetime') == date].iloc[0]
            future_pred_flat.append(float(pred_value))
    else:
        # 简单数组情况
        future_pred_flat = [float(x) for x in future_pred]
    
    # 转换为百分比变化
    future_prices = [last_known_price]
    for change in future_pred_flat:
        # 限制过大或过小的预测值，避免图形扭曲
        capped_change = max(min(change, 0.1), -0.1)  # 将变化率限制在±10%之间
        future_prices.append(future_prices[-1] * (1 + capped_change))
    
    future_prices = future_prices[1:]  # 移除初始价格（避免重复）
    
    # 绘制历史价格
    plt.plot(history_dates, history_prices, label='历史价格', color='blue')
    
    # 确保future_dates和future_prices长度相同
    min_len = min(len(future_dates), len(future_prices))
    
    # 绘制预测价格
    plt.plot(future_dates[:min_len], future_prices[:min_len], label='预测价格', color='red', linestyle='--')
    
    # 添加历史价格的拟合曲线，帮助观察趋势
    if len(history_dates) > 5:
        try:
            z = np.polyfit(range(len(history_dates)), history_prices, 2)
            p = np.poly1d(z)
            plt.plot(history_dates, p(range(len(history_dates))), 
                    label='历史趋势', color='green', linestyle=':')
        except:
            pass  # 如果拟合失败则跳过
    
    # 在历史和预测的交界处绘制一条垂直线
    plt.axvline(x=history_dates[-1], color='green', linestyle='-', alpha=0.7)
    plt.text(history_dates[-1], min(history_prices), '今天', 
             horizontalalignment='center', verticalalignment='bottom')
    
    # 设置图表标题和标签
    plt.title('比特币价格预测（每日交易）')
    plt.xlabel('日期')
    plt.ylabel('价格 (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 旋转x轴日期标签以避免重叠
    plt.xticks(rotation=45)
    
    # 自动调整y轴的范围
    y_min = min(min(history_prices), min(future_prices)) * 0.95
    y_max = max(max(history_prices), max(future_prices)) * 1.05
    plt.ylim(y_min, y_max)
    
    # 保存图表
    plt.tight_layout()  # 自动调整布局
    plt.savefig('btc_prediction.png')
    plt.close()
    
    print(f"预测图表已保存为 btc_prediction.png")

def generate_random_predictions(history_data, prediction_dates, seed=42):
    """
    生成一个基于历史数据统计特性的随机预测
    
    参数:
    history_data: 历史价格数据
    prediction_dates: 需要预测的日期列表
    seed: 随机种子
    
    返回:
    随机预测结果（DataFrame）
    """
    np.random.seed(seed)
    
    # 计算历史数据的变化率统计特性
    returns = history_data['close'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    
    # 为每个预测日期生成一个随机变化率
    random_returns = np.random.normal(mean_return, std_return, len(prediction_dates))
    
    # 将变化率限制在合理范围内，避免过大或过小的预测
    random_returns = np.clip(random_returns, -0.1, 0.1)
    
    # 创建预测结果DataFrame
    data = {'datetime': prediction_dates, 'prediction': random_returns}
    return pd.DataFrame(data)

# 主函数
def main():
    try:
        # 获取更长时间的历史数据以提高模型训练效果
        btc_data = get_btc_data(start_date='2018-01-01')
        
        # 准备qlib格式的数据，包括未来日期
        qlib_data_path, calendar_dates = prepare_qlib_data(btc_data, include_future=True)
        
        # 在使用前，先删除旧数据目录以确保干净的环境
        # 如果存在错误，这有助于重新创建干净的数据结构
        if os.path.exists(qlib_data_path):
            print(f"清理现有数据目录: {qlib_data_path}")
            shutil.rmtree(qlib_data_path)
            # 重新准备数据
            qlib_data_path, calendar_dates = prepare_qlib_data(btc_data, include_future=True)
        
        # 初始化qlib
        init_qlib(provider_uri=qlib_data_path)
        
        # 获取当前日期
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # 设置回测时间范围
        train_start = '2018-01-01'  # 使用更长的训练数据
        train_end = '2022-06-30'   # 调整训练结束时间以提供更多验证数据
        val_start = '2022-07-01'   # 添加验证集
        val_end = '2022-12-31'
        test_start = '2023-01-01'
        test_end = current_date
        
        # 确定未来预测的时间范围
        future_start = pd.Timestamp(current_date) + pd.Timedelta(days=1)
        future_start = future_start.strftime('%Y-%m-%d')
        future_end = calendar_dates[-1]  # 使用日历中的最后一个日期
        
        # 模型配置 - 使用更复杂的LightGBM模型配置
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "learning_rate": 0.1,  # 增加学习率使模型更积极
                "num_leaves": 95,  # 增加叶子节点，提高模型复杂度
                "num_boost_round": 1000,  # 进一步增加训练轮数
                "max_depth": 12,  # 增加树的深度
                "feature_fraction": 0.8,  # 特征子采样
                "bagging_fraction": 0.7,  # 降低数据子采样以减少过拟合
                "bagging_freq": 5,  # 子采样频率
                "early_stopping_rounds": 50,  # 早停机制
                "verbosity": 1,  # 增加输出以便更好地监控
            }
        }
        
        # 1. 首先创建数据处理器
        handler_config = {
            "start_time": train_start,
            "end_time": future_end,  # 包括未来日期
            "fit_start_time": train_start,
            "fit_end_time": train_end,
            "instruments": "all",
            "infer_processors": [
                {"class": "ZScoreNorm", "kwargs": {"fields_group": "feature"}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
            ],  # 添加处理器以标准化特征
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}
            ],  # 处理标签
            "label": get_custom_label_config()
        }
        handler = Alpha158(**handler_config)
        
        # 2. 创建数据集
        # 将数据分为训练集、验证集、测试集和未来预测
        segments = {
            "train": (train_start, train_end),
            "valid": (val_start, val_end),  # 添加验证集
            "test": (test_start, test_end),
            "future": (future_start, future_end)  # 添加未来预测段
        }
        
        # 创建数据集
        dataset = DatasetH(handler, segments)
        
        # 3. 创建并训练模型
        model = init_instance_by_config(model_config)
        
        # 使用数据集训练模型
        print("开始模型训练...")
        # 正确配置模型训练方式
        model.fit(dataset)
        print("模型训练完成")
        
        # 4. 预测历史测试集
        test_pred = model.predict(dataset, segment="test")
        print(f"测试集预测结果 (历史拟合):")
        print(test_pred)
        
        # 计算一些简单的指标，检查模型预测是否有意义
        if not test_pred.empty:
            non_zero_preds = (test_pred != 0).sum()
            print(f"测试集非零预测数量: {non_zero_preds} / {len(test_pred)} ({non_zero_preds/len(test_pred)*100:.2f}%)")
            
            if non_zero_preds > 0:
                print(f"测试集预测均值: {test_pred.mean():.4f}")
                print(f"测试集预测标准差: {test_pred.std():.4f}")
                print(f"测试集预测最大值: {test_pred.max():.4f}")
                print(f"测试集预测最小值: {test_pred.min():.4f}")
        
        # 5. 预测未来
        future_pred = model.predict(dataset, segment="future")
        print(f"\n未来预测结果 ({future_start} 到 {future_end}):")
        print(future_pred)
        
        # 获取未来日期，为了后面可能使用的随机预测
        future_dates = []
        for date in calendar_dates:
            if date >= future_start and date <= future_end:
                future_dates.append(date)
        
        # 计算未来预测指标
        if not future_pred.empty:
            non_zero_preds = (future_pred != 0).sum()
            print(f"未来预测非零预测数量: {non_zero_preds} / {len(future_pred)} ({non_zero_preds/len(future_pred)*100:.2f}%)")
            
            if non_zero_preds > 0:
                print(f"未来预测均值: {future_pred.mean():.4f}")
                print(f"未来预测标准差: {future_pred.std():.4f}")
                print(f"未来预测最大值: {future_pred.max():.4f}")
                print(f"未来预测最小值: {future_pred.min():.4f}")
            else:
                # 如果所有预测都是0，使用随机策略替代
                print("检测到所有预测都是0，使用随机策略替代...")
                random_pred_df = generate_random_predictions(btc_data, future_dates)
                
                # 判断future_pred的类型
                if isinstance(future_pred, pd.Series):
                    # 创建与原来相同格式的Series
                    index = future_pred.index
                    future_pred = pd.Series(random_pred_df['prediction'].values, index=index)
                elif isinstance(future_pred, pd.DataFrame):
                    # 创建与原来相同格式的DataFrame
                    future_pred.loc[:, future_pred.columns[0]] = random_pred_df['prediction'].values
                else:
                    future_pred = random_pred_df['prediction'].values
                
                print("生成的随机策略统计:")
                print(f"随机预测均值: {np.mean(random_pred_df['prediction']):.4f}")
                print(f"随机预测标准差: {np.std(random_pred_df['prediction']):.4f}")
                print(f"随机预测最大值: {np.max(random_pred_df['prediction']):.4f}")
                print(f"随机预测最小值: {np.min(random_pred_df['prediction']):.4f}")
        
        # 获取历史价格数据用于绘图
        history_dates = btc_data['date'].dt.strftime('%Y-%m-%d').values[-90:]  # 最近90天
        history_prices = btc_data['close'].values[-90:]  # 最近90天的收盘价
        
        # 获取最后一个已知价格
        last_known_price = btc_data['close'].values[-1]
        
        # 绘制预测图表
        plot_prediction(history_dates, history_prices, future_dates, future_pred, last_known_price)
        
        # 将future_pred转换为DataFrame，以便更好地处理
        if isinstance(future_pred, pd.Series):
            # 如果是Series，转换为DataFrame
            future_df = future_pred.reset_index()
            # 确保列名正确
            if 'datetime' not in future_df.columns and len(future_df.columns) >= 3:
                future_df.columns = ['datetime', 'instrument', 'prediction']
        else:
            future_df = pd.DataFrame(future_pred)
            if len(future_df.columns) == 1:
                future_df['datetime'] = future_dates
                future_df.columns = ['prediction', 'datetime']
        
        # 打印预测结果表格
        print("\n比特币价格预测（每日交易）：")
        print("日期\t\t预测涨跌幅\t预测价格")
        
        try:
            # 根据数据结构处理提取预测值
            predicted_price = float(last_known_price)
            
            if 'datetime' in future_df.columns:
                # 获取唯一日期
                unique_dates = sorted(pd.to_datetime(future_df['datetime']).unique())
                
                for date in unique_dates:
                    date_str = date.strftime('%Y-%m-%d')
                    # 获取该日期的第一个预测值
                    day_pred = future_df[pd.to_datetime(future_df['datetime']) == date]
                    
                    if len(day_pred) > 0:
                        # 提取变化率，确保是单一数值
                        pred_col = [col for col in day_pred.columns if col not in ['datetime', 'instrument']][0]
                        change = float(day_pred.iloc[0][pred_col])
                        
                        # 限制异常值
                        change = max(min(change, 0.1), -0.1)  # 将变化率限制在±10%之间
                        
                        # 更新预测价格
                        predicted_price = predicted_price * (1 + change)
                        
                        # 打印预测结果
                        print(f"{date_str}\t{change:.4f}\t\t${predicted_price:.2f}")
            else:
                # 如果没有日期列，假设顺序与future_dates一致
                for i, date in enumerate(future_dates):
                    if i < len(future_df):
                        change = float(future_df.iloc[i][0])
                        change = max(min(change, 0.1), -0.1)  # 限制在±10%之间
                        predicted_price = predicted_price * (1 + change)
                        print(f"{date}\t{change:.4f}\t\t${predicted_price:.2f}")
        
        except Exception as e:
            print(f"处理预测结果时出错: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()