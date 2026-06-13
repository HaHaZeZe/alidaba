"""
数据预处理模块
负责读取原始CSV数据、解析时间、累计值转逐笔、清洗异常值
"""
from typing import Any, cast

import pandas as pd
import numpy as np
import os


# 编码自动检测候选列表（按常见优先级排列）
_ENCODING_CANDIDATES = ['gbk', 'gb2312', 'gb18030', 'utf-8-sig', 'utf-8', 'latin1']


def _detect_encoding(file_path: str, sample_bytes: int = 4096) -> str:
    """
    自动检测 CSV 文件的编码。
    依次尝试候选编码列表，优先选择能成功解码且不含乱码的编码。
    回退到 gbk。
    """
    with open(file_path, 'rb') as f:
        raw = f.read(sample_bytes)

    for enc in _ENCODING_CANDIDATES:
        try:
            decoded = raw.decode(enc)
            # 简单启发式：解码后不含常见的替换字符（�）或过多不可打印字符
            if '\ufffd' not in decoded:
                # 额外检查：中文文本应包含常见中文字符或ASCII
                return enc
        except (UnicodeDecodeError, LookupError):
            continue

    # 全失败则回退 gbk
    return 'gbk'


def load_stock_data(stock_code: str, raw_dir: str) -> dict:
    """
    加载单只股票的全部原始数据（自动检测文件编码）
    返回 dict: {'quotes': DataFrame, 'trades': DataFrame, 'orders': DataFrame}
    """
    stock_dir = os.path.join(raw_dir, stock_code)
    data: dict[str, Any] = {}

    # 行情快照
    quotes_path = os.path.join(stock_dir, '行情.csv')
    if os.path.exists(quotes_path):
        data['quotes'] = pd.read_csv(quotes_path, encoding=_detect_encoding(quotes_path))
    else:
        data['quotes'] = None

    # 逐笔成交
    trades_path = os.path.join(stock_dir, '逐笔成交.csv')
    if os.path.exists(trades_path):
        data['trades'] = pd.read_csv(trades_path, encoding=_detect_encoding(trades_path))
    else:
        data['trades'] = None

    # 逐笔委托
    orders_path = os.path.join(stock_dir, '逐笔委托.csv')
    if os.path.exists(orders_path):
        data['orders'] = pd.read_csv(orders_path, encoding=_detect_encoding(orders_path))
    else:
        data['orders'] = None

    return data


def preprocess_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理行情快照数据：
    1. 解析时间字段提取 时/分/秒
    2. 累计值(当日累计成交量/成交额/成交笔数)转逐笔 diff
    3. 计算价格变动
    4. 解析十档盘口
    """
    df = df.copy()

    # --- 时间解析：时间字段为 HHMMSSmmm 格式（如 91401000 = 09:14:01.000）---
    time_str = df['时间'].astype(str).str.zfill(9)
    df['hour'] = time_str.str[:2].astype(int)
    df['minute'] = time_str.str[2:4].astype(int)
    df['second'] = time_str.str[4:6].astype(int)
    df['millisecond'] = time_str.str[6:9].astype(int)

    # 综合时间戳（毫秒级，用于排序和间隔计算）
    df['timestamp_ms'] = (
        df['hour'] * 3600000 +
        df['minute'] * 60000 +
        df['second'] * 1000 +
        df['millisecond']
    )

    # --- 日期标准化 ---
    df['transaction_date'] = df['自然日'].astype(str)

    # --- 累计值转逐笔量 ---
    # 当日累计成交量 / 当日成交额 / 成交笔数 均为当日累计值，需 diff 得到逐笔量
    df = df.sort_values('timestamp_ms').reset_index(drop=True)

    for col, tick_col in [
        ('当日累计成交量', 'tick_volume'),
        ('当日成交额', 'tick_amount'),
        ('成交笔数', 'tick_transactions')
    ]:
        if col in df.columns:
            df[tick_col] = df[col].diff().fillna(0).clip(lower=0)

    # --- 价格变动 ---
    df['price'] = df['成交价']
    df['price_change'] = df['price'].diff().fillna(0)

    # --- 盘口解析：申买/申卖 1-10 档 ---
    for i in range(1, 11):
        bid_price_col = f'申买价{i}'
        bid_vol_col = f'申买量{i}'
        ask_price_col = f'申卖价{i}'
        ask_vol_col = f'申卖量{i}'
        for c in [bid_price_col, bid_vol_col, ask_price_col, ask_vol_col]:
            if c in df.columns:
                series: pd.Series = df[c]
                df[c] = cast(pd.Series, pd.to_numeric(series, errors='coerce')).fillna(0)

    # 最优买卖价
    df['best_bid'] = df['申买价1']
    df['best_ask'] = df['申卖价1']
    df['best_bid_vol'] = df['申买量1']
    df['best_ask_vol'] = df['申卖量1']

    # 买卖价差
    df['spread'] = df['best_ask'] - df['best_bid']
    df.loc[(df['best_bid'] <= 0) | (df['best_ask'] <= 0), 'spread'] = np.nan
    df['spread_pct'] = df['spread'] / (df['best_bid'] + 1e-8)

    # 加权买卖价
    for col_name, target_name in [
        ('加权平均叫买价', 'weighted_bid'),
        ('加权平均叫卖价', 'weighted_ask'),
    ]:
        if col_name in df.columns:
            col_series: pd.Series = df[col_name]
            df[target_name] = cast(pd.Series, pd.to_numeric(col_series, errors='coerce')).fillna(0)
        else:
            df[target_name] = 0.0

    df['weighted_spread'] = df['weighted_ask'] - df['weighted_bid']

    # 总买卖量
    for col_name, target_name in [
        ('叫买总量', 'total_bid_vol'),
        ('叫卖总量', 'total_ask_vol'),
    ]:
        if col_name in df.columns:
            col_series: pd.Series = df[col_name]
            df[target_name] = cast(pd.Series, pd.to_numeric(col_series, errors='coerce')).fillna(0)
        else:
            df[target_name] = 0.0

    # --- 异常值过滤 ---
    df = cast(pd.DataFrame, df.loc[(df['price'] > 0) & (df['tick_volume'] >= 0) & (df['tick_amount'] >= 0)])

    return df


def preprocess_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理逐笔成交数据：
    1. 解析时间
    2. 提取BS标志（买卖方向）
    3. 标准化价格和数量字段
    """
    df = df.copy()

    # 时间解析
    time_str = df['时间'].astype(str).str.zfill(9)
    df['hour'] = time_str.str[:2].astype(int)
    df['minute'] = time_str.str[2:4].astype(int)
    df['second'] = time_str.str[4:6].astype(int)
    df['millisecond'] = time_str.str[6:9].astype(int)
    df['timestamp_ms'] = (
        df['hour'] * 3600000 +
        df['minute'] * 60000 +
        df['second'] * 1000 +
        df['millisecond']
    )

    df['trade_price'] = pd.to_numeric(df['成交价格'], errors='coerce')
    df['trade_volume'] = pd.to_numeric(df['成交数量'], errors='coerce')
    df['trade_amount'] = df['trade_price'] * df['trade_volume']
    df['bs_flag'] = df['BS标志']  # B=买, S=卖

    # 过滤异常
    df = cast(pd.DataFrame, df.loc[(df['trade_price'] > 0) & (df['trade_volume'] > 0)])
    df = df.sort_values('timestamp_ms').reset_index(drop=True)

    return df


def preprocess_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理逐笔委托数据：
    1. 解析时间
    2. 识别撤单（委托类型 D = 撤单）
    3. 标准化字段
    """
    df = df.copy()

    # 时间解析
    time_str = df['时间'].astype(str).str.zfill(9)
    df['hour'] = time_str.str[:2].astype(int)
    df['minute'] = time_str.str[2:4].astype(int)
    df['second'] = time_str.str[4:6].astype(int)
    df['millisecond'] = time_str.str[6:9].astype(int)
    df['timestamp_ms'] = (
        df['hour'] * 3600000 +
        df['minute'] * 60000 +
        df['second'] * 1000 +
        df['millisecond']
    )

    df['order_price'] = pd.to_numeric(df['委托价格'], errors='coerce')
    df['order_volume'] = pd.to_numeric(df['委托数量'], errors='coerce')
    df['order_type'] = df['委托类型']  # A=委托, D=撤单
    df['order_code'] = df['委托代码']  # B=买, S=卖

    df = df.sort_values('timestamp_ms').reset_index(drop=True)

    return df


def load_and_preprocess_stock(stock_code: str, raw_dir: str) -> dict[str, Any]:
    """
    加载并预处理单只股票的全部数据
    返回预处理后的数据字典
    """
    raw_data = load_stock_data(stock_code, raw_dir)

    result: dict[str, Any] = {'stock_code': stock_code}

    if raw_data.get('quotes') is not None and len(raw_data['quotes']) > 0:
        result['quotes'] = preprocess_quotes(raw_data['quotes'])
        if len(result['quotes']) > 0:
            result['transaction_date'] = result['quotes']['transaction_date'].iloc[0]
    else:
        result['quotes'] = None

    if raw_data.get('trades') is not None and len(raw_data['trades']) > 0:
        result['trades'] = preprocess_trades(raw_data['trades'])
    else:
        result['trades'] = None

    if raw_data.get('orders') is not None and len(raw_data['orders']) > 0:
        result['orders'] = preprocess_orders(raw_data['orders'])
    else:
        result['orders'] = None

    return result
