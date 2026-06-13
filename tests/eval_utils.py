"""
eval_utils.py — 底层工具函数
=======================
编码检测 / DTW距离 / 价格曲线提取 / 常量配置
"""
import os
import numpy as np
import pandas as pd

# ============================================================
# 常量
# ============================================================
N_INTERVALS = 48       # 5分钟K线数
RANDOM_SEED = 42

VALID_CAPITAL_TYPES = {'游资', '量化', '散户'}
VALID_INTENTIONS = {'买入', '卖出', 'T0交易'}

_ENCODING_CANDIDATES = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']


# ============================================================
# 编码检测
# ============================================================

def detect_encoding(file_path: str, sample_bytes: int = 4096) -> str:
    """
    自动检测文件编码。
    UTF-8-sig 优先（特征文件），GBK 兜底（原始数据中文列名）。
    """
    with open(file_path, 'rb') as f:
        raw = f.read(sample_bytes)
    for enc in _ENCODING_CANDIDATES:
        try:
            decoded = raw.decode(enc)
            if '\ufffd' not in decoded:
                return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return 'gbk'


# ============================================================
# DTW 距离
# ============================================================

def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    纯 numpy 实现的 DTW 距离。
    对于 48 点序列 O(n²)=2304，极快，无需额外依赖。
    """
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(x[i - 1]) - float(y[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


# ============================================================
# 价格曲线提取（用于 DTW）
# ============================================================

def load_price_curve(stock_code: str, raw_dir: str,
                     n_intervals: int = N_INTERVALS) -> np.ndarray:
    """
    从原始行情.csv中提取价格曲线（5分钟K线均价）。
    返回 shape=(n_intervals,) 的 float64 数组，缺失时段填 0。
    """
    stock_dir = os.path.join(raw_dir, stock_code)
    quotes_path = os.path.join(stock_dir, '行情.csv')
    if not os.path.exists(quotes_path):
        return np.zeros(n_intervals, dtype=np.float64)

    try:
        enc = detect_encoding(quotes_path)
        df = pd.read_csv(quotes_path, encoding=enc)

        # 时间解析 (HHMMSSmmm 格式)
        time_str = df['时间'].astype(str).str.zfill(9)
        hour = time_str.str[:2].astype(int)
        minute = time_str.str[2:4].astype(int)
        minute_of_day = hour * 60 + minute

        # 过滤交易时段: 9:30-11:30, 13:00-15:00
        trading_mask = (
            ((hour == 9) & (minute >= 30)) |
            (hour == 10) |
            ((hour == 11) & (minute <= 30)) |
            (hour == 13) |
            (hour == 14)
        )
        df = df.loc[trading_mask].copy()

        if df.empty:
            return np.zeros(n_intervals, dtype=np.float64)

        # 提取成交价
        price_col = '成交价' if '成交价' in df.columns else 'price'
        if price_col not in df.columns:
            return np.zeros(n_intervals, dtype=np.float64)

        df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        df = df[df['price'] > 0]

        if df.empty:
            return np.zeros(n_intervals, dtype=np.float64)

        # 按 5 分钟桶分组取均值
        df['bucket'] = ((minute_of_day.loc[df.index] - 570) // 5).clip(0, n_intervals - 1).astype(int)
        grouped = df.groupby('bucket')['price'].mean().reset_index()
        curve = np.zeros(n_intervals, dtype=np.float64)
        for _, row in grouped.iterrows():
            idx = int(row['bucket'])
            if 0 <= idx < n_intervals:
                curve[idx] = float(row['price'])

        return curve

    except Exception:
        return np.zeros(n_intervals, dtype=np.float64)
