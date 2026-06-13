"""
特征工程模块
从预处理后的数据中提取参考特征集的所有特征
每个股票处理完后立即保存特征到 data/features/
"""
from typing import Any

import pandas as pd
import numpy as np
import os


# ============================================================
# OSS 大单分级特征
# ============================================================
def extract_oss_features(quotes: pd.DataFrame) -> dict:
    """基于逐笔成交量进行大单分级"""
    f = {}
    if quotes is None or quotes.empty or len(quotes.columns) == 0:
        return {k: 0.0 for k in [
            'oss_mega_amount_pct', 'oss_mega_count_pct',
            'oss_large_amount_pct', 'oss_large_count_pct',
            'oss_medium_amount_pct', 'oss_medium_count_pct',
            'oss_small_amount_pct', 'oss_small_count_pct',
        ]}
    group = quotes[quotes['tick_volume'] > 0].copy()
    if len(group) == 0:
        return {k: 0.0 for k in [
            'oss_mega_amount_pct', 'oss_mega_count_pct',
            'oss_large_amount_pct', 'oss_large_count_pct',
            'oss_medium_amount_pct', 'oss_medium_count_pct',
            'oss_small_amount_pct', 'oss_small_count_pct',
        ]}

    total_amt = group['tick_amount'].sum() + 1e-8
    total_cnt = len(group)

    # 阈值：超大单≥50000股, 大单≥10000股, 中单≥1000股, 小单<1000股
    mega = group['tick_volume'] >= 50000
    large = (group['tick_volume'] >= 10000) & (group['tick_volume'] < 50000)
    medium = (group['tick_volume'] >= 1000) & (group['tick_volume'] < 10000)
    small = group['tick_volume'] < 1000

    f['oss_mega_amount_pct'] = group.loc[mega, 'tick_amount'].sum() / total_amt
    f['oss_mega_count_pct'] = mega.sum() / total_cnt
    f['oss_large_amount_pct'] = group.loc[large, 'tick_amount'].sum() / total_amt
    f['oss_large_count_pct'] = large.sum() / total_cnt
    f['oss_medium_amount_pct'] = group.loc[medium, 'tick_amount'].sum() / total_amt
    f['oss_medium_count_pct'] = medium.sum() / total_cnt
    f['oss_small_amount_pct'] = group.loc[small, 'tick_amount'].sum() / total_amt
    f['oss_small_count_pct'] = small.sum() / total_cnt

    return f


# ============================================================
# TRD 逐笔交易结构特征
# ============================================================
def extract_trd_features(quotes: pd.DataFrame) -> dict:
    """交易结构特征"""
    f = {}
    if quotes is None or quotes.empty or len(quotes.columns) == 0:
        return {k: 0.0 for k in [
            'trd_avg_trade_size', 'trd_avg_trade_amount',
            'trd_trade_volume_std', 'trd_change_percent',
            'trd_range_percent', 'trd_large_vol_ratio',
        ]}
    total_vol = quotes['tick_volume'].sum() + 1e-8
    total_amt = quotes['tick_amount'].sum() + 1e-8
    total_txn = quotes['tick_transactions'].sum() + 1e-8

    f['trd_avg_trade_size'] = total_vol / total_txn
    f['trd_avg_trade_amount'] = total_amt / total_txn

    # 成交量标准差
    pos_vol = quotes.loc[quotes['tick_volume'] > 0, 'tick_volume']
    f['trd_trade_volume_std'] = pos_vol.std() if len(pos_vol) > 1 else 0.0

    # 涨跌幅（使用最后一条快照的成交价 vs 前收盘）
    prev_close_col = quotes['前收盘'] if '前收盘' in quotes.columns else None
    if prev_close_col is not None and prev_close_col.iloc[0] > 0:
        prev_close = prev_close_col.iloc[0]
        f['trd_change_percent'] = (quotes['price'].iloc[-1] - prev_close) / prev_close
        high = quotes['最高价'].max()
        low = quotes['最低价'].min()
        f['trd_range_percent'] = (high - low) / prev_close if low > 0 else 0.0
    else:
        # 无有效前收盘价：涨跌幅置0，振幅用首条价格做参考
        f['trd_change_percent'] = 0.0
        first_price = quotes['price'].iloc[0]
        if first_price > 0:
            high = quotes['最高价'].max()
            low = quotes['最低价'].min()
            f['trd_range_percent'] = (high - low) / first_price if low > 0 else 0.0
        else:
            f['trd_range_percent'] = 0.0

    # 大单占比（使用逐笔量≥10000作为大单阈值）
    large_vol = quotes.loc[quotes['tick_volume'] >= 10000, 'tick_volume'].sum()
    f['trd_large_vol_ratio'] = large_vol / total_vol

    return f


# ============================================================
# RS 订单时序特征（基于逐笔成交）
# ============================================================
def extract_rs_features(trades: pd.DataFrame) -> dict:
    """订单时序特征：成交间隔变异系数、拆单相似度、订单爆发率"""
    f = {}
    if trades is None or len(trades) < 2:
        return {
            'rs_interval_mean_ms': 0, 'rs_interval_median_ms': 0,
            'rs_interval_cv': 0, 'rs_burst_ratio': 0,
            'rs_buy_interval_cv': 0, 'rs_sell_interval_cv': 0,
            'rs_split_similarity': 0, 'rs_split_run_ratio': 0,
        }

    ts = np.asarray(trades['timestamp_ms'].values)
    intervals = np.diff(ts)
    intervals = intervals[intervals > 0]

    if len(intervals) == 0:
        f['rs_interval_mean_ms'] = 0
        f['rs_interval_median_ms'] = 0
        f['rs_interval_cv'] = 0
    else:
        f['rs_interval_mean_ms'] = float(np.mean(intervals))
        f['rs_interval_median_ms'] = float(np.median(intervals))
        f['rs_interval_cv'] = float(np.std(intervals) / (np.mean(intervals) + 1e-8))

    # 爆发率：间隔 < 100ms 的成交占比
    burst = intervals[intervals < 100]
    f['rs_burst_ratio'] = len(burst) / len(intervals) if len(intervals) > 0 else 0.0

    # 买入/卖出间隔CV
    for flag, key in [('B', 'rs_buy_interval_cv'), ('S', 'rs_sell_interval_cv')]:
        sub = trades[trades['bs_flag'] == flag]
        if len(sub) >= 2:
            sub_intervals = np.diff(np.asarray(sub['timestamp_ms'].values))
            sub_intervals = sub_intervals[sub_intervals > 0]
            if len(sub_intervals) > 0:
                f[key] = float(np.std(sub_intervals) / (np.mean(sub_intervals) + 1e-8))
            else:
                f[key] = 0.0
        else:
            f[key] = 0.0

    # 拆单相似度：连续成交的数量变异系数（CV越低=越相似）
    # 使用 1/(1+CV) 平滑衰减，避免 min(CV,1) 硬截断导致 CV≥1 全归零
    vols = np.asarray(trades['trade_volume'].values)
    if len(vols) >= 2:
        cv = float(np.std(vols) / (np.mean(vols) + 1e-8))
        f['rs_split_similarity'] = 1.0 / (1.0 + cv)
    else:
        f['rs_split_similarity'] = 0.0

    # 拆单连续运行比率：连续同向成交的占比
    bs = trades['bs_flag'].values
    if len(bs) >= 2:
        runs = []
        run_len = 1
        for i in range(1, len(bs)):
            if bs[i] == bs[i-1]:
                run_len += 1
            else:
                if run_len >= 3:
                    runs.append(run_len)
                run_len = 1
        if run_len >= 3:
            runs.append(run_len)
        f['rs_split_run_ratio'] = sum(runs) / len(bs) if runs else 0.0
    else:
        f['rs_split_run_ratio'] = 0.0

    return f


# ============================================================
# CB 撤单系列特征（基于逐笔委托）
# ============================================================
def extract_cb_features(orders: pd.DataFrame) -> dict:
    """撤单特征"""
    f = {}
    if orders is None or len(orders) == 0:
        return {
            'cb_cancel_order_count': 0, 'cb_cancel_order_ratio': 0,
            'cb_cancel_volume_ratio': 0, 'cb_cancel_amount_ratio': 0,
            'cb_fast_cancel_ratio': 0, 'cb_cancel_interval_cv': 0,
            'cb_buy_cancel_ratio': 0, 'cb_sell_cancel_ratio': 0,
        }

    total_orders = len(orders)
    total_vol = orders['order_volume'].sum() + 1e-8
    total_amt = (orders['order_price'] * orders['order_volume']).sum() + 1e-8

    # 撤单 = 委托类型 D
    cancels = orders[orders['order_type'] == 'D']
    adds = orders[orders['order_type'] == 'A']

    f['cb_cancel_order_count'] = len(cancels)
    f['cb_cancel_order_ratio'] = len(cancels) / total_orders if total_orders > 0 else 0.0
    f['cb_cancel_volume_ratio'] = cancels['order_volume'].sum() / total_vol
    f['cb_cancel_amount_ratio'] = (cancels['order_price'] * cancels['order_volume']).sum() / total_amt

    # 快速撤单：撤单时间 - 委托时间 < 1000ms（需要匹配委托-撤单对，这里简化：统计撤单间隔）
    if len(cancels) >= 2:
        cancel_ts = np.asarray(cancels['timestamp_ms'].values)
        cancel_intervals = np.diff(cancel_ts)
        cancel_intervals = cancel_intervals[cancel_intervals > 0]
        if len(cancel_intervals) > 0:
            f['cb_cancel_interval_cv'] = float(np.std(cancel_intervals) / (np.mean(cancel_intervals) + 1e-8))
        else:
            f['cb_cancel_interval_cv'] = 0.0
        # 快速撤单：撤单间隔 < 500ms
        fast = cancel_intervals[cancel_intervals < 500]
        f['cb_fast_cancel_ratio'] = len(fast) / len(cancel_intervals)
    else:
        f['cb_cancel_interval_cv'] = 0.0
        f['cb_fast_cancel_ratio'] = 0.0

    # 买卖撤单率
    for flag, key_add, key_cancel in [
        ('B', 'cb_buy_add_count', 'cb_buy_cancel_count'),
        ('S', 'cb_sell_add_count', 'cb_sell_cancel_count')
    ]:
        add_sub = adds[adds['order_code'] == flag]
        cancel_sub = cancels[cancels['order_code'] == flag]
        f[key_cancel] = len(cancel_sub)
        f[key_add] = len(add_sub)

    buy_total = f.get('cb_buy_add_count', 0) + f.get('cb_buy_cancel_count', 0) + 1e-8
    sell_total = f.get('cb_sell_add_count', 0) + f.get('cb_sell_cancel_count', 0) + 1e-8
    f['cb_buy_cancel_ratio'] = f.get('cb_buy_cancel_count', 0) / buy_total
    f['cb_sell_cancel_ratio'] = f.get('cb_sell_cancel_count', 0) / sell_total
    # 清理中间字段
    for k in ['cb_buy_add_count', 'cb_buy_cancel_count', 'cb_sell_add_count', 'cb_sell_cancel_count']:
        f.pop(k, None)

    return f


# ============================================================
# AP 主动成交特征（基于逐笔成交 BS 标志）
# ============================================================
def extract_ap_features(trades: pd.DataFrame) -> dict:
    """主动成交特征"""
    f = {}
    if trades is None or len(trades) == 0:
        return {
            'ap_active_buy_pct': 0.5, 'ap_active_sell_pct': 0.5,
            'ap_active_net_pct': 0.0, 'ap_unilateral_intensity': 0.0,
            'ap_active_buy_run_max': 0, 'ap_active_sell_run_max': 0,
            'ap_active_volume_buy_pct': 0.5, 'ap_active_volume_sell_pct': 0.5,
        }

    total_amt = trades['trade_amount'].sum() + 1e-8
    total_vol = trades['trade_volume'].sum() + 1e-8

    buy = trades[trades['bs_flag'] == 'B']
    sell = trades[trades['bs_flag'] == 'S']

    buy_amt = buy['trade_amount'].sum()
    sell_amt = sell['trade_amount'].sum()
    buy_vol = buy['trade_volume'].sum()
    sell_vol = sell['trade_volume'].sum()

    f['ap_active_buy_pct'] = buy_amt / total_amt
    f['ap_active_sell_pct'] = sell_amt / total_amt
    f['ap_active_net_pct'] = (buy_amt - sell_amt) / total_amt
    f['ap_active_volume_buy_pct'] = buy_vol / (total_vol + 1e-8)
    f['ap_active_volume_sell_pct'] = sell_vol / (total_vol + 1e-8)

    # 单边强度：|买-卖| / (买+卖)
    f['ap_unilateral_intensity'] = abs(buy_amt - sell_amt) / total_amt

    # 最大连续买入/卖出笔数
    bs = trades['bs_flag'].values
    for flag, key in [('B', 'ap_active_buy_run_max'), ('S', 'ap_active_sell_run_max')]:
        max_run = 0
        cur_run = 0
        for b in bs:
            if b == flag:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        f[key] = max_run

    return f


# ============================================================
# OBP 盘口微观特征（基于行情快照的十档盘口）
# ============================================================
def extract_obp_features(quotes: pd.DataFrame) -> dict:
    """盘口微观特征"""
    f = {}
    if quotes is None or quotes.empty or len(quotes.columns) == 0:
        return {k: 0.0 for k in [
            'obp_spread', 'obp_spread_pct', 'book_imbalance',
            'obp_big_bid_ratio', 'obp_big_ask_ratio',
            'obp_weighted_spread_mean', 'obp_weighted_spread_std',
            'obp_imbalance_mean', 'obp_imbalance_std',
            'obp_imbalance_max', 'obp_imbalance_min',
            'obp_total_bid_mean', 'obp_total_ask_mean', 'obp_bid_ask_ratio',
        ]}

    # 过滤盘口有效数据（最优买卖价均>0）
    valid = quotes[(quotes['best_bid'] > 0) & (quotes['best_ask'] > 0)].copy()
    if len(valid) == 0:
        valid = quotes.copy()

    # --- 方案A：首条快照的精确盘口特征 ---
    first = valid.iloc[0]

    # 最优买卖价差
    f['obp_spread'] = first.get('spread', 0) if not pd.isna(first.get('spread', 0)) else 0.0
    f['obp_spread_pct'] = first.get('spread_pct', 0) if not pd.isna(first.get('spread_pct', 0)) else 0.0

    # 盘口失衡度（首条）：(买量-卖量)/(买量+卖量)
    bid_vol_1 = first.get('best_bid_vol', 0)
    ask_vol_1 = first.get('best_ask_vol', 0)
    f['book_imbalance'] = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1 + 1e-8)

    # 前5档买卖挂单占比
    total_bid_5 = sum(first.get(f'申买量{i}', 0) for i in range(1, 6))
    total_ask_5 = sum(first.get(f'申卖量{i}', 0) for i in range(1, 6))
    total_bid_all = sum(first.get(f'申买量{i}', 0) for i in range(1, 11))
    total_ask_all = sum(first.get(f'申卖量{i}', 0) for i in range(1, 11))
    f['obp_big_bid_ratio'] = total_bid_5 / (total_bid_all + 1e-8)
    f['obp_big_ask_ratio'] = total_ask_5 / (total_ask_all + 1e-8)

    # --- 方案B：全天盘口统计量 ---
    # 加权价差统计
    ws = valid['weighted_spread'].values
    ws = ws[~np.isnan(ws)]
    f['obp_weighted_spread_mean'] = float(np.nanmean(ws)) if len(ws) > 0 else 0.0
    f['obp_weighted_spread_std'] = float(np.nanstd(ws)) if len(ws) > 0 else 0.0

    # 盘口失衡度时间序列（基于叫买总量/叫卖总量）
    tb = np.asarray(valid['total_bid_vol'].values, dtype=float)
    ta = np.asarray(valid['total_ask_vol'].values, dtype=float)
    denom = tb + ta + 1e-8
    imbalance_series = (tb - ta) / denom
    f['obp_imbalance_mean'] = float(np.nanmean(imbalance_series))
    f['obp_imbalance_std'] = float(np.nanstd(imbalance_series))
    f['obp_imbalance_max'] = float(np.nanmax(imbalance_series))
    f['obp_imbalance_min'] = float(np.nanmin(imbalance_series))

    # 委买/委卖总量均值及比值
    f['obp_total_bid_mean'] = float(np.nanmean(tb))
    f['obp_total_ask_mean'] = float(np.nanmean(ta))
    f['obp_bid_ask_ratio'] = float(np.nanmean(tb)) / (float(np.nanmean(ta)) + 1e-8)

    return f


# ============================================================
# PI 日内时段特征
# ============================================================
def extract_pi_features(quotes: pd.DataFrame) -> dict:
    """日内时段特征"""
    f = {}
    if quotes is None or quotes.empty or len(quotes.columns) == 0:
        return {k: 0.0 for k in [
            'pi_open_30min_amount_pct', 'pi_close_10min_amount_pct',
            'pi_time_concentration', 'pi_price_std_pct',
            'pi_vwap_deviation', 'pi_herfindahl_5min',
        ]}
    total_amt = quotes['tick_amount'].sum() + 1e-8

    # 开盘30分钟（9:30-10:00）
    open_mask = (
        ((quotes['hour'] == 9) & (quotes['minute'] >= 30)) |
        ((quotes['hour'] == 10) & (quotes['minute'] == 0))
    )
    open_30 = quotes[open_mask]
    f['pi_open_30min_amount_pct'] = open_30['tick_amount'].sum() / total_amt

    # 尾盘10分钟（14:50-15:00）
    close_mask = (quotes['hour'] == 14) & (quotes['minute'] >= 50)
    close_10 = quotes[close_mask]
    f['pi_close_10min_amount_pct'] = close_10['tick_amount'].sum() / total_amt

    # 时段集中度
    f['pi_time_concentration'] = f['pi_open_30min_amount_pct'] + f['pi_close_10min_amount_pct']

    # 价格波动（价格标准差/均价）
    prices = quotes.loc[quotes['price'] > 0, 'price']
    if len(prices) > 1:
        f['pi_price_std_pct'] = float(prices.std() / (prices.mean() + 1e-8))
    else:
        f['pi_price_std_pct'] = 0.0

    # VWAP偏离度
    if total_amt > 0:
        vwap = (quotes['price'] * quotes['tick_amount']).sum() / total_amt
        last_price = quotes['price'].iloc[-1]
        f['pi_vwap_deviation'] = (last_price - vwap) / (vwap + 1e-8)
    else:
        f['pi_vwap_deviation'] = 0.0

    # 5分钟赫芬达尔集中度
    quotes_copy = quotes.copy()
    quotes_copy['minute_bucket'] = quotes_copy['hour'] * 60 + quotes_copy['minute']
    quotes_copy['minute_5bucket'] = quotes_copy['minute_bucket'] // 5
    bucket_amt = quotes_copy.groupby('minute_5bucket')['tick_amount'].sum()
    if bucket_amt.sum() > 0:
        shares = bucket_amt / bucket_amt.sum()
        f['pi_herfindahl_5min'] = float((shares ** 2).sum())
    else:
        f['pi_herfindahl_5min'] = 0.0

    return f


# ============================================================
# PD 价格发现特征
# ============================================================
def extract_pd_features(quotes: pd.DataFrame) -> dict:
    """价格发现特征"""
    f = {}
    if quotes is None or quotes.empty or len(quotes.columns) == 0:
        return {k: 0.0 for k in ['pd_impact', 'pd_Q1_ratio']}

    # 价格冲击：价格变化绝对值之和 / 成交均价
    price_changes = quotes['price_change'].abs().sum()
    avg_price = quotes.loc[quotes['price'] > 0, 'price'].mean()
    f['pd_impact'] = price_changes / (avg_price + 1e-8) if avg_price > 0 else 0.0

    # Q1比率：基于盘口不平衡的方向性
    valid = quotes[(quotes['total_bid_vol'] > 0) | (quotes['total_ask_vol'] > 0)]
    if len(valid) > 0:
        tb = np.asarray(valid['total_bid_vol'].values, dtype=float)
        ta = np.asarray(valid['total_ask_vol'].values, dtype=float)
        # 买单占比
        bid_ratio = tb / (tb + ta + 1e-8)
        f['pd_Q1_ratio'] = float(np.nanmean(bid_ratio))
    else:
        f['pd_Q1_ratio'] = 0.5

    return f


# ============================================================
# 主特征提取函数
# ============================================================

# 模块级空 DataFrame 单例，用于获取默认特征字段名（避免每次 fallback 重复创建）
_EMPTY_DF = pd.DataFrame()

# 预计算各类特征的默认值字典
_QUOTES_FEATURE_DEFAULTS: dict[str, float] = {
    **{k: 0.0 for k in extract_oss_features(_EMPTY_DF)},
    **{k: 0.0 for k in extract_trd_features(_EMPTY_DF)},
    **{k: 0.0 for k in extract_obp_features(_EMPTY_DF)},
    **{k: 0.0 for k in extract_pi_features(_EMPTY_DF)},
    **{k: 0.0 for k in extract_pd_features(_EMPTY_DF)},
}

_TRADES_FEATURE_DEFAULTS: dict[str, float] = {
    **{k: 0.0 for k in extract_rs_features(_EMPTY_DF)},
    **{k: 0.0 for k in extract_ap_features(_EMPTY_DF)},
}

_ORDERS_FEATURE_DEFAULTS: dict[str, float] = {
    **{k: 0.0 for k in extract_cb_features(_EMPTY_DF)},
}


def extract_all_features(stock_code: str, data: dict) -> dict:
    """
    从预处理后的数据中提取全量特征
    data: {'stock_code', 'transaction_date', 'quotes', 'trades', 'orders'}
    返回特征字典
    """
    features: dict[str, Any] = {
        'stock_code': stock_code,
        'transaction_date': data.get('transaction_date', ''),
    }

    quotes = data.get('quotes')
    trades = data.get('trades')
    orders = data.get('orders')

    # 依次提取各类特征
    if quotes is not None and len(quotes) > 0:
        features.update(extract_oss_features(quotes))
        features.update(extract_trd_features(quotes))
        features.update(extract_obp_features(quotes))
        features.update(extract_pi_features(quotes))
        features.update(extract_pd_features(quotes))
    else:
        features.update(_QUOTES_FEATURE_DEFAULTS)

    if trades is not None and len(trades) > 0:
        features.update(extract_rs_features(trades))
        features.update(extract_ap_features(trades))
    else:
        features.update(_TRADES_FEATURE_DEFAULTS)

    if orders is not None and len(orders) > 0:
        features.update(extract_cb_features(orders))
    else:
        features.update(_ORDERS_FEATURE_DEFAULTS)

    return features


def save_features(stock_code: str, features: dict, output_dir: str) -> str:
    """保存单只股票的特征到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{stock_code}.csv')
    df = pd.DataFrame([features])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path


def load_all_features(features_dir: str) -> pd.DataFrame:
    """加载所有股票的特征文件，合并为DataFrame"""
    all_features = []
    if not os.path.exists(features_dir):
        return pd.DataFrame()
    for fname in os.listdir(features_dir):
        if fname.endswith('.csv'):
            fpath = os.path.join(features_dir, fname)
            df = pd.read_csv(fpath, encoding='utf-8-sig')
            all_features.append(df)
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
