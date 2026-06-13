"""
Task2：资金类型与交易意图识别（无真值规则判别任务）
使用11维多因子归一化打分判定游资/量化/散户，双源盘口联合规则识别交易意图
"""
import pandas as pd
import numpy as np
import os

# ============================================================
# 11维打分维度配置
# ============================================================
# 每个维度包含若干特征列，维度索引说明见下方 yz_like_dims
DIMENSION_CONFIG = [
    # 维度索引 0: OSS大额成交 — 值越大越像游资
    ['oss_mega_amount_pct', 'oss_large_amount_pct'],
    # 维度索引 1: RS拆单时序 — 值越大越像量化（拆单均匀）
    ['rs_split_similarity', 'rs_burst_ratio'],
    # 维度索引 2: CB撤单分化 — 值越大越像量化（撤单多→量化）
    ['cb_fast_cancel_ratio', 'cb_buy_cancel_ratio'],
    # 维度索引 3: AP主动单边 — 值越大越像游资（单边买入）
    ['ap_active_buy_pct', 'ap_active_net_pct'],
    # 维度索引 4: OBP盘口 — 值越大越像量化（价差窄、均衡）
    ['obp_spread_pct', 'book_imbalance'],
    # 维度索引 5: PD价格冲击 — 值越大越像游资（冲击大）
    ['pd_impact', 'pd_Q1_ratio'],
    # 维度索引 6: PI时段波动 — 值越大越像游资（集中、波动大）
    ['pi_time_concentration', 'pi_price_std_pct'],
    # 维度索引 7: 连续买入笔数 — 值越大越像游资
    ['ap_active_buy_run_max'],
    # 维度索引 8: 盘口大单挂单 — 值越大越像游资（大单集中在前档）
    ['obp_big_bid_ratio', 'obp_big_ask_ratio'],
    # 维度索引 9: 卖出撤单率 — 值越大越像量化（卖撤多→量化）
    ['cb_sell_cancel_ratio'],
    # 维度索引 10: 单边强度 — 值越大越像游资
    ['ap_unilateral_intensity'],
]

# 游资倾向维度索引：这些维度的值越大 → 游资分越高
YZ_LIKE_DIMS = {0, 3, 5, 6, 7, 8, 10}

# 各维度权重（游资视角 / 量化视角，一一对应）
# 游资侧：大额、单边、冲击、集中权重更高
WEIGHT_YZ = [0.15, 0.06, 0.06, 0.14, 0.06, 0.10, 0.10, 0.08, 0.08, 0.06, 0.11]
# 量化侧：拆单、撤单、盘口均衡、波动低权重更高
WEIGHT_QT = [0.06, 0.14, 0.12, 0.06, 0.14, 0.06, 0.06, 0.06, 0.06, 0.12, 0.12]

# ============================================================
# 散户打分特征配置
# ============================================================
# 每个元素为 (特征列名或特征列名列表, 是否正向, 权重)
# 正向 = 值越大越像散户，反向 = 值越小越像散户
RETAIL_FEATURES = [
    # 1. 小单成交占比 — 正向：小单多→散户
    (['oss_small_amount_pct', 'oss_small_count_pct'], True, 0.25),
    # 2. 每笔交易规模 — 反向：交易额小→散户
    (['trd_avg_trade_size', 'trd_avg_trade_amount'], False, 0.18),
    # 3. 拆单相似度 — 反向：不拆单→散户
    (['rs_split_similarity'], False, 0.15),
    # 4. 成交间隔变异 — 正向：间隔不均匀→散户
    (['rs_interval_cv', 'rs_burst_ratio'], True, 0.12),
    # 5. 撤单活跃度 — 反向：撤单少→散户
    (['cb_fast_cancel_ratio', 'cb_cancel_order_ratio'], False, 0.12),
    # 6. 价格冲击 — 反向：冲击小→散户
    (['pd_impact', 'pd_Q1_ratio'], False, 0.10),
    # 7. 单边强度 — 正向：追涨杀跌→散户
    (['ap_unilateral_intensity'], True, 0.08),
]

# ============================================================
# 交易意图识别 — 双源盘口联合规则阈值配置
# ============================================================
# 综合失衡度：首条快照 vs 全天均值的融合权重
IMBALANCE_SNAP_WEIGHT = 0.4
IMBALANCE_MEAN_WEIGHT = 0.6

# 主动成交占比阈值（买入/卖出判定）
ACTIVE_BUY_PCT_THRESHOLD = 0.6
ACTIVE_SELL_PCT_THRESHOLD = 0.6

# 盘口失衡度阈值（正=买盘失衡，负=卖盘失衡）
IMBALANCE_BUY_THRESHOLD = 0.08
IMBALANCE_SELL_THRESHOLD = -0.08


def _compute_retail_score(norm_df: pd.DataFrame, df: pd.DataFrame,
                          n_samples: int) -> np.ndarray:
    """
    计算散户得分（基于 RETAIL_FEATURES 配置）
    优先使用 norm_df 中已归一化的值；对 norm_df 中不存在的列从 df 中实时 MinMax 归一化。
    返回长度为 n_samples 的 numpy 数组
    """
    score_sh = np.zeros(n_samples)
    total_weight = 0.0

    for feature_cols, is_direct, weight in RETAIL_FEATURES:
        # 1) 取已在 norm_df 中的列
        in_norm = [c for c in feature_cols if c in norm_df.columns]
        # 2) 取只在 df 中、需要临时归一化的列
        in_df_only = [c for c in feature_cols if c not in norm_df.columns and c in df.columns]

        if not in_norm and not in_df_only:
            continue

        # 收集所有列归一化后的值（扁平列表，确保每列等权）
        all_col_vals = []
        if in_norm:
            all_col_vals.extend([norm_df[c].values for c in in_norm])
        # 从 df 中临时 MinMax 归一化
        if in_df_only:
            for col in in_df_only:
                raw = df[col].fillna(0).replace([np.inf, -np.inf], 0).to_numpy(dtype=float)
                vmin, vmax = raw.min(), raw.max()
                if vmax - vmin > 1e-8:
                    all_col_vals.append((raw - vmin) / (vmax - vmin))
                else:
                    all_col_vals.append(np.full(n_samples, 0.5))

        raw_vals = np.mean(all_col_vals, axis=0)

        if is_direct:
            score_sh += raw_vals * weight
        else:
            score_sh += (1.0 - raw_vals) * weight
        total_weight += weight

    if total_weight > 0:
        score_sh /= total_weight
    return score_sh


def identify_capital_type(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    11维多因子归一化打分，三分类判定：游资 / 量化 / 散户
    返回带有 capital_type 列的 DataFrame
    """
    df = df_features.copy()
    n_samples = len(df)

    if n_samples == 0:
        df['capital_type'] = '量化'
        return df

    # --- 1. 过滤可用维度（特征列必须存在） ---
    valid_dims = []
    valid_dim_orig_indices = []
    for orig_idx, dim_cols in enumerate(DIMENSION_CONFIG):
        available_cols = [c for c in dim_cols if c in df.columns]
        if available_cols:
            valid_dims.append(available_cols)
            valid_dim_orig_indices.append(orig_idx)

    if not valid_dims:
        # 维度全缺时兜底：走散户打分
        df['capital_type'] = '散户'
        return df

    # --- 2. 跨样本全局 MinMax 归一化每个特征列 ---
    norm_df = pd.DataFrame(index=df.index)
    for dim_cols in valid_dims:
        for col in dim_cols:
            raw = df[col].fillna(0).replace([np.inf, -np.inf], 0).to_numpy(dtype=float)
            vmin, vmax = raw.min(), raw.max()
            if vmax - vmin > 1e-8:
                norm_df[col] = (raw - vmin) / (vmax - vmin)
            else:
                norm_df[col] = 0.5  # 常数列置中

    # --- 3. 游资/量化 加权打分 ---
    score_yz = np.zeros(n_samples)  # 游资得分
    score_qt = np.zeros(n_samples)  # 量化得分

    for dim_idx_local, dim_cols in enumerate(valid_dims):
        orig_dim_idx = valid_dim_orig_indices[dim_idx_local]

        # 计算该维度的平均归一化值
        dim_score = np.mean([norm_df[c].values for c in dim_cols], axis=0)

        wyz = WEIGHT_YZ[orig_dim_idx]
        wqt = WEIGHT_QT[orig_dim_idx]

        if orig_dim_idx in YZ_LIKE_DIMS:
            # 游资倾向维度：值越大 → 游资分越高
            score_yz += dim_score * wyz
            score_qt += (1 - dim_score) * wqt
        else:
            # 量化倾向维度：值越大 → 游资分越低（量化分越高）
            score_yz += (1 - dim_score) * wyz
            score_qt += dim_score * wqt

    # 归一化权重使总分可比
    total_w_yz = sum(WEIGHT_YZ[i] for i in valid_dim_orig_indices)
    total_w_qt = sum(WEIGHT_QT[i] for i in valid_dim_orig_indices)
    if total_w_yz > 0:
        score_yz /= total_w_yz
    if total_w_qt > 0:
        score_qt /= total_w_qt

    # --- 4. 散户打分 ---
    score_sh = _compute_retail_score(norm_df, df, n_samples)

    # --- 5. 三分类判定：得分最高者胜出 ---
    df['capital_type'] = '量化'  # 默认
    for i in range(n_samples):
        max_score = max(score_yz[i], score_qt[i], score_sh[i])
        if max_score == score_yz[i]:
            df.loc[df.index[i], 'capital_type'] = '游资'
        elif max_score == score_sh[i]:
            df.loc[df.index[i], 'capital_type'] = '散户'
        # else: 保持默认 '量化'

    # 输出分布
    yz_cnt = (df['capital_type'] == '游资').sum()
    qt_cnt = (df['capital_type'] == '量化').sum()
    sh_cnt = (df['capital_type'] == '散户').sum()
    print(f"  [Task2资金类型] 游资: {yz_cnt} | 量化: {qt_cnt} | 散户: {sh_cnt}")
    print(f"    游资占比: {yz_cnt / n_samples:.1%} | 量化占比: {qt_cnt / n_samples:.1%} | 散户占比: {sh_cnt / n_samples:.1%}")

    return df


def identify_intention(df: pd.DataFrame) -> pd.DataFrame:
    """
    双源盘口联合规则 + 主动成交占比，识别交易意图（向量化实现）
    返回带有 capital_intention 列的 DataFrame
    """
    df = df.copy()

    # --- 提取特征列，填充缺失值 ---
    buy_pct = df.get('ap_active_buy_pct', pd.Series(0.5, index=df.index)).fillna(0.5).astype(float)
    sell_pct = df.get('ap_active_sell_pct', pd.Series(0.5, index=df.index)).fillna(0.5).astype(float)
    imbalance_snap = df.get('book_imbalance', pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)
    imbalance_mean = df.get('obp_imbalance_mean', pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)

    # --- 综合失衡度：首条快照 + 全天均值加权融合 ---
    imbalance = IMBALANCE_SNAP_WEIGHT * imbalance_snap + IMBALANCE_MEAN_WEIGHT * imbalance_mean

    # --- 向量化三分类判定 ---
    is_buy = (buy_pct > ACTIVE_BUY_PCT_THRESHOLD) & (imbalance > IMBALANCE_BUY_THRESHOLD)
    is_sell = (sell_pct > ACTIVE_SELL_PCT_THRESHOLD) & (imbalance < IMBALANCE_SELL_THRESHOLD)

    df['capital_intention'] = np.select(
        [is_buy, is_sell],
        ['买入', '卖出'],
        default='T0交易',
    )

    # --- 统计输出 ---
    n = len(df)
    buy_count = is_buy.sum()
    sell_count = is_sell.sum()
    t0_count = n - buy_count - sell_count

    print(f"  [Task2交易意图] 买入: {buy_count} | 卖出: {sell_count} | T0交易: {t0_count}")
    if n > 0:
        print(f"    买入占比: {buy_count / n:.1%} | 卖出占比: {sell_count / n:.1%} | T0占比: {t0_count / n:.1%}")

    return df


def identify_all(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Task2 完整流程：资金类型判定 → 交易意图识别
    """
    df = identify_capital_type(df_features)
    df = identify_intention(df)
    return df


def save_predict_result(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """保存 predict_result.csv"""
    result = df[['stock_code', 'transaction_date', 'capital_type', 'capital_intention']].copy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  [保存] predict_result.csv → {output_path}")

    # 格式校验
    assert result['capital_type'].isin(['游资', '量化', '散户']).all(), \
        f"capital_type 包含非法值: {result['capital_type'].unique()}"
    assert result['capital_intention'].isin(['买入', '卖出', 'T0交易']).all(), \
        f"capital_intention 包含非法值: {result['capital_intention'].unique()}"
    print(f"  [校验] 格式合法 ✓")

    return result
