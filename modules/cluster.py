"""
Task1：交易模式识别（无监督聚类）
使用KMeans聚类 + 多条件联合匹配为每个簇赋予交易模式语义
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

RANDOM_SEED = 42
N_CLUSTERS = 8


def prepare_feature_matrix(df_features: pd.DataFrame) -> tuple:
    """
    准备特征矩阵：选择数值特征、标准化
    返回 (X_scaled, feature_cols)
    """
    # 排除非特征列
    exclude_cols = ['stock_code', 'transaction_date', 'cluster_id', 'pattern_type', 'pattern_explanation']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    # 确保所有特征列都是数值
    X = df_features[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 填充缺失值和无穷值
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_cols, scaler


def run_clustering(df_features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list]:
    """
    对特征矩阵执行KMeans聚类
    返回 (df_with_cluster_id, X_scaled, feature_cols)
    """
    df = df_features.copy()
    X_scaled, feature_cols, scaler = prepare_feature_matrix(df)

    n_samples = len(df)
    n_clusters_actual = min(N_CLUSTERS, n_samples)

    if n_clusters_actual < 2:
        df['cluster_id'] = 0
        print(f"  [聚类] 样本数({n_samples})不足，仅生成1个聚类")
        return df, X_scaled, feature_cols

    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=RANDOM_SEED, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)

    # 评估指标
    sil = silhouette_score(X_scaled, df['cluster_id'])
    ch = calinski_harabasz_score(X_scaled, df['cluster_id'])
    db = davies_bouldin_score(X_scaled, df['cluster_id'])
    print(f"  [聚类评估] 轮廓系数: {sil:.4f} | CH指数: {ch:.2f} | DB指数: {db:.4f}")
    print(f"  [聚类] {n_clusters_actual} 个聚类 | 各簇样本数:")

    for cid in sorted(df['cluster_id'].unique()):
        cnt = (df['cluster_id'] == cid).sum()
        print(f"    簇{cid}: {cnt} 只股票")

    return df, X_scaled, feature_cols


def build_cluster_profiles(df_features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """构建各簇的关键特征画像（各特征均值）"""
    profiles = df_features.groupby('cluster_id')[feature_cols].mean()
    return profiles


def assign_patterns(df_features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    多条件联合匹配：为每个聚类赋予交易模式语义
    每个模式需 ≥3 个条件命中才生效，否则兜底为"机构长线配置"
    """
    df = df_features.copy()
    profiles = build_cluster_profiles(df, feature_cols)

    # 模式匹配规则（条件：特征 + 阈值）
    pattern_rules = {
        '游资强势连板拉升': [
            ('oss_mega_amount_pct', 'gt', 0.10),      # 超大单占比高
            ('book_imbalance', 'gt', 0.10),             # 买盘失衡（买>卖）
            ('ap_active_buy_pct', 'gt', 0.52),          # 主动买入占优
            ('pi_time_concentration', 'gt', 0.25),      # 开盘/尾盘集中
        ],
        '游资对倒出货': [
            ('oss_large_amount_pct', 'gt', 0.08),
            ('ap_active_sell_pct', 'gt', 0.52),
            ('cb_fast_cancel_ratio', 'gt', 0.10),
            ('rs_burst_ratio', 'gt', 0.05),
            ('book_imbalance', 'lt', -0.05),
        ],
        '游资吸筹建仓': [
            ('oss_medium_amount_pct', 'gt', 0.10),
            ('ap_active_buy_pct', 'gt', 0.53),
            ('book_imbalance', 'gt', 0.05),
            ('cb_buy_cancel_ratio', 'lt', 0.15),
            ('pi_open_30min_amount_pct', 'gt', 0.10),
        ],
        '量化高频T0交易': [
            ('rs_interval_cv', 'lt', 0.3),              # 成交间隔均匀
            ('rs_split_similarity', 'gt', 0.5),         # 拆单相似度高
            ('oss_small_amount_pct', 'gt', 0.40),       # 小单为主
            ('ap_unilateral_intensity', 'lt', 0.15),     # 单边强度低
        ],
        '量化套利交易': [
            ('obp_spread_pct', 'gt', 0.0002),           # 价差较大
            ('pi_price_std_pct', 'lt', 0.01),            # 波动低
            ('ap_active_net_pct', 'lt', 0.08),           # 净买卖接近0
            ('obp_imbalance_mean', 'lt', 0.05),
        ],
        '量化对冲配置': [
            ('trd_change_percent', 'lt', 0.02),         # 涨跌幅小
            ('obp_bid_ask_ratio', 'gt', 0.8),           # 买卖均衡
            ('pi_herfindahl_5min', 'lt', 0.1),          # 成交分散
            ('rs_burst_ratio', 'lt', 0.02),
        ],
        '机构减仓撤离': [
            ('ap_active_sell_pct', 'gt', 0.55),
            ('oss_large_amount_pct', 'gt', 0.06),
            ('trd_change_percent', 'lt', -0.01),
            ('book_imbalance', 'lt', -0.08),
        ],
        '机构长线配置': [
            ('trd_change_percent', 'lt', 0.03),
            ('pi_time_concentration', 'lt', 0.30),
            ('oss_medium_amount_pct', 'gt', 0.05),
            ('ap_unilateral_intensity', 'lt', 0.12),
        ],
    }

    pattern_names = list(pattern_rules.keys())

    for cluster_id in sorted(df['cluster_id'].unique()):
        profile = profiles.loc[cluster_id]
        scores = {name: 0 for name in pattern_names}

        for pattern_name, rules in pattern_rules.items():
            for feat, op, threshold in rules:
                if feat not in profile.index:
                    continue
                val = profile[feat]
                if pd.isna(val):
                    continue
                if op == 'gt' and val > threshold:
                    scores[pattern_name] += 1
                elif op == 'lt' and val < threshold:
                    scores[pattern_name] += 1

        # ≥3个条件命中才生效
        best_name = max(scores, key=lambda k: scores[k])
        best_score = scores[best_name]
        if best_score >= 3:
            pattern_name = best_name
        else:
            pattern_name = '机构长线配置'  # 兜底

        df.loc[df['cluster_id'] == cluster_id, 'pattern_type'] = pattern_name

    # 模式解释
    explanations = {
        '游资强势连板拉升': '超大单主导、主动买入积极、盘口买盘失衡明显、开盘或尾盘集中放量，呈现典型的游资连板拉升特征',
        '游资对倒出货': '大单成交活跃但主动卖出占优、撤单频繁、成交爆发、卖盘失衡，呈现游资对倒出货特征',
        '游资吸筹建仓': '中单稳步吸纳、主动买入略优、买盘微幅失衡、买入撤单率低、开盘时段活跃，呈现游资吸筹特征',
        '量化高频T0交易': '成交间隔均匀、拆单规整、小单为主、买卖均衡，呈现量化机构高频T0交易特征',
        '量化套利交易': '价差较宽、波动可控、净买卖接近零、盘口买卖均衡，呈现量化套利交易特征',
        '量化对冲配置': '涨跌幅小、买卖挂单均衡、成交时间分散、无爆发式交易，呈现量化对冲配置特征',
        '机构减仓撤离': '主动卖出明显占优、大单卖出、价格下行、卖盘失衡，呈现机构减仓撤离特征',
        '机构长线配置': '交易平稳、无极端集中、买卖相对均衡、波动可控，呈现机构长线配置特征',
    }

    df['pattern_explanation'] = df['pattern_type'].map(explanations).fillna('交易模式不明确，呈现均衡交易特征')

    return df


def save_pattern_result(df: pd.DataFrame, output_path: str):
    """保存 pattern_reco.csv"""
    result = df[['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']].copy()
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  [保存] pattern_reco.csv → {output_path}")
    return result
