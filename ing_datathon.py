import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import IsotonicRegression
import optuna
import warnings
import gc
import matplotlib.pyplot as plt

# Uyarıları bastır
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Optuna loglamasını açarak her adımı izleyelim
optuna.logging.set_verbosity(optuna.logging.INFO)

# =============================================================================
# YARIŞMA METRİĞİ AYARLARI
# =============================================================================

def recall_at_k(y_true, y_prob, k=0.1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    P = y_true.sum()
    return float(tp_at_k / P) if P > 0 else 0.0

def lift_at_k(y_true, y_prob, k=0.1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    precision_at_k = tp_at_k / m
    prevalence = y_true.mean()
    return float(precision_at_k / prevalence) if prevalence > 0 else 0.0

def convert_auc_to_gini(auc):
    return 2 * auc - 1

def ing_hubs_datathon_metric(y_true, y_prob):
    score_weights = {"gini": 0.4, "recall_at_10perc": 0.3, "lift_at_10perc": 0.3}
    baseline_scores = {"roc_auc": 0.6925726757936908, "recall_at_10perc": 0.18469015795868773, "lift_at_10perc": 1.847159286784029}
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        return 0.0
        
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)

    new_scores = {"roc_auc": roc_auc, "recall_at_10perc": recall_at_10perc, "lift_at_10perc": lift_at_10perc}
    
    baseline_scores["gini"] = convert_auc_to_gini(baseline_scores["roc_auc"])
    new_scores["gini"] = convert_auc_to_gini(new_scores["roc_auc"])
    
    final_gini_score = new_scores["gini"] / baseline_scores["gini"] if baseline_scores["gini"] != 0 else 0
    final_recall_score = new_scores["recall_at_10perc"] / baseline_scores["recall_at_10perc"] if baseline_scores["recall_at_10perc"] != 0 else 0
    final_lift_score = new_scores["lift_at_10perc"] / baseline_scores["lift_at_10perc"] if baseline_scores["lift_at_10perc"] != 0 else 0
    
    final_score = (final_gini_score * score_weights["gini"] + 
                   final_recall_score * score_weights["recall_at_10perc"] + 
                   final_lift_score * score_weights["lift_at_10perc"])
    return final_score

# =============================================================================
# VERİ YÜKLEME VE ÖZELLİK MÜHENDİSLİĞİ
# =============================================================================
def load_data():
    """Tüm CSV dosyalarını yükler."""
    print("Veri setleri yükleniyor...")
    customer_history = pd.read_csv('/kaggle/input/ing-hackathon-data/customer_history.csv', parse_dates=['date'])
    customers = pd.read_csv('/kaggle/input/ing-hackathon-data/customers.csv')
    train_ref = pd.read_csv('/kaggle/input/ing-hackathon-data/referance_data.csv', parse_dates=['ref_date'])
    test_ref = pd.read_csv('/kaggle/input/ing-hackathon-data/referance_data_test.csv', parse_dates=['ref_date'])
    sample_submission = pd.read_csv('/kaggle/input/ing-hackathon-data/sample_submission.csv')
    print("Veri setleri başarıyla yüklendi.")
    return customer_history, customers, train_ref, test_ref, sample_submission

def feature_engineering(ref_df, customers_df, history_df, kmeans_model=None, scaler_health=None):
    """Kapsamlı ve birleştirilmiş "ultimate" özellik mühendisliği uygular."""
    print(f"Gelişmiş özellik mühendisliği başlıyor. İşlenecek {len(ref_df)} müşteri var.")
    
    df = pd.merge(ref_df, customers_df, on='cust_id', how='left')
    df.loc[(df['work_sector'].isnull()) & (df['work_type'] == 'Student'), 'work_sector'] = 'Student'
    df.loc[(df['work_sector'].isnull()) & (df['work_type'] == 'Unemployed'), 'work_sector'] = 'Unemployed'
    df.loc[(df['work_sector'].isnull()) & (df['work_type'] == 'Retired'), 'work_sector'] = 'Pension'

    history = pd.merge(df[['cust_id', 'ref_date']], history_df, on='cust_id', how='left')
    history = history[history['date'] <= history['ref_date']].copy()
    
    history['total_transaction_amt'] = history['mobile_eft_all_amt'] + history['cc_transaction_all_amt']
    history['total_transaction_cnt'] = history['mobile_eft_all_cnt'] + history['cc_transaction_all_cnt']
    history['months_before_ref'] = (history['ref_date'].dt.year - history['date'].dt.year) * 12 + (history['ref_date'].dt.month - history['date'].dt.month)

    print("  -> Zaman bazlı agregasyon özellikleri oluşturuluyor...")
    aggs = {}
    cols_to_agg = [
        'mobile_eft_all_cnt', 'active_product_category_nbr', 'mobile_eft_all_amt', 
        'cc_transaction_all_amt', 'cc_transaction_all_cnt', 'total_transaction_amt', 
        'total_transaction_cnt'
    ]
    stats_to_calc = ['mean', 'sum', 'std', 'min', 'max','median']
    
    for col in cols_to_agg:
        aggs[col] = stats_to_calc
    
    grouped_all = history.groupby('cust_id').agg(aggs)
    grouped_all.columns = ['_'.join(col).strip() + '_all' for col in grouped_all.columns.values]
    df = pd.merge(df, grouped_all, on='cust_id', how='left')
    
    time_windows = [1, 3, 6, 9, 12, 15, 18, 24]
    for window in time_windows:
        period_history = history[history['months_before_ref'] < window]
        grouped_period = period_history.groupby('cust_id').agg(aggs)
        grouped_period.columns = ['_'.join(col).strip() + f'_last_{window}m' for col in grouped_period.columns.values]
        df = pd.merge(df, grouped_period, on='cust_id', how='left')

    print("  -> Büyüme, mevsimsellik ve anomali özellikleri oluşturuluyor...")
    for window in [3, 6, 12]:
        if f'total_transaction_amt_sum_last_{window*2}m' in df.columns:
            df[f'transaction_growth_{window}m'] = (df[f'total_transaction_amt_sum_last_{window}m'] - df[f'total_transaction_amt_sum_last_{window*2}m']) / (df[f'total_transaction_amt_sum_last_{window*2}m'] + 1e-6)

    df['ref_month'] = df['ref_date'].dt.month
    df['ref_quarter'] = df['ref_date'].dt.quarter
    df['is_year_end'] = df['ref_month'].isin([12, 1]).astype(int)

    for col in ['total_transaction_amt', 'total_transaction_cnt']:
        for window in [3, 6]:
            mean_col = f'{col}_mean_last_{window}m'
            std_col = f'{col}_std_last_{window}m'
            current_col = f'{col}_sum_last_1m'
            if mean_col in df.columns and std_col in df.columns and current_col in df.columns:
                df[f'{col}_zscore_1m_vs_{window}m'] = (df[current_col] - df[mean_col]) / (df[std_col] + 1e-6)
    
    print("  -> RFM ve diğer davranışsal özellikler ekleniyor...")
    last_transaction = history.groupby('cust_id')['date'].max().reset_index(name='last_transaction_date')
    df = pd.merge(df, last_transaction, on='cust_id', how='left')
    df['days_since_last_transaction'] = (df['ref_date'] - df['last_transaction_date']).dt.days
    
    first_transaction = history.groupby('cust_id')['date'].min().reset_index(name='first_transaction_date')
    df = pd.merge(df, first_transaction, on='cust_id', how='left')

    df['total_months_observed'] = ((df['ref_date'] - df['first_transaction_date']).dt.days / 30.44).round()
    date_nunique = history.groupby('cust_id')['date'].nunique()
    df['date_nunique_all'] = df['cust_id'].map(date_nunique)
    
    df['transaction_frequency_ratio'] = df['date_nunique_all'] / (df['total_months_observed'] + 1e-6)
    df['inactive_months_count'] = df['total_months_observed'] - df['date_nunique_all']
    df['banking_age'] = df['tenure'] - 17
    df['avg_monetary_value_all'] = df['total_transaction_amt_sum_all'] / (df['total_transaction_cnt_sum_all'] + 1e-6)

    rfm_cols = ['days_since_last_transaction', 'transaction_frequency_ratio', 'avg_monetary_value_all']
    rfm_data = df[rfm_cols].fillna(0) 

    if kmeans_model is None:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['loyalty_tier'] = kmeans.fit_predict(rfm_data)
        kmeans_model_to_return = kmeans
    else: 
        df['loyalty_tier'] = kmeans_model.predict(rfm_data)
        kmeans_model_to_return = None

    print("  -> Müşteri Sağlık Skoru oluşturuluyor...")
    health_cols = {'pos': ['transaction_frequency_ratio'], 'neg': ['days_since_last_transaction', 'inactive_months_count']}
    health_data = df[health_cols['pos'] + health_cols['neg']].fillna(0)

    if scaler_health is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        health_scaled = scaler.fit_transform(health_data)
        scaler_health_to_return = scaler
    else:
        health_scaled = scaler_health.transform(health_data)
        scaler_health_to_return = None
    
    health_scaled_df = pd.DataFrame(health_scaled, index=df.index, columns=health_cols['pos'] + health_cols['neg'])
    df['customer_health_score'] = health_scaled_df[health_cols['pos']].sum(axis=1) - health_scaled_df[health_cols['neg']].sum(axis=1)

    bins_age = [17, 30, 45, 65, 120]; labels_age = ['young_adults', 'established_adults', 'prime_age_high_risk', 'senior']
    df['age_group'] = pd.cut(df['age'], bins=bins_age, labels=labels_age, right=True)
    bins_tenure = [-1, 12, 36, 1200]; labels_tenure = ['new_customer', 'loyal_customer', 'veteran_customer']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins_tenure, labels=labels_tenure, right=True)
    
    df['gender_x_age_group'] = df['gender'].astype(str) + '_' + df['age_group'].astype(str)
    df['tenure_group_x_work_sector'] = df['tenure_group'].astype(str) + '_' + df['work_sector'].astype(str)
    df['province_x_work_sector'] = df['province'].astype(str) + '_' + df['work_sector'].astype(str)
    df['religion_x_age_group'] = df['religion'].astype(str) + '_' + df['age_group'].astype(str)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    cat_cols_final = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    for col in cat_cols_final:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if '-' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('-')
            df[col] = df[col].fillna("-")
        else:
            df[col] = pd.Categorical(df[col].fillna("-"))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"Özellik mühendisliği tamamlandı. Toplam {df.shape[1]} özellik oluşturuldu.")
    return df, kmeans_model_to_return, scaler_health_to_return

# =============================================================================
# Model Eğitimi ve Tahmin
# =============================================================================
def optimize_and_train_catboost(train_df, test_df, sample_submission_df, n_trials_optuna=50):
    """Optuna ile optimize edilmiş CatBoost modeli eğitir."""
    print("Nihai CatBoost modeli eğitim süreci başlıyor...")
    
    target = 'churn'
    features_to_drop = ['cust_id', 'ref_date', target]
    original_features = [col for col in train_df.columns if col not in features_to_drop]
    
    cat_cols_initial = [col for col in original_features if train_df[col].dtype == 'object' or isinstance(train_df[col].dtype, pd.CategoricalDtype)]
    num_cols_initial = [col for col in original_features if col not in cat_cols_initial]

    print("  -> Nadir kategoriler 'rare' olarak birleştiriliyor...")
    for col in cat_cols_initial:
        value_counts = train_df[col].value_counts()
        rare_values = value_counts[value_counts < 25].index
        
        if 'rare' not in train_df[col].cat.categories:
            train_df[col] = train_df[col].cat.add_categories('rare')
        if 'rare' not in test_df[col].cat.categories:
            test_df[col] = test_df[col].cat.add_categories('rare')

        train_df[col] = train_df[col].replace(rare_values, 'rare')
        test_df[col] = test_df[col].replace(rare_values, 'rare')

    print("  -> Yüksek korelasyonlu özellikler kaldırılıyor...")
    corr_matrix = train_df[num_cols_initial].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= 0.96)]
    
    train_df = train_df.drop(columns=to_drop)
    test_df = test_df.drop(columns=to_drop)
    print(f"    -> {len(to_drop)} özellik kaldırıldı.")
    
    features = [col for col in train_df.columns if col not in features_to_drop]
    cat_cols = [col for col in features if isinstance(train_df[col].dtype, pd.CategoricalDtype)]
    
    # Optuna Objective Function
    def objective(trial):
        params = {
            'iterations': 6000,
            'depth': trial.suggest_int('depth', 5, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'max_bin': trial.suggest_int('max_bin', 32, 128),
            'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
            'allow_writing_files': False,
            'random_state': 0,
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 1.0)
        }
        
        cv_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv_opt.split(train_df, train_df[target]):
            X_train_fold, y_train_fold = train_df.iloc[train_idx].copy(), train_df[target].iloc[train_idx]
            X_val_fold = train_df.iloc[val_idx].copy()
            
            global_mean_fold = y_train_fold.mean()
            alpha = 5
            for col in cat_cols:
                counts = X_train_fold[col].value_counts()
                means = y_train_fold.groupby(X_train_fold[col]).mean()
                smooth_means = (means * counts + global_mean_fold * alpha) / (counts + alpha)
                
                X_train_fold[col] = X_train_fold[col].map(smooth_means).astype(float)
                X_val_fold[col] = X_val_fold[col].map(smooth_means).astype(float).fillna(global_mean_fold)
            
            model = CatBoostClassifier(**params)
            model.fit(X_train_fold[features], y_train_fold,
                      eval_set=(X_val_fold[features], train_df[target].iloc[val_idx]),
                      early_stopping_rounds=100, verbose=0)
            
            preds = model.predict_proba(X_val_fold[features])[:, 1]
            scores.append(ing_hubs_datathon_metric(train_df[target].iloc[val_idx], preds))
        return np.mean(scores)

    print(f"\nOptuna ile optimizasyon başlıyor ({n_trials_optuna} deneme)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials_optuna)
    best_params = study.best_params
    print("\nOptimizasyon tamamlandı. En iyi parametreler bulundu:", best_params)

    # Final Model Eğitimi
    final_model_params = {
        'iterations': 6000,
        **best_params 
    }

    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    test_df_encoded = test_df.copy()
    global_mean_full = train_df[target].mean()
    alpha = 5
    for col in cat_cols:
        counts = train_df[col].value_counts()
        means = train_df[target].groupby(train_df[col]).mean()
        smooth_means = (means * counts + global_mean_full * alpha) / (counts + alpha)
        test_df_encoded[col] = test_df_encoded[col].map(smooth_means).astype(float).fillna(global_mean_full)

    print(f"\n7-Fold StratifiedKFold ile eğitim başlıyor...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, train_df[target])):
        print(f"===== Fold {fold+1} =====")
        
        X_train_fold, y_train_fold = train_df.iloc[train_idx].copy(), train_df[target].iloc[train_idx]
        X_val_fold = train_df.iloc[val_idx].copy()
        
        global_mean_fold = y_train_fold.mean()
        for col in cat_cols:
            counts = X_train_fold[col].value_counts()
            means = y_train_fold.groupby(X_train_fold[col]).mean()
            smooth_means = (means * counts + global_mean_fold * alpha) / (counts + alpha)
            
            X_train_fold[col] = X_train_fold[col].map(smooth_means).astype(float)
            X_val_fold[col] = X_val_fold[col].map(smooth_means).astype(float).fillna(global_mean_fold)
        
        model = CatBoostClassifier(**final_model_params)
        model.fit(X_train_fold[features], y_train_fold,
                  eval_set=(X_val_fold[features], train_df[target].iloc[val_idx]),
                  early_stopping_rounds=40,
                  verbose=500)
        
        oof_preds[val_idx] = model.predict_proba(X_val_fold[features])[:, 1]
        test_preds += model.predict_proba(test_df_encoded[features])[:, 1] / cv.n_splits
        
        del X_train_fold, y_train_fold, X_val_fold
        gc.collect()

    final_oof_score = ing_hubs_datathon_metric(train_df[target], oof_preds)
    print("\n==============================================")
    print(f"Final OOF Skoru (Kalibrasyon Öncesi): {final_oof_score:.5f}")
    
    print("  -> Olasılık kalibrasyonu uygulanıyor...")
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(oof_preds, train_df[target])
    calibrated_test_preds = iso_reg.transform(test_preds)

    calibrated_oof_score = ing_hubs_datathon_metric(train_df[target], iso_reg.transform(oof_preds))
    print(f"Final OOF Skoru (Kalibrasyon Sonrası): {calibrated_oof_score:.5f}")
    print("==============================================")
    
    submission_df = sample_submission_df.copy()
    submission_df['churn'] = calibrated_test_preds
    submission_df.to_csv('submission_catboost_calibrated.csv', index=False)
    
    print("\nİşlem tamamlandı ve 'submission_catboost_calibrated.csv' dosyası oluşturuldu.")
    return submission_df

# =============================================================================
# Ana Çalıştırma Bloğu
# =============================================================================
if __name__ == "__main__":
    customer_history, customers, train_ref, test_ref, sample_submission = load_data()
    
    train_featured, kmeans_model, scaler_health = feature_engineering(train_ref, customers, customer_history)
    test_featured, _, _ = feature_engineering(test_ref, customers, customer_history, kmeans_model=kmeans_model, scaler_health=scaler_health)
    
    del customer_history, customers, train_ref, test_ref
    gc.collect()

    submission = optimize_and_train_catboost(train_featured, test_featured, sample_submission, n_trials_optuna=50)
    
    print("\nNihai işlem tamamlandı. Gönderim dosyasının ilk 5 satırı:")
    print(submission.head())