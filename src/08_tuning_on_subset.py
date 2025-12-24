"""
Store Sales Forecasting - Hyperparameter Tuning on Subset
Version intelligente : tune sur le même subset que les baselines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
TUNED_MODELS_PATH = Path("models/tuned_subset")
REPORTS_PATH = Path("reports/figures")
TUNED_MODELS_PATH.mkdir(parents=True, exist_ok=True)

print("="*70)
print("🔬 HYPERPARAMETER TUNING - SMART VERSION")
print("Tuning sur subset puis application au full dataset")
print("="*70)
print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")

# ============================================
# 1. LOAD DATA
# ============================================
print("\n" + "="*70)
print("📦 CHARGEMENT DES DONNÉES")
print("="*70)

df = pd.read_csv(PROCESSED_PATH / "train_processed.csv", parse_dates=['date'])
test_full = pd.read_csv(PROCESSED_PATH / "test_processed.csv", parse_dates=['date'])

print(f"✅ Train: {df.shape}")
print(f"✅ Test: {test_full.shape}")

# ============================================
# 2. SELECT SAME SUBSET AS BASELINES
# ============================================
print("\n" + "="*70)
print("🎯 SÉLECTION DU SUBSET (GROCERY I + Store 44)")
print("="*70)

top_family = 'GROCERY I'
top_store = 44

subset = df[(df['family'] == top_family) & (df['store_nbr'] == top_store)].copy()
subset = subset.sort_values('date').reset_index(drop=True)

print(f"📊 Subset: {subset.shape}")

# Train/Test split
test_days = 30
train = subset[:-test_days].copy()
test = subset[-test_days:].copy()

print(f"✅ Train: {len(train)} jours")
print(f"✅ Test: {len(test)} jours")

# ============================================
# 3. PREPARE FEATURES
# ============================================
feature_cols = [
    'dayofweek', 'day', 'month', 'quarter', 'dayofyear', 'weekofyear',
    'is_weekend', 'is_month_start', 'is_month_end', 'is_payday',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    'sales_rolling_mean_7', 'sales_rolling_std_7',
    'sales_rolling_mean_14', 'sales_rolling_std_14',
    'sales_rolling_mean_30', 'sales_rolling_std_30',
    'is_holiday', 'has_promotion', 'onpromotion',
    'cluster', 'transactions', 'type_encoded'
]

feature_cols = [col for col in feature_cols if col in train.columns]

X_train = train[feature_cols].fillna(0)
y_train = train['sales']
X_test = test[feature_cols].fillna(0)
y_test = test['sales']

print(f"\n📊 {len(feature_cols)} features")

# ============================================
# 4. EVALUATION
# ============================================

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return rmse, mae, mape

# ============================================
# 5. XGBOOST TUNING
# ============================================
print("\n" + "="*70)
print("1️⃣ XGBOOST TUNING")
print("="*70)

try:
    import xgboost as xgb
    
    xgb_configs = [
        {'name': 'Baseline', 'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'name': 'Deep', 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'name': 'Regularized', 'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.08, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2},
        {'name': 'Fast', 'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.15, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'name': 'Aggressive', 'n_estimators': 400, 'max_depth': 9, 'learning_rate': 0.03, 'subsample': 0.75, 'colsample_bytree': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 1.5},
    ]
    
    xgb_results = []
    
    for config in xgb_configs:
        start = time.time()
        
        model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            reg_alpha=config.get('reg_alpha', 0),
            reg_lambda=config.get('reg_lambda', 1),
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)
        
        rmse, mae, mape = evaluate_model(y_test, pred)
        elapsed = time.time() - start
        
        print(f"\n{config['name']:15s}: RMSE={rmse:7.2f} | MAE={mae:7.2f} | MAPE={mape:6.2f}% | {elapsed:.1f}s")
        
        xgb_results.append({
            'name': config['name'],
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'time': elapsed,
            'config': config
        })
    
    best_xgb = min(xgb_results, key=lambda x: x['mape'])
    print(f"\n🏆 Meilleur XGBoost: {best_xgb['name']} - MAPE={best_xgb['mape']:.2f}%")
    
    xgb_success = True
    
except Exception as e:
    print(f"❌ Erreur XGBoost: {e}")
    xgb_success = False

# ============================================
# 6. LIGHTGBM TUNING
# ============================================
print("\n" + "="*70)
print("2️⃣ LIGHTGBM TUNING")
print("="*70)

try:
    import lightgbm as lgb
    
    lgb_configs = [
        {'name': 'Baseline', 'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'name': 'Deep', 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'name': 'Regularized', 'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.08, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2},
        {'name': 'Fast', 'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.15, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'name': 'Aggressive', 'n_estimators': 400, 'max_depth': 9, 'learning_rate': 0.03, 'subsample': 0.75, 'colsample_bytree': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 1.5},
    ]
    
    lgb_results = []
    
    for config in lgb_configs:
        start = time.time()
        
        model = lgb.LGBMRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            reg_alpha=config.get('reg_alpha', 0),
            reg_lambda=config.get('reg_lambda', 1),
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)
        
        rmse, mae, mape = evaluate_model(y_test, pred)
        elapsed = time.time() - start
        
        print(f"{config['name']:15s}: RMSE={rmse:7.2f} | MAE={mae:7.2f} | MAPE={mape:6.2f}% | {elapsed:.1f}s")
        
        lgb_results.append({
            'name': config['name'],
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'time': elapsed,
            'config': config
        })
    
    best_lgb = min(lgb_results, key=lambda x: x['mape'])
    print(f"\n🏆 Meilleur LightGBM: {best_lgb['name']} - MAPE={best_lgb['mape']:.2f}%")
    
    lgb_success = True
    
except Exception as e:
    print(f"❌ Erreur LightGBM: {e}")
    lgb_success = False

# ============================================
# 7. TRAIN ON FULL DATASET
# ============================================
print("\n" + "="*70)
print("🚀 APPLICATION AU FULL DATASET")
print("="*70)

X_full = df[feature_cols].fillna(0)
y_full = df['sales']
X_test_full = test_full[feature_cols].fillna(0)

if xgb_success:
    print("\n⏳ XGBoost sur full dataset...")
    start = time.time()
    
    best_xgb_model = xgb.XGBRegressor(
        n_estimators=best_xgb['config']['n_estimators'],
        max_depth=best_xgb['config']['max_depth'],
        learning_rate=best_xgb['config']['learning_rate'],
        subsample=best_xgb['config']['subsample'],
        colsample_bytree=best_xgb['config']['colsample_bytree'],
        reg_alpha=best_xgb['config'].get('reg_alpha', 0),
        reg_lambda=best_xgb['config'].get('reg_lambda', 1),
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    best_xgb_model.fit(X_full, y_full)
    pred_xgb = np.maximum(best_xgb_model.predict(X_test_full), 0)
    
    elapsed = time.time() - start
    print(f"   ⏱️ Training: {elapsed:.2f}s")
    
    joblib.dump(best_xgb_model, TUNED_MODELS_PATH / 'xgboost_best.pkl')
    print("   💾 Sauvegardé")

if lgb_success:
    print("\n⏳ LightGBM sur full dataset...")
    start = time.time()
    
    best_lgb_model = lgb.LGBMRegressor(
        n_estimators=best_lgb['config']['n_estimators'],
        max_depth=best_lgb['config']['max_depth'],
        learning_rate=best_lgb['config']['learning_rate'],
        subsample=best_lgb['config']['subsample'],
        colsample_bytree=best_lgb['config']['colsample_bytree'],
        reg_alpha=best_lgb['config'].get('reg_alpha', 0),
        reg_lambda=best_lgb['config'].get('reg_lambda', 1),
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    best_lgb_model.fit(X_full, y_full)
    pred_lgb = np.maximum(best_lgb_model.predict(X_test_full), 0)
    
    elapsed = time.time() - start
    print(f"   ⏱️ Training: {elapsed:.2f}s")
    
    joblib.dump(best_lgb_model, TUNED_MODELS_PATH / 'lightgbm_best.pkl')
    print("   💾 Sauvegardé")

# Ensemble
if xgb_success and lgb_success:
    pred_ensemble = (pred_xgb + pred_lgb) / 2
    
    submission = test_full[['id']].copy()
    submission['sales'] = pred_ensemble
    submission.to_csv(TUNED_MODELS_PATH / 'submission_ensemble_best.csv', index=False)
    
    submission['sales'] = pred_xgb
    submission.to_csv(TUNED_MODELS_PATH / 'submission_xgboost_best.csv', index=False)
    
    submission['sales'] = pred_lgb
    submission.to_csv(TUNED_MODELS_PATH / 'submission_lightgbm_best.csv', index=False)
    
    print("\n✅ 3 submissions sauvegardées")

# ============================================
# 8. COMPARISON
# ============================================
print("\n" + "="*70)
print("📊 RÉSUMÉ DES RÉSULTATS")
print("="*70)

if xgb_success and lgb_success:
    # Charger résultats originaux
    all_results = pd.read_csv(MODELS_PATH / 'all_models_results.csv')
    original_xgb = all_results[all_results['model'] == 'XGBoost'].iloc[0]
    original_ensemble = all_results[all_results['model'] == 'Ensemble'].iloc[0]
    
    print(f"\n📈 XGBOOST:")
    print(f"   Original:  {original_xgb['mape']:.2f}%")
    print(f"   Tuned:     {best_xgb['mape']:.2f}%")
    improvement_xgb = ((original_xgb['mape'] - best_xgb['mape']) / original_xgb['mape']) * 100
    print(f"   {'+' if improvement_xgb > 0 else ''}{improvement_xgb:.1f}% {'✅' if improvement_xgb > 0 else '❌'}")
    
    print(f"\n📈 LIGHTGBM:")
    print(f"   Tuned:     {best_lgb['mape']:.2f}%")
    
    ensemble_mape = (best_xgb['mape'] + best_lgb['mape']) / 2
    print(f"\n📈 ENSEMBLE (estimé):")
    print(f"   Original:  {original_ensemble['mape']:.2f}%")
    print(f"   Tuned:     {ensemble_mape:.2f}%")
    improvement_ensemble = ((original_ensemble['mape'] - ensemble_mape) / original_ensemble['mape']) * 100
    print(f"   {'+' if improvement_ensemble > 0 else ''}{improvement_ensemble:.1f}% {'✅' if improvement_ensemble > 0 else '❌'}")

# ============================================
# 9. VISUALIZATIONS
# ============================================
print("\n" + "="*70)
print("📈 VISUALISATIONS")
print("="*70)

if xgb_success:
    xgb_df = pd.DataFrame(xgb_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].barh(xgb_df['name'], xgb_df['rmse'], color='steelblue')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('XGBoost Configs - RMSE', fontweight='bold')
    axes[0].invert_yaxis()
    
    axes[1].barh(xgb_df['name'], xgb_df['mae'], color='coral')
    axes[1].set_xlabel('MAE')
    axes[1].set_title('XGBoost Configs - MAE', fontweight='bold')
    axes[1].invert_yaxis()
    
    axes[2].barh(xgb_df['name'], xgb_df['mape'], color='green')
    axes[2].set_xlabel('MAPE (%)')
    axes[2].set_title('XGBoost Configs - MAPE', fontweight='bold')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / '17_xgboost_tuning_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 17_xgboost_tuning_final.png")

if lgb_success:
    lgb_df = pd.DataFrame(lgb_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].barh(lgb_df['name'], lgb_df['rmse'], color='steelblue')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('LightGBM Configs - RMSE', fontweight='bold')
    axes[0].invert_yaxis()
    
    axes[1].barh(lgb_df['name'], lgb_df['mae'], color='coral')
    axes[1].set_xlabel('MAE')
    axes[1].set_title('LightGBM Configs - MAE', fontweight='bold')
    axes[1].invert_yaxis()
    
    axes[2].barh(lgb_df['name'], lgb_df['mape'], color='green')
    axes[2].set_xlabel('MAPE (%)')
    axes[2].set_title('LightGBM Configs - MAPE', fontweight='bold')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / '18_lightgbm_tuning_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 18_lightgbm_tuning_final.png")

print("\n" + "="*70)
print("✅ TUNING TERMINÉ !")
print(f"⏰ Fin: {datetime.now().strftime('%H:%M:%S')}")
print("="*70)
