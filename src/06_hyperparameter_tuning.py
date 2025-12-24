"""
Store Sales Forecasting - Hyperparameter Tuning
Optimisation des modèles XGBoost et LightGBM
Sans toucher aux modèles existants
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
TUNED_MODELS_PATH = Path("models/tuned")
REPORTS_PATH = Path("reports/figures")
TUNED_MODELS_PATH.mkdir(parents=True, exist_ok=True)

print("="*70)
print("🔬 HYPERPARAMETER TUNING - ADVANCED OPTIMIZATION")
print("="*70)
print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")

# ============================================
# 1. LOAD DATA
# ============================================
print("\n" + "="*70)
print("📦 CHARGEMENT DES DONNÉES")
print("="*70)

start_load = time.time()
train = pd.read_csv(PROCESSED_PATH / "train_processed.csv", parse_dates=['date'])
test_full = pd.read_csv(PROCESSED_PATH / "test_processed.csv", parse_dates=['date'])

elapsed_load = time.time() - start_load
print(f"✅ Données chargées en {elapsed_load:.2f}s")
print(f"   Train: {train.shape}")
print(f"   Test: {test_full.shape}")

# ============================================
# 2. PREPARE FEATURES
# ============================================
print("\n" + "="*70)
print("🔧 PRÉPARATION DES FEATURES")
print("="*70)

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
print(f"📊 {len(feature_cols)} features sélectionnées")

X_full = train[feature_cols].fillna(0)
y_full = train['sales']
X_test = test_full[feature_cols].fillna(0)

print(f"✅ X_full: {X_full.shape}")
print(f"✅ X_test: {X_test.shape}")

# ============================================
# 3. TIME SERIES CROSS-VALIDATION SPLIT
# ============================================
print("\n" + "="*70)
print("✂️ TIME SERIES CROSS-VALIDATION SETUP")
print("="*70)

# Utiliser 80% pour train, 20% pour validation
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"📊 {n_splits} splits pour cross-validation")

# ============================================
# 4. EVALUATION FUNCTION
# ============================================

def evaluate_predictions(y_true, y_pred):
    """Calcule les métriques"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return rmse, mae, mape

def cross_validate_model(model, X, y, cv, model_name):
    """Cross-validation avec Time Series Split"""
    print(f"\n⏳ Cross-validation {model_name}...")
    
    scores = {'rmse': [], 'mae': [], 'mape': []}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        start_fold = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred = np.maximum(y_pred, 0)
        elapsed_fold = time.time() - start_fold
        
        rmse, mae, mape = evaluate_predictions(y_val, y_pred)
        scores['rmse'].append(rmse)
        scores['mae'].append(mae)
        scores['mape'].append(mape)
        
        print(f"   Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}% ({elapsed_fold:.1f}s)")
    
    avg_rmse = np.mean(scores['rmse'])
    avg_mae = np.mean(scores['mae'])
    avg_mape = np.mean(scores['mape'])
    
    print(f"   📊 Moyenne: RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}, MAPE={avg_mape:.2f}%")
    
    return avg_rmse, avg_mae, avg_mape

# ============================================
# 5. XGBOOST HYPERPARAMETER TUNING
# ============================================
print("\n" + "="*70)
print("1️⃣ XGBOOST - HYPERPARAMETER TUNING")
print("="*70)

try:
    import xgboost as xgb
    
    print("\n🔍 Test de plusieurs configurations XGBoost...")
    print("   (Ton GPU va chauffer ! 🔥)")
    
    # Configurations à tester
    xgb_configs = [
        {
            'name': 'Config 1 (Baseline)',
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
            }
        },
        {
            'name': 'Config 2 (Deep Trees)',
            'params': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
            }
        },
        {
            'name': 'Config 3 (Regularized)',
            'params': {
                'n_estimators': 250,
                'max_depth': 7,
                'learning_rate': 0.08,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.5,
                'reg_lambda': 2,
            }
        },
        {
            'name': 'Config 4 (Fast Learning)',
            'params': {
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.15,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
            }
        },
        {
            'name': 'Config 5 (Aggressive)',
            'params': {
                'n_estimators': 400,
                'max_depth': 9,
                'learning_rate': 0.03,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'gamma': 0.05,
                'reg_alpha': 0.3,
                'reg_lambda': 1.5,
            }
        }
    ]
    
    xgb_results = []
    
    for config in xgb_configs:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {config['name']}")
        print(f"{'='*70}")
        
        start_config = time.time()
        
        model = xgb.XGBRegressor(
            **config['params'],
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        avg_rmse, avg_mae, avg_mape = cross_validate_model(model, X_full, y_full, tscv, config['name'])
        
        elapsed_config = time.time() - start_config
        
        xgb_results.append({
            'name': config['name'],
            'rmse': avg_rmse,
            'mae': avg_mae,
            'mape': avg_mape,
            'time': elapsed_config,
            'params': config['params']
        })
        
        print(f"   ⏱️ Temps total config: {elapsed_config:.1f}s")
    
    # Trouver la meilleure config
    best_xgb = min(xgb_results, key=lambda x: x['mape'])
    
    print(f"\n{'='*70}")
    print(f"🏆 MEILLEURE CONFIG XGBOOST")
    print(f"{'='*70}")
    print(f"Config: {best_xgb['name']}")
    print(f"RMSE: {best_xgb['rmse']:.2f}")
    print(f"MAE: {best_xgb['mae']:.2f}")
    print(f"MAPE: {best_xgb['mape']:.2f}%")
    
    # Entraîner sur full dataset avec meilleure config
    print(f"\n⏳ Entraînement final sur full dataset...")
    start_final = time.time()
    
    best_xgb_model = xgb.XGBRegressor(
        **best_xgb['params'],
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    best_xgb_model.fit(X_full, y_full)
    predictions_xgb_tuned = best_xgb_model.predict(X_test)
    predictions_xgb_tuned = np.maximum(predictions_xgb_tuned, 0)
    
    elapsed_final = time.time() - start_final
    print(f"   ⏱️ Training time: {elapsed_final:.2f}s")
    
    # Save
    joblib.dump(best_xgb_model, TUNED_MODELS_PATH / 'xgboost_tuned.pkl')
    joblib.dump(best_xgb['params'], TUNED_MODELS_PATH / 'xgboost_best_params.pkl')
    print(f"   💾 Modèle sauvegardé: {TUNED_MODELS_PATH / 'xgboost_tuned.pkl'}")
    
    xgb_success = True
    
except Exception as e:
    print(f"❌ Erreur XGBoost tuning: {e}")
    xgb_success = False

# ============================================
# 6. LIGHTGBM HYPERPARAMETER TUNING
# ============================================
print("\n" + "="*70)
print("2️⃣ LIGHTGBM - HYPERPARAMETER TUNING")
print("="*70)

try:
    import lightgbm as lgb
    
    print("\n🔍 Test de plusieurs configurations LightGBM...")
    
    lgb_configs = [
        {
            'name': 'Config 1 (Baseline)',
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0,
                'reg_lambda': 1,
            }
        },
        {
            'name': 'Config 2 (Deep Trees)',
            'params': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0,
                'reg_lambda': 1,
            }
        },
        {
            'name': 'Config 3 (Regularized)',
            'params': {
                'n_estimators': 250,
                'max_depth': 7,
                'learning_rate': 0.08,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_samples': 30,
                'reg_alpha': 0.5,
                'reg_lambda': 2,
            }
        },
        {
            'name': 'Config 4 (Fast Learning)',
            'params': {
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.15,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_samples': 15,
                'reg_alpha': 0,
                'reg_lambda': 1,
            }
        },
        {
            'name': 'Config 5 (Aggressive)',
            'params': {
                'n_estimators': 400,
                'max_depth': 9,
                'learning_rate': 0.03,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_samples': 25,
                'reg_alpha': 0.3,
                'reg_lambda': 1.5,
            }
        }
    ]
    
    lgb_results = []
    
    for config in lgb_configs:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {config['name']}")
        print(f"{'='*70}")
        
        start_config = time.time()
        
        model = lgb.LGBMRegressor(
            **config['params'],
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        avg_rmse, avg_mae, avg_mape = cross_validate_model(model, X_full, y_full, tscv, config['name'])
        
        elapsed_config = time.time() - start_config
        
        lgb_results.append({
            'name': config['name'],
            'rmse': avg_rmse,
            'mae': avg_mae,
            'mape': avg_mape,
            'time': elapsed_config,
            'params': config['params']
        })
        
        print(f"   ⏱️ Temps total config: {elapsed_config:.1f}s")
    
    # Trouver la meilleure config
    best_lgb = min(lgb_results, key=lambda x: x['mape'])
    
    print(f"\n{'='*70}")
    print(f"🏆 MEILLEURE CONFIG LIGHTGBM")
    print(f"{'='*70}")
    print(f"Config: {best_lgb['name']}")
    print(f"RMSE: {best_lgb['rmse']:.2f}")
    print(f"MAE: {best_lgb['mae']:.2f}")
    print(f"MAPE: {best_lgb['mape']:.2f}%")
    
    # Entraîner sur full dataset
    print(f"\n⏳ Entraînement final sur full dataset...")
    start_final = time.time()
    
    best_lgb_model = lgb.LGBMRegressor(
        **best_lgb['params'],
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    best_lgb_model.fit(X_full, y_full)
    predictions_lgb_tuned = best_lgb_model.predict(X_test)
    predictions_lgb_tuned = np.maximum(predictions_lgb_tuned, 0)
    
    elapsed_final = time.time() - start_final
    print(f"   ⏱️ Training time: {elapsed_final:.2f}s")
    
    # Save
    joblib.dump(best_lgb_model, TUNED_MODELS_PATH / 'lightgbm_tuned.pkl')
    joblib.dump(best_lgb['params'], TUNED_MODELS_PATH / 'lightgbm_best_params.pkl')
    print(f"   💾 Modèle sauvegardé: {TUNED_MODELS_PATH / 'lightgbm_tuned.pkl'}")
    
    lgb_success = True
    
except Exception as e:
    print(f"❌ Erreur LightGBM tuning: {e}")
    lgb_success = False

# ============================================
# 7. ENSEMBLE TUNED MODELS
# ============================================
print("\n" + "="*70)
print("🎯 ENSEMBLE DES MODÈLES TUNÉS")
print("="*70)

if xgb_success and lgb_success:
    print("\n⏳ Création ensemble tuné...")
    predictions_ensemble_tuned = (predictions_xgb_tuned + predictions_lgb_tuned) / 2
    predictions_ensemble_tuned = np.maximum(predictions_ensemble_tuned, 0)
    
    # Save submissions
    submission = test_full[['id']].copy()
    
    submission['sales'] = predictions_ensemble_tuned
    submission.to_csv(TUNED_MODELS_PATH / 'submission_ensemble_tuned.csv', index=False)
    print("   ✅ submission_ensemble_tuned.csv")
    
    submission['sales'] = predictions_xgb_tuned
    submission.to_csv(TUNED_MODELS_PATH / 'submission_xgboost_tuned.csv', index=False)
    print("   ✅ submission_xgboost_tuned.csv")
    
    submission['sales'] = predictions_lgb_tuned
    submission.to_csv(TUNED_MODELS_PATH / 'submission_lightgbm_tuned.csv', index=False)
    print("   ✅ submission_lightgbm_tuned.csv")

# ============================================
# 8. COMPARISON: BEFORE vs AFTER TUNING
# ============================================
print("\n" + "="*70)
print("📊 COMPARAISON: AVANT vs APRÈS TUNING")
print("="*70)

if xgb_success and lgb_success:
    # Charger les résultats précédents
    all_results = pd.read_csv(MODELS_PATH / 'all_models_results.csv')
    
    # Modèles originaux
    original_ensemble = all_results[all_results['model'] == 'Ensemble'].iloc[0]
    original_xgb = all_results[all_results['model'] == 'XGBoost'].iloc[0]
    
    print(f"\n📈 XGBOOST:")
    print(f"   Avant tuning:  MAPE = {original_xgb['mape']:.2f}%")
    print(f"   Après tuning:  MAPE = {best_xgb['mape']:.2f}%")
    improvement_xgb = ((original_xgb['mape'] - best_xgb['mape']) / original_xgb['mape']) * 100
    print(f"   Amélioration: {improvement_xgb:+.1f}%")
    
    print(f"\n📈 LIGHTGBM:")
    print(f"   Après tuning:  MAPE = {best_lgb['mape']:.2f}%")
    
    # Ensemble estimate (moyenne des MAPE)
    ensemble_tuned_mape = (best_xgb['mape'] + best_lgb['mape']) / 2
    print(f"\n📈 ENSEMBLE:")
    print(f"   Avant tuning:  MAPE = {original_ensemble['mape']:.2f}%")
    print(f"   Après tuning (estimé):  MAPE = {ensemble_tuned_mape:.2f}%")
    improvement_ensemble = ((original_ensemble['mape'] - ensemble_tuned_mape) / original_ensemble['mape']) * 100
    print(f"   Amélioration estimée: {improvement_ensemble:+.1f}%")

# ============================================
# 9. VISUALIZATIONS
# ============================================
print("\n" + "="*70)
print("📈 GÉNÉRATION DES VISUALISATIONS")
print("="*70)

if xgb_success:
    import matplotlib.pyplot as plt
    
    # XGBoost configs comparison
    print("\n📊 1. Comparaison configs XGBoost...")
    xgb_df = pd.DataFrame(xgb_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].barh(xgb_df['name'], xgb_df['rmse'], color='steelblue')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('XGBoost - RMSE', fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    
    axes[1].barh(xgb_df['name'], xgb_df['mae'], color='coral')
    axes[1].set_xlabel('MAE')
    axes[1].set_title('XGBoost - MAE', fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    
    axes[2].barh(xgb_df['name'], xgb_df['mape'], color='green')
    axes[2].set_xlabel('MAPE (%)')
    axes[2].set_title('XGBoost - MAPE', fontweight='bold')
    axes[2].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / '13_xgboost_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 13_xgboost_tuning.png")

if lgb_success:
    # LightGBM configs comparison
    print("\n📊 2. Comparaison configs LightGBM...")
    lgb_df = pd.DataFrame(lgb_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].barh(lgb_df['name'], lgb_df['rmse'], color='steelblue')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('LightGBM - RMSE', fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    
    axes[1].barh(lgb_df['name'], lgb_df['mae'], color='coral')
    axes[1].set_xlabel('MAE')
    axes[1].set_title('LightGBM - MAE', fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    
    axes[2].barh(lgb_df['name'], lgb_df['mape'], color='green')
    axes[2].set_xlabel('MAPE (%)')
    axes[2].set_title('LightGBM - MAPE', fontweight='bold')
    axes[2].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / '14_lightgbm_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 14_lightgbm_tuning.png")

# ============================================
# 10. SUMMARY
# ============================================
print("\n" + "="*70)
print("📊 RÉSUMÉ DU TUNING")
print("="*70)

print(f"""
🔬 Configurations testées:
   - XGBoost: {len(xgb_configs)} configs
   - LightGBM: {len(lgb_configs)} configs
   - Total: {len(xgb_configs) + len(lgb_configs)} configurations

💾 Fichiers sauvegardés:
   - models/tuned/xgboost_tuned.pkl
   - models/tuned/lightgbm_tuned.pkl
   - models/tuned/xgboost_best_params.pkl
   - models/tuned/lightgbm_best_params.pkl
   - models/tuned/submission_xgboost_tuned.csv
   - models/tuned/submission_lightgbm_tuned.csv
   - models/tuned/submission_ensemble_tuned.csv
""")

print("\n" + "="*70)
print("✅ HYPERPARAMETER TUNING TERMINÉ !")
print(f"⏰ Fin: {datetime.now().strftime('%H:%M:%S')}")
print("="*70)
