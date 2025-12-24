"""
Store Sales Forecasting - Advanced Models
XGBoost, LightGBM, LSTM et Ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Paths
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports/figures")

print("="*60)
print("🚀 ADVANCED MODELS - TIME SERIES FORECASTING")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📦 Chargement des données...")

df = pd.read_csv(PROCESSED_PATH / "train_processed.csv", parse_dates=['date'])
print(f"✅ Data shape: {df.shape}")

# ============================================
# 2. SELECT SAME SUBSET AS BASELINE
# ============================================
print("\n" + "="*60)
print("🎯 SÉLECTION DU MÊME SUBSET")
print("="*60)

top_family = df.groupby('family')['sales'].sum().idxmax()
top_store = df.groupby('store_nbr')['sales'].sum().idxmax()

subset = df[(df['family'] == top_family) & (df['store_nbr'] == top_store)].copy()
subset = subset.sort_values('date').reset_index(drop=True)

print(f"🏆 Famille: {top_family}")
print(f"🏪 Magasin: {top_store}")
print(f"📊 Shape: {subset.shape}")

# Train/Test split
test_days = 30
train = subset[:-test_days].copy()
test = subset[-test_days:].copy()

print(f"\n✅ Train: {len(train)} jours")
print(f"✅ Test: {len(test)} jours")

# ============================================
# 3. PREPARE FEATURES
# ============================================
print("\n" + "="*60)
print("🔧 PRÉPARATION DES FEATURES")
print("="*60)

# Features à utiliser
feature_cols = [
    # Temporal
    'dayofweek', 'day', 'month', 'quarter', 'dayofyear', 'weekofyear',
    'is_weekend', 'is_month_start', 'is_month_end', 'is_payday',
    # Lags
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    # Rolling
    'sales_rolling_mean_7', 'sales_rolling_std_7',
    'sales_rolling_mean_14', 'sales_rolling_std_14',
    'sales_rolling_mean_30', 'sales_rolling_std_30',
    # External
    'is_holiday', 'has_promotion', 'onpromotion',
    'cluster', 'transactions',
    # Encoded
    'type_encoded'
]

# Filtrer features disponibles
feature_cols = [col for col in feature_cols if col in train.columns]
print(f"\n📊 {len(feature_cols)} features sélectionnées")

# Préparer X, y
X_train = train[feature_cols].fillna(0)
y_train = train['sales']
X_test = test[feature_cols].fillna(0)
y_test = test['sales']

print(f"✅ X_train: {X_train.shape}")
print(f"✅ X_test: {X_test.shape}")

# ============================================
# 4. EVALUATION FUNCTION
# ============================================

def evaluate_model(y_true, y_pred, model_name):
    """Calcule les métriques"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    print(f"\n📊 {model_name}:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE:  {mae:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}

results = []

# ============================================
# 5. MODEL 1: XGBOOST
# ============================================
print("\n" + "="*60)
print("1️⃣ XGBOOST")
print("="*60)

try:
    import xgboost as xgb
    import time
    
    print("\n⏳ Entraînement XGBoost...")
    start = time.time()
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model_xgb.fit(X_train, y_train, verbose=False)
    pred_xgb = model_xgb.predict(X_test)
    pred_xgb = np.maximum(pred_xgb, 0)
    
    elapsed = time.time() - start
    print(f"⏱️ Temps: {elapsed:.2f}s")
    
    results.append(evaluate_model(y_test, pred_xgb, 'XGBoost'))
    
    # Save model
    import joblib
    joblib.dump(model_xgb, MODELS_PATH / 'xgboost_model.pkl')
    print("💾 Modèle sauvegardé: xgboost_model.pkl")
    
except Exception as e:
    print(f"❌ Erreur XGBoost: {e}")
    pred_xgb = np.full(len(y_test), y_train.mean())

# ============================================
# 6. MODEL 2: LIGHTGBM
# ============================================
print("\n" + "="*60)
print("2️⃣ LIGHTGBM")
print("="*60)

try:
    import lightgbm as lgb
    import time
    
    print("\n⏳ Entraînement LightGBM...")
    start = time.time()
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model_lgb.fit(X_train, y_train)
    pred_lgb = model_lgb.predict(X_test)
    pred_lgb = np.maximum(pred_lgb, 0)
    
    elapsed = time.time() - start
    print(f"⏱️ Temps: {elapsed:.2f}s")
    
    results.append(evaluate_model(y_test, pred_lgb, 'LightGBM'))
    
    # Save model
    joblib.dump(model_lgb, MODELS_PATH / 'lightgbm_model.pkl')
    print("💾 Modèle sauvegardé: lightgbm_model.pkl")
    
except Exception as e:
    print(f"❌ Erreur LightGBM: {e}")
    pred_lgb = np.full(len(y_test), y_train.mean())

# ============================================
# 7. MODEL 3: LSTM
# ============================================
print("\n" + "="*60)
print("3️⃣ LSTM (Deep Learning)")
print("="*60)

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import StandardScaler
    import time
    
    print("\n⏳ Entraînement LSTM...")
    start = time.time()
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    
    # Reshape pour LSTM [samples, timesteps, features]
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Build model
    model_lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model_lstm.compile(optimizer='adam', loss='mse')
    
    # Train
    model_lstm.fit(
        X_train_lstm, y_train_scaled,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_split=0.2
    )
    
    # Predict
    pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)
    pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled).flatten()
    pred_lstm = np.maximum(pred_lstm, 0)
    
    elapsed = time.time() - start
    print(f"⏱️ Temps: {elapsed:.2f}s")
    
    results.append(evaluate_model(y_test, pred_lstm, 'LSTM'))
    
    # Save model
    model_lstm.save(MODELS_PATH / 'lstm_model.h5')
    joblib.dump(scaler_X, MODELS_PATH / 'scaler_X.pkl')
    joblib.dump(scaler_y, MODELS_PATH / 'scaler_y.pkl')
    print("💾 Modèle sauvegardé: lstm_model.h5")
    
except Exception as e:
    print(f"❌ Erreur LSTM: {e}")
    pred_lstm = np.full(len(y_test), y_train.mean())

# ============================================
# 8. MODEL 4: ENSEMBLE
# ============================================
print("\n" + "="*60)
print("4️⃣ ENSEMBLE (Moyenne des meilleurs)")
print("="*60)

try:
    # Moyenne simple des 3 modèles
    pred_ensemble = (pred_xgb + pred_lgb + pred_lstm) / 3
    pred_ensemble = np.maximum(pred_ensemble, 0)
    
    results.append(evaluate_model(y_test, pred_ensemble, 'Ensemble'))
    
except Exception as e:
    print(f"❌ Erreur Ensemble: {e}")

# ============================================
# 9. RESULTS COMPARISON
# ============================================
print("\n" + "="*60)
print("📊 COMPARAISON DES MODÈLES AVANCÉS")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rmse')

print("\n🏆 Classement par RMSE:")
print(results_df.to_string(index=False))

# Charger les résultats baseline
baseline_df = pd.read_csv(MODELS_PATH / 'baseline_results.csv')

# Combiner
all_results = pd.concat([baseline_df, results_df], ignore_index=True)
all_results = all_results.sort_values('rmse')

print("\n🎯 TOUS LES MODÈLES (Baseline + Advanced):")
print(all_results.to_string(index=False))

# Save
all_results.to_csv(MODELS_PATH / 'all_models_results.csv', index=False)
print(f"\n💾 Résultats sauvegardés: {MODELS_PATH / 'all_models_results.csv'}")

# ============================================
# 10. VISUALIZATIONS
# ============================================
print("\n" + "="*60)
print("📈 GÉNÉRATION DES VISUALISATIONS")
print("="*60)

# Plot 1: Advanced models predictions
print("\n📊 1. Prédictions modèles avancés...")

fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(test['date'], y_test, 'o-', label='Actual', linewidth=2, markersize=6, color='black')

try:
    ax.plot(test['date'], pred_xgb, '--', label='XGBoost', alpha=0.8, linewidth=2)
except:
    pass

try:
    ax.plot(test['date'], pred_lgb, '--', label='LightGBM', alpha=0.8, linewidth=2)
except:
    pass

try:
    ax.plot(test['date'], pred_lstm, '--', label='LSTM', alpha=0.8, linewidth=2)
except:
    pass

try:
    ax.plot(test['date'], pred_ensemble, '--', label='Ensemble', alpha=0.8, linewidth=2, color='red')
except:
    pass

ax.set_title(f'Advanced Models - Prédictions vs Actual\n{top_family} @ Store {top_store}', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '11_advanced_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 11_advanced_predictions.png")

# Plot 2: All models comparison
print("\n📊 2. Comparaison complète...")

fig, ax = plt.subplots(figsize=(12, 8))

models = all_results['model']
rmse_values = all_results['rmse']

colors = ['green' if 'XGBoost' in m or 'LightGBM' in m or 'LSTM' in m or 'Ensemble' in m 
          else 'steelblue' for m in models]

ax.barh(models, rmse_values, color=colors, alpha=0.7)
ax.set_xlabel('RMSE', fontsize=12)
ax.set_title('Comparaison de tous les modèles (RMSE)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Advanced Models'),
    Patch(facecolor='steelblue', alpha=0.7, label='Baseline Models')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(REPORTS_PATH / '12_all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 12_all_models_comparison.png")

# ============================================
# 11. BEST MODEL
# ============================================
print("\n" + "="*60)
print("🏆 MEILLEUR MODÈLE")
print("="*60)

best_model = all_results.iloc[0]
print(f"\n🥇 Modèle: {best_model['model']}")
print(f"   RMSE: {best_model['rmse']:.2f}")
print(f"   MAE:  {best_model['mae']:.2f}")
print(f"   MAPE: {best_model['mape']:.2f}%")

# Improvement over baseline
best_baseline = baseline_df.iloc[0]
improvement = ((best_baseline['rmse'] - best_model['rmse']) / best_baseline['rmse']) * 100

print(f"\n📈 Amélioration vs meilleure baseline: {improvement:.1f}%")

print("\n" + "="*60)
print("✅ ADVANCED MODELS TERMINÉ !")
print("="*60)
