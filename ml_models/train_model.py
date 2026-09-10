import pandas as pd
import numpy as np
import os

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
import pickle
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n" + "=" * 65)
print(" ML MODEL TRAINING")
print("=" * 65)

csv_path = os.path.join(SCRIPT_DIR, 'user_calorie_dataset.csv')
df = pd.read_csv('user_calorie_dataset.csv')

df.info()

# Input features (what user provides + derived features)
feature_cols = ['age', 'gender', 'height_cm', 'weight_kg',
                'activity_level', 'purpose',
                'bmi', 'body_fat_pct', 'lean_body_mass_kg']

target_col = 'calories_needed'

# Create feature matrix
X = df[feature_cols].copy()
y = df[target_col].copy()

# Also prepare multi-output targets for macros
y_protein = df['protein_g']
y_carbs   = df['carbs_g']
y_fat     = df['fat_g']

# --- Encode Categoricals ---
le_gender   = LabelEncoder()
le_activity = LabelEncoder()
le_purpose  = LabelEncoder()

X['gender']         = le_gender.fit_transform(X['gender'])
X['activity_level'] = le_activity.fit_transform(X['activity_level'])
X['purpose']        = le_purpose.fit_transform(X['purpose'])

# --- Add Interaction Features ---
X['bmi_x_activity']  = X['bmi'] * X['activity_level']
X['weight_x_height'] = X['weight_kg'] * X['height_cm']
X['age_x_bmi']       = X['age'] * X['bmi']
X['lbm_x_activity']  = X['lean_body_mass_kg'] * X['activity_level']

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Also split macro targets
_, _, y_prot_train, y_prot_test = train_test_split(
    X, y_protein, test_size=0.2, random_state=42)
_, _, y_carb_train, y_carb_test = train_test_split(
    X, y_carbs, test_size=0.2, random_state=42)
_, _, y_fat_train, y_fat_test = train_test_split(
    X, y_fat, test_size=0.2, random_state=42)

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")



models = {
    'Ridge Regression':          Ridge(alpha=1.0),
    'Lasso Regression':          Lasso(alpha=0.1),
    'KNN (k=7)':                 KNeighborsRegressor(n_neighbors=7),
    'Random Forest':             RandomForestRegressor(
                                    n_estimators=200, max_depth=15,
                                    min_samples_split=5, random_state=42,
                                    n_jobs=-1),
    'Extra Trees':               ExtraTreesRegressor(
                                    n_estimators=200, max_depth=15,
                                    random_state=42, n_jobs=-1),
    'Gradient Boosting':         GradientBoostingRegressor(
                                    n_estimators=300, max_depth=6,
                                    learning_rate=0.1, subsample=0.8,
                                    random_state=42),
}

results = []
print(f"\n{'Model':<28} {'MAE':>8} {'RMSE':>8} {'R²':>8} "
      f"{'MAPE%':>8} {'Time(s)':>8}")
print("─" * 72)

best_model = None
best_r2 = -999

for name, model in models.items():
    start = time.time()

    # Use scaled data for linear models, raw for tree-based
    if name in ['Ridge Regression', 'Lasso Regression',
                'KNN (k=7)', 'SVR (RBF)']:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    elapsed = time.time() - start

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred) * 100

    results.append({
        'Model': name, 'MAE': mae, 'RMSE': rmse,
        'R2': r2, 'MAPE': mape, 'Time': elapsed
    })

    print(f"{name:<28} {mae:>8.1f} {rmse:>8.1f} {r2:>8.4f} "
          f"{mape:>7.2f}% {elapsed:>7.2f}s")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)

print(f"\n BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")


print(f"\n{'─'*65}")
print(f"5-FOLD CROSS-VALIDATION — {best_model_name}")
print(f"{'─'*65}")

cv_scores = cross_val_score(
    best_model, X_train, y_train,
    cv=5, scoring='r2', n_jobs=-1
)
cv_mae = -cross_val_score(
    best_model, X_train, y_train,
    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
)

print(f"R² scores:  {cv_scores.round(4)}")
print(f"R² mean:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"MAE scores: {cv_mae.round(1)}")
print(f"MAE mean:   {cv_mae.mean():.1f} ± {cv_mae.std():.1f}")


if hasattr(best_model, 'feature_importances_'):
    print(f"\n{'─'*65}")
    print("FEATURE IMPORTANCE")
    print(f"{'─'*65}")

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    for _, row in importance.iterrows():
        bar = "█" * int(row['Importance'] * 100)
        print(f"  {row['Feature']:<22} {row['Importance']:.4f}  {bar}")


print(f"\n{'─'*65}")
print("TRAINING MACRO NUTRIENT MODELS")
print(f"{'─'*65}")

macro_models = {}
for target_name, yt_train, yt_test in [
    ('Protein (g)', y_prot_train, y_prot_test),
    ('Carbs (g)',   y_carb_train, y_carb_test),
    ('Fat (g)',     y_fat_train,  y_fat_test),
]:
    m = GradientBoostingRegressor(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42
    )
    m.fit(X_train, yt_train)
    pred = m.predict(X_test)

    r2  = r2_score(yt_test, pred)
    mae = mean_absolute_error(yt_test, pred)

    macro_models[target_name] = m
    print(f"  {target_name:<14}  R²={r2:.4f}  MAE={mae:.1f}")


artifacts = {
    'calorie_model': best_model,
    'protein_model': macro_models['Protein (g)'],
    'carbs_model':   macro_models['Carbs (g)'],
    'fat_model':     macro_models['Fat (g)'],
    'scaler': scaler,
    'le_gender': le_gender,
    'le_activity': le_activity,
    'le_purpose': le_purpose,
    'feature_cols': list(X.columns),
}

output_path = os.path.join(SCRIPT_DIR, 'trained_models.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\n All models saved to '{output_path}'")