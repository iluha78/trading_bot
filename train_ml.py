import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df, target_horizon=15):
    """Подготовка фичей для ML"""

    # Список фичей для обучения
    feature_columns = [
        'rsi', 'adx', 'macd', 'macd_signal', 'macd_hist',
        'ema_short', 'ema_medium', 'ema_long',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'volume_avg',
        'returns', 'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_10', 'volatility_20',
        'volume_ratio', 'volume_change',
        'high_low_ratio',
        'ema_spread_short', 'ema_spread_long'
    ]

    # Проверяем наличие колонок
    available_features = [col for col in feature_columns if col in df.columns]

    # Создаем целевую переменную (сигнал)
    # 1 = BUY (если цена вырастет > 0.5% через N минут)
    # 0 = HOLD/SELL
    target_col = f'future_return_{target_horizon}min'

    if target_col not in df.columns:
        print(f"❌ Target column {target_col} not found!")
        return None, None, None

    df['signal'] = (df[target_col] > 0.005).astype(int)  # >0.5% = BUY

    # Удаляем NaN
    df_clean = df[available_features + ['signal']].dropna()

    X = df_clean[available_features]
    y = df_clean['signal']

    return X, y, available_features


def train_model_for_ticker(ticker, data_path='data'):
    """Обучение модели для конкретной акции"""

    print(f"\n{'='*60}")
    print(f"Training model for {ticker}")
    print('='*60)

    # Загружаем данные
    try:
        df = pd.read_excel(f'{data_path}/{ticker}_historical_data.xlsx')
    except:
        try:
            df = pd.read_csv(f'{data_path}/{ticker}_historical_data.csv')
        except:
            print(f"❌ Data file not found for {ticker}")
            return None

    print(f"Loaded {len(df):,} rows")

    # Подготавливаем фичи
    X, y, features = prepare_features(df, target_horizon=15)

    if X is None:
        return None

    print(f"Features: {len(features)}")
    print(f"Samples: {len(X):,}")
    print(f"Buy signals: {y.sum():,} ({y.mean()*100:.1f}%)")

    if y.sum() < 100:
        print(f"⚠️  Too few buy signals, skipping...")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучаем модель
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Оценка
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"\nTrain accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 features:")
    print(feature_importance.head(10).to_string(index=False))

    # Предсказания на тесте
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['HOLD', 'BUY']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"             Predicted")
    print(f"              HOLD  BUY")
    print(f"Actual HOLD   {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       BUY    {cm[1][0]:5d} {cm[1][1]:5d}")

    # Сохраняем модель
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{ticker}_model.pkl'
    joblib.dump(model, model_path)

    # Сохраняем список фичей
    features_path = f'models/{ticker}_features.pkl'
    joblib.dump(features, features_path)

    # Сохраняем feature importance
    feature_importance.to_csv(f'models/{ticker}_feature_importance.csv', index=False)

    print(f"\n✅ Model saved to {model_path}")

    return {
        'ticker': ticker,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'cv_accuracy': cv_scores.mean(),
        'features': features,
        'feature_importance': feature_importance,
        'model': model
    }


def main():
    """Обучение моделей для всех акций"""

    print("=" * 80)
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 80)

    # Список акций
    tickers = ['SBER', 'TATN', 'VTBR', 'NVTK', 'PHOR', 'FIVE', 'MGNT']

    results = []

    for ticker in tickers:
        result = train_model_for_ticker(ticker)
        if result:
            results.append(result)

    # Итоговая статистика
    if results:
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY:")
        print("=" * 80)

        results_df = pd.DataFrame([{
            'Ticker': r['ticker'],
            'Train Acc': f"{r['train_accuracy']:.3f}",
            'Test Acc': f"{r['test_accuracy']:.3f}",
            'CV Acc': f"{r['cv_accuracy']:.3f}",
        } for r in results])

        print(results_df.to_string(index=False))

        print(f"\n✅ Trained {len(results)} models")
        print(f"📁 Models saved in 'models/' directory")
    else:
        print("\n❌ No models trained!")


if __name__ == "__main__":
    main()