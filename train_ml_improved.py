import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df, target_horizon=30, target_threshold=0.002):
    """Подготовка фичей для ML

    Args:
        target_horizon: горизонт предсказания в минутах (30 вместо 15)
        target_threshold: порог роста для BUY сигнала (0.2% вместо 0.5%)
    """

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

    # Создаем целевую переменную
    # BUY = 1 если цена вырастет > target_threshold через N минут
    target_col = f'future_return_{target_horizon}min'

    if target_col not in df.columns:
        print(f"❌ Target column {target_col} not found!")
        return None, None, None

    # СНИЖЕН ПОРОГ: 0.2% вместо 0.5%
    df['signal'] = (df[target_col] > target_threshold).astype(int)

    # Удаляем NaN
    df_clean = df[available_features + ['signal']].dropna()

    X = df_clean[available_features]
    y = df_clean['signal']

    return X, y, available_features


def balance_classes(X, y):
    """Балансировка классов через undersampling мажоритарного класса"""

    # Объединяем обратно для ресемплинга
    df_combined = pd.concat([X, y], axis=1)

    # Разделяем на классы
    df_majority = df_combined[df_combined['signal'] == 0]
    df_minority = df_combined[df_combined['signal'] == 1]

    minority_count = len(df_minority)
    majority_count = len(df_majority)

    print(f"Before balancing: HOLD={majority_count}, BUY={minority_count}")

    if minority_count == 0:
        print("⚠️  No BUY samples found!")
        return X, y

    # Undersampling: берем в 3 раза больше HOLD, чем BUY (вместо 1:1)
    # Это даст модели больше примеров, но сохранит баланс
    target_majority_count = min(majority_count, minority_count * 3)

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=target_majority_count,
        random_state=42
    )

    # Объединяем
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"After balancing: HOLD={len(df_balanced[df_balanced['signal']==0])}, BUY={len(df_balanced[df_balanced['signal']==1])}")

    X_balanced = df_balanced.drop('signal', axis=1)
    y_balanced = df_balanced['signal']

    return X_balanced, y_balanced


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

    # Подготавливаем фичи с новыми параметрами
    X, y, features = prepare_features(
        df,
        target_horizon=30,      # 30 минут вместо 15
        target_threshold=0.002  # 0.2% вместо 0.5%
    )

    if X is None:
        return None

    print(f"Features: {len(features)}")
    print(f"Samples: {len(X):,}")
    print(f"Buy signals: {y.sum():,} ({y.mean()*100:.1f}%)")

    if y.sum() < 50:
        print(f"⚠️  Too few buy signals, skipping...")
        return None

    # БАЛАНСИРОВКА КЛАССОВ
    X_balanced, y_balanced = balance_classes(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    # Обучаем модель с настройкой для сбалансированных классов
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,      # Меньше для большей гибкости
        min_samples_leaf=10,       # Меньше для большей гибкости
        class_weight='balanced',   # Автоматическая балансировка весов
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
    y_pred_proba = model.predict_proba(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['HOLD', 'BUY']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"             Predicted")
    print(f"              HOLD  BUY")
    print(f"Actual HOLD   {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       BUY    {cm[1][0]:5d} {cm[1][1]:5d}")

    # Проверка распределения вероятностей
    buy_probas = y_pred_proba[:, 1]
    print(f"\nBUY probability distribution on test set:")
    print(f"  Min:    {buy_probas.min():.3f}")
    print(f"  Max:    {buy_probas.max():.3f}")
    print(f"  Mean:   {buy_probas.mean():.3f}")
    print(f"  Median: {np.median(buy_probas):.3f}")

    # Сколько сигналов будет при разных порогах
    print(f"\nPredicted BUY signals at different thresholds:")
    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
        count = (buy_probas > threshold).sum()
        pct = count / len(buy_probas) * 100
        print(f"  >{threshold:.2f}: {count:4d} ({pct:.1f}%)")

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
        'model': model,
        'max_proba': buy_probas.max(),
        'mean_proba': buy_probas.mean(),
    }


def main():
    """Обучение моделей для всех акций"""

    print("=" * 80)
    print("IMPROVED ML MODEL TRAINING")
    print("=" * 80)
    print("\nImprovements:")
    print("  - Lower threshold: 0.2% growth (was 0.5%)")
    print("  - Longer horizon: 30 minutes (was 15)")
    print("  - Balanced classes with 3:1 ratio")
    print("  - class_weight='balanced' in Random Forest")
    print()

    # Список акций с данными
    data_path = 'data'
    tickers = []

    for file in os.listdir(data_path):
        if file.endswith('_historical_data.xlsx'):
            ticker = file.replace('_historical_data.xlsx', '')
            tickers.append(ticker)

    print(f"Found data for: {', '.join(tickers)}\n")

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
            'Max Proba': f"{r['max_proba']:.3f}",
            'Mean Proba': f"{r['mean_proba']:.3f}",
        } for r in results])

        print(results_df.to_string(index=False))

        print(f"\n✅ Trained {len(results)} models")
        print(f"📁 Models saved in 'models/' directory")

        # Проверка, что модели будут торговать
        print("\n" + "=" * 80)
        print("TRADING READINESS CHECK:")
        print("=" * 80)

        for r in results:
            if r['max_proba'] > 0.55:
                status = "✅ READY"
            elif r['max_proba'] > 0.45:
                status = "⚠️  MARGINAL"
            else:
                status = "❌ LOW CONFIDENCE"

            print(f"{r['ticker']}: Max confidence {r['max_proba']:.3f} - {status}")

    else:
        print("\n❌ No models trained!")


if __name__ == "__main__":
    main()