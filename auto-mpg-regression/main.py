import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Фиксируем random_state, чтобы результаты можно было
# воспроизводить при повторном запуске программы.
RANDOM_STATE = 42

# Размер тестовой выборки.
# 20% данных оставляем для финальной проверки модели.
TEST_SIZE = 0.2

# Количество блоков при кросс-валидации.
CV_FOLDS = 5

# Создаём папку images, если её ещё нет.
# Все графики будут сохраняться туда.
os.makedirs("images", exist_ok=True)

DATASET_PATH = "data/auto-mpg.data"

column_names = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name"
]

rows = []

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=8)

        rows.append(parts)

# Создаём DataFrame.
df = pd.DataFrame(
    rows,
    columns=column_names
)

# Заменяем "?" на пропущенное значение.
df = df.replace("?", np.nan)

# Преобразуем числовые столбцы в числовой тип.
numeric_columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")


print("\nПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ")


print("\nПервые 5 строк датасета:")
print(df.head())

print("\nРазмерность датасета:")
print(df.shape)

print("\nТипы данных:")
print(df.dtypes)

print("\nКоличество уникальных значений:")
print(df.nunique())

# Считаем количество пропущенных значений в каждом столбце.
missing_values = df.isnull().sum()

print("\nКоличество пропусков:")
print(missing_values)

print("ПРОВЕРКА ДУБЛИКАТОВ")

# duplicated() возвращает True для строк,
# которые полностью повторяют предыдущую строку.
duplicate_count = df.duplicated().sum()

print(f"\nКоличество полных дубликатов: {duplicate_count}")


print("\nПОДГОТОВКА И ПРЕДОБРАБОТКА ДАННЫХ")

print(f"\nПропуски в horsepower: {df["horsepower"].isnull().sum()}")

# Удаление названия машин
X = df.drop(columns=["mpg", "car_name"])
y = df["mpg"]


# X_train, y_train — данные для обучения моделей.
# X_test, y_test   — данные для финальной проверки.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")


# Получаем названия числовых признаков.
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# Получаем названия категориальных признаков.
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("\nЧисловые признаки:")
print(numeric_features)

print("\nКатегориальные признаки:")
print(categorical_features)

# ------------------------------------------------------------
# Числовые признаки
# ------------------------------------------------------------

# Для числовых признаков выполняем заполнение пропусков
# медианой и масштабирование.
#
# StandardScaler преобразует признаки таким образом,
# чтобы они имели примерно среднее значение 0
# и стандартное отклонение 1.
#
# Это особенно важно для линейных моделей,
# например Linear Regression и Ridge.

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# Категориальные признаки
# ------------------------------------------------------------

# Для категориальных признаков используем OneHotEncoder.
#
# Он превращает категориальные значения в набор бинарных
# признаков 0/1.
#
# handle_unknown="ignore" нужен для ситуации,
# когда в тестовой выборке встретится категория,
# которой не было в обучающей выборке.

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ------------------------------------------------------------
# Объединяем обработку
# ------------------------------------------------------------

# ColumnTransformer позволяет применить:
#
# SimpleImputer + StandardScaler -> к числовым признакам
# OneHotEncoder  -> к категориальным признакам

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


print("\nСОЗДАНИЕ МОДЕЛЕЙ РЕГРЕССИИ")


models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(),

    "Random Forest": RandomForestRegressor(
        random_state=RANDOM_STATE
    )
}

print("\nИспользуемые модели:")

for model_name in models:
    print(f"- {model_name}")


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

# Mean Absolute Error — средняя абсолютная ошибка.
#
# Показывает, насколько в среднем предсказание модели
# отличается от реального значения.
#
# Чем меньше MAE, тем лучше.

scoring = {
    "MAE": "neg_mean_absolute_error",

    # MSE — средняя квадратичная ошибка.
    # Чем меньше, тем лучше.
    "MSE": "neg_mean_squared_error",

    # R2 — коэффициент детерминации.
    # Чем ближе к 1, тем лучше.
    "R2": "r2"
}

print("\nКРОСС-ВАЛИДАЦИЯ МОДЕЛЕЙ")

# Здесь будем сохранять результаты.
cv_results = []


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Модель: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


    # --------------------------------------------------------
    # Кросс-валидация
    # --------------------------------------------------------

    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring=scoring,
        return_train_score=False
    )


    # --------------------------------------------------------
    # Средние значения метрик
    # --------------------------------------------------------

    # В sklearn ошибки возвращаются со знаком минус,
    # потому что система поиска гиперпараметров
    # максимизирует score.
    mean_mae = -cv_result["test_MAE"].mean()

    mean_mse = -cv_result["test_MSE"].mean()

    # Из MSE можно получить RMSE.
    mean_rmse = np.sqrt(mean_mse)

    mean_r2 = cv_result["test_R2"].mean()


    # Сохраняем результат.
    cv_results.append(
        {
            "Model": model_name,
            "MAE": mean_mae,
            "MSE": mean_mse,
            "RMSE": mean_rmse,
            "R2": mean_r2
        }
    )


    # Выводим результат.
    print(f"Средний MAE:  {mean_mae:.4f}")
    print(f"Средний MSE:  {mean_mse:.4f}")
    print(f"Средний RMSE: {mean_rmse:.4f}")
    print(f"Средний R²:   {mean_r2:.4f}")


print("\nСРАВНЕНИЕ МОДЕЛЕЙ")

# Преобразуем результаты в DataFrame.
results_df = pd.DataFrame(cv_results)

print("\nРезультаты кросс-валидации:")
print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Выбор лучшей модели
# ------------------------------------------------------------

# В качестве основной метрики используем R².
#
# Чем больше R², тем лучше модель объясняет вариацию
# целевой переменной.
#
# idxmax() находит строку с максимальным значением R².

best_model_name = results_df.loc[
    results_df["R2"].idxmax(),
    "Model"
]


print(f"\nЛучшая модель по среднему R²: {best_model_name}")

# Получаем объект выбранной модели.
best_model = models[best_model_name]


# Создаём Pipeline для выбранной модели.
#
# Pipeline гарантирует, что при каждом обучении:
#
# 1. Предобработка выполняется только на соответствующей
#    обучающей части данных.
#
# 2. После этого обучается модель.
#
# Это особенно важно при кросс-валидации, поскольку
# предотвращает утечку информации между частями данных.

best_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            best_model
        )
    ]
)


# GRID SEARCH

print("\nПОДБОР ГИПЕРПАРАМЕТРОВ")

param_grids = {
    "Linear Regression": {
        "model__fit_intercept": [
            True,
            False
        ]
    },

    "Ridge": {
        "model__alpha": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    },

    # n_estimators — количество деревьев;
    # max_depth — максимальная глубина дерева;
    # min_samples_split — минимальное количество объектов
    # для разделения внутреннего узла;
    # min_samples_leaf — минимальное количество объектов
    # в листе.
    "Random Forest": {
        "model__n_estimators": [
            50,
            100,
            200
        ],

        "model__max_depth": [
            None,
            5,
            10,
            15
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ]
    }
}


# Получаем сетку параметров для выбранной модели.
best_param_grid = param_grids[best_model_name]


print("\nМодель для подбора:")
print(best_model_name)

# ------------------------------------------------------------
# GridSearchCV
# ------------------------------------------------------------

# GridSearchCV перебирает все комбинации параметров
# из указанной сетки.
#
# Для каждой комбинации выполняется кросс-валидация.
#
# scoring="r2" означает, что лучшей считается комбинация,
# которая имеет максимальное среднее значение R².

grid_search = GridSearchCV(
    estimator=best_pipeline,
    param_grid=best_param_grid,
    cv=CV_FOLDS,
    scoring="r2",
    n_jobs=-1,
    return_train_score=False
)


# Запускаем подбор параметров.
grid_search.fit(
    X_train,
    y_train
)

print("\nЛучшие параметры:")
print(grid_search.best_params_)

print("\nЛучший средний R²:")
print(
    f"{grid_search.best_score_:.4f}"
)

final_model = grid_search.best_estimator_

y_pred = final_model.predict(X_test)

print("\nФИНАЛЬНЫЕ МЕТРИКИ")

# MAE.
#
# Показывает среднюю абсолютную ошибку предсказания.
# Единица измерения совпадает с mpg.

final_mae = mean_absolute_error(
    y_test,
    y_pred
)

final_mse = mean_squared_error(
    y_test,
    y_pred
)

final_rmse = np.sqrt(
    final_mse
)

final_r2 = r2_score(
    y_test,
    y_pred
)

print(f"MAE:  {final_mae:.4f}")
print(f"MSE:  {final_mse:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"R²:   {final_r2:.4f}")

# Создаём отдельную таблицу для удобного анализа результатов.

prediction_df = pd.DataFrame(
    {
        "Actual MPG": y_test.values,
        "Predicted MPG": y_pred,
        "Absolute Error": np.abs(
            y_test.values - y_pred
        )
    }
)

# График
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Строим идеальную линию:
# если модель предсказывает идеально,
# все точки должны находиться на этой линии.
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Фактические и предсказанные значения MPG")
plt.xlabel("Фактический MPG")
plt.ylabel("Предсказанный MPG")

plt.tight_layout()

plt.savefig(
    "images/actual_vs_predicted.png",
    dpi=300
)

plt.show()

