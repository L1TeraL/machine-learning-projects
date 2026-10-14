import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier



# 1. Загрузка данных


df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

print("\nТипы данных:")
print(df.dtypes)

print("\nОсновная статистика:")
print(df.describe())

print("\nКоличество уникальных значений:")
print(df.nunique())

print("\nКоличество дубликатов:", df.duplicated().sum())



# 2. Анализ целевой переменной

churn_counts = df["Churn"].value_counts()
churn_percent = df["Churn"].value_counts(normalize=True) * 100

print("\nРаспределение целевой переменной:")
print(churn_counts)

print("\nРаспределение в процентах:")
print(churn_percent.round(2))

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(7, 5))

ax = sns.countplot(data=df, x="Churn")

plt.title("Распределение клиентов по целевой переменной Churn")
plt.xlabel("Churn")
plt.ylabel("Количество клиентов")

for i, value in enumerate(churn_counts):
    percent = churn_percent.iloc[i]
    ax.text(
        i,
        value + 100,
        f"{value}\n({percent:.1f}%)",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "images/churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# 3. Подготовка данных


data = df.copy()

# Идентификатор клиента не используется при обучении
data = data.drop(columns=["customerID"])

# TotalCharges изначально хранится как строка.
# Пустые строки преобразуются в NaN.
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Преобразуем целевую переменную в числа
data["Churn"] = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = data.drop(columns=["Churn"])
y = data["Churn"]

print("\nПропуски после преобразования:")
print(X.isna().sum()[X.isna().sum() > 0])



# 4. Разделение на обучающую и тестовую выборки


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

print("\nРаспределение классов в обучающей выборке:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True).round(4) * 100)



# 5. Обработка дисбаланса


print("\nРаспределение классов до балансировки:")
print(y_train.value_counts())

ros = RandomOverSampler(random_state=42) # увеличивает количество объектов меньшего класса за счёт повторного выбора существующих объектов

_, y_train_balanced = ros.fit_resample(
    X_train,
    y_train
)

print("\nРаспределение классов после балансировки:")
print(y_train_balanced.value_counts())



# 6. Предобработка данных


numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),           # замена пропуска медианой
        ("scaler", StandardScaler())                             # стандартизация к одному масштабу
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(                                      # категории в числовые признаки
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)



# 7. Проверка выбросов


print("\nПроверка выбросов методом IQR:") # поиск выбросов и аномалий

for column in numeric_features:
    q1 = X_train[column].quantile(0.25)
    q3 = X_train[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (X_train[column] < lower_bound)
        | (X_train[column] > upper_bound)
    ).sum()

    print(f"{column}: {outliers}")



# 8. Создание моделей


models = {
    "KNN": KNeighborsClassifier(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}



# 9. Pipeline для моделей


pipelines = {}

for name, model in models.items():

    pipelines[name] = ImbPipeline(
        steps=[
            (
                "sampler",
                RandomOverSampler(random_state=42)
            ),
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



# 10. Кросс-валидация трёх моделей


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


results = []

print("\nРезультаты базовых моделей:")

for name, pipeline in pipelines.items():

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean()
    }

    results.append(result)

    print(f"\n{name}:")
    print(f"Accuracy:  {result['Accuracy']:.4f}")       # Показывает общую долю правильных ответов модели
    print(f"Precision: {result['Precision']:.4f}")      # Насколько точны положительные предсказания
    print(f"Recall:    {result['Recall']:.4f}")         # Насколько хорошо модель обнаруживает клиентов, которые действительно ушли
    print(f"F1:        {result['F1']:.4f}")             # среднее Precision и Recall


results_df = pd.DataFrame(results)

print("\nСравнение моделей:")
print(results_df)



# 11. Выбор лучшей модели


best_model_name = results_df.loc[
    results_df["F1"].idxmax(),
    "Model"
]

best_model = pipelines[best_model_name]

print(
    f"\nЛучшая модель по F1-score: "
    f"{best_model_name}"
)



# 12. Настройка лучшей модели

param_grids = {

    "KNN": {
        "model__n_neighbors": [
            3, 5, 7, 9, 11, 15, 21
        ],
        "model__weights": [
            "uniform",
            "distance"
        ],
        "model__metric": [
            "euclidean",
            "manhattan"
        ]
    },

    "Decision Tree": {
        "model__max_depth": [
            None, 5, 10, 20
        ],
        "model__min_samples_split": [
            2, 5, 10
        ],
        "model__min_samples_leaf": [
            1, 2, 4
        ]
    },

    "Random Forest": {
        "model__n_estimators": [
            100, 200, 300
        ],
        "model__max_depth": [
            None, 5, 10, 20
        ],
        "model__min_samples_split": [
            2, 5, 10
        ],
        "model__min_samples_leaf": [
            1, 2, 4
        ],
        "model__max_features": [
            "sqrt",
            "log2"
        ]
    }
}


grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grids[best_model_name],
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

print("\nЗапуск Grid Search...")

grid_search.fit(
    X_train,
    y_train
)

print("\nЛучшие параметры:")
[print(f'{key}: {value}') for key, value in grid_search.best_params_.items()]

print(
    f"\nЛучший F1-score на CV: "
    f"{grid_search.best_score_:.4f}"
)


# 13. Финальная модель


final_model = grid_search.best_estimator_

final_model.fit(
    X_train,
    y_train
)

y_pred = final_model.predict(X_test)


# 14. Оценка на тестовой выборке

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\nФинальная оценка на тестовой выборке:")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")



# 15. Матрица ошибок

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nМатрица ошибок:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No", "Yes"]
)

disp.plot()

plt.title(
    f"Матрица ошибок — {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()