import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("mushroom_bases.csv")
# Отображение данных
# Отображение данных
st.title("Анализ данных с помощью моделей машинного обучения")
st.write("### Данные: Грибы Грибочки")
st.write(data.head())

# Выбор модели
model_name = st.sidebar.selectbox("Выберите модель", ("KNN", "SVC", "Дерево решений", "Случайный лес", "Градиентный бустинг"))

# Гиперпараметры для каждой модели
def get_model_params(model_name):
    params = {}
    if model_name == "KNN":
        params["n_neighbors"] = st.sidebar.slider("Количество соседей", 1, 20, value=5)
    elif model_name == "SVC":
        params["C"] = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, value=1.0)
        params["kernel"] = st.sidebar.selectbox("Ядро", ("linear", "poly", "rbf", "sigmoid"))
    elif model_name == "Дерево решений":
        params["max_depth"] = st.sidebar.slider("Максимальная глубина дерева", 1, 20, value=5)
    elif model_name == "Случайный лес":
        params["n_estimators"] = st.sidebar.slider("Количество деревьев", 10, 100, value=50)
        params["max_depth"] = st.sidebar.slider("Максимальная глубина дерева", 1, 20, value=5)
    elif model_name == "Градиентный бустинг":
        params["n_estimators"] = st.sidebar.slider("Количество деревьев", 10, 100, value=50)
        params["learning_rate"] = st.sidebar.slider("Скорость обучения", 0.01, 1.0, value=0.1)
        params["max_depth"] = st.sidebar.slider("Максимальная глубина дерева", 1, 20, value=5)
    return params

params = get_model_params(model_name)

# Разделение данных
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели
def create_model(model_name, params):
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    elif model_name == "SVC":
        model = SVC(C=params["C"], kernel=params["kernel"], probability=True)
    elif model_name == "Дерево решений":
        model = DecisionTreeClassifier(max_depth=params["max_depth"])
    elif model_name == "Случайный лес":
        model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
    elif model_name == "Градиентный бустинг":
        model = GradientBoostingClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], max_depth=params["max_depth"], random_state=42)
    return model

model = create_model(model_name, params)

# Обучение модели
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Отображение метрик
st.write(f"### Метрики для модели {model_name}")
st.write(f"Точность: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"Точность (Precision): {precision_score(y_test, y_pred, average='binary'):.2f}")
st.write(f"Полнота (Recall): {recall_score(y_test, y_pred, average='binary'):.2f}")
st.write(f"F1-Score: {f1_score(y_test, y_pred, average='binary'):.2f}")
st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# Отображение отчета классификации
st.write("### Отчет классификации")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.write(report_df)

# Отображение матрицы ошибок
st.write("### Матрица ошибок")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Отображение ROC-кривой
st.write("### ROC-кривая")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
st.pyplot(fig)