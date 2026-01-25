"""
Baseline модели для предсказания времени выполнения Spark.

Модели:
- DummyRegressor: предсказание медианы (baseline)
- RandomForestRegressor + RandomizedSearchCV
- RandomForestRegressor + Simulated Annealing
- MLPRegressor (sklearn)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor


# === Результат обучения модели ===
@dataclass
class ModelResult:
    """Результат обучения модели - хранит модель, метрики и параметры."""

    name: str                                        # Название модели
    model: Any                                       # Сама модель (sklearn объект)
    metrics: Dict[str, float]                        # MAE, RMSE, R2, MAPE
    best_params: Optional[Dict[str, Any]] = None     # Лучшие гиперпараметры
    history: Optional[List[Tuple[int, float]]] = None  # История обучения

    def __repr__(self) -> str:
        return f"ModelResult({self.name}, MAE={self.metrics.get('MAE', 'N/A'):.4f})"


# === Метрики качества ===

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error - корень из среднеквадратичной ошибки."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error - средняя абсолютная процентная ошибка."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0  # избегаем деления на ноль
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Вычисляет все метрики для оценки модели."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),   # Средняя абс. ошибка (в секундах)
        "RMSE": rmse(y_true, y_pred),                        # Корень из MSE
        "R2": float(r2_score(y_true, y_pred)),               # Коэф. детерминации (1 = идеально)
        "MAPE": mape(y_true, y_pred),                        # Ошибка в процентах
    }


# === Модель 1: Dummy (самый простой baseline) ===
class DummyBaseline:
    """
    Простейшая модель - предсказывает медиану времени выполнения.
    Нужна для сравнения - если другие модели не лучше, значит они бесполезны.
    """

    def __init__(self, strategy: str = "median"):
        self.strategy = strategy  # "median" или "mean"
        self.model = DummyRegressor(strategy=strategy)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyBaseline":
        # Просто запоминает медиану y
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Возвращает одно и то же значение для всех входов
        return self.model.predict(X)

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelResult:
        """Обучает и оценивает модель на тестовых данных."""
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(name=f"Dummy_{self.strategy}", model=self.model, metrics=metrics)


# === Модель 2: Random Forest + RandomizedSearchCV ===
class RandomForestBaseline:
    """
    Random Forest с автоматическим подбором гиперпараметров.
    RandomizedSearchCV пробует случайные комбинации параметров и выбирает лучшую.
    """

    def __init__(
        self,
        n_iter: int = 40,      # Сколько комбинаций попробовать
        cv: int = 3,           # Кол-во фолдов кросс-валидации
        random_state: int = 42,
        n_jobs: int = -1,      # -1 = использовать все ядра
    ):
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.best_params_ = None

    def _get_param_space(self) -> Dict[str, Any]:
        """Пространство поиска гиперпараметров RF."""
        return {
            "n_estimators": np.arange(80, 401, 20),    # Кол-во деревьев
            "max_depth": list(range(3, 22)) + [None],  # Глубина дерева
            "min_samples_split": np.arange(2, 12),     # Мин. сэмплов для разбиения
            "min_samples_leaf": np.arange(1, 8),       # Мин. сэмплов в листе
            "max_features": ["sqrt", "log2", None],    # Кол-во фич для разбиения
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        """Обучает RF с автоподбором гиперпараметров."""
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        # RandomizedSearchCV пробует n_iter случайных комбинаций параметров
        search = RandomizedSearchCV(
            rf,
            self._get_param_space(),
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            scoring="neg_root_mean_squared_error",  # Минимизируем RMSE
        )
        search.fit(X, y)
        self.model = search.best_estimator_     # Лучшая модель
        self.best_params_ = search.best_params_ # Лучшие параметры
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelResult:
        """Обучает и оценивает модель на тестовых данных."""
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(
            name="RandomForest_RandomSearch",
            model=self.model,
            metrics=metrics,
            best_params=self.best_params_,
        )


# === Модель 3: Random Forest + Simulated Annealing ===
class SimulatedAnnealingRF:
    """
    Random Forest с оптимизацией гиперпараметров через Simulated Annealing.

    SA (имитация отжига) - метод оптимизации, который может выходить из
    локальных минимумов. В начале "температура" высокая и алгоритм может
    принимать худшие решения, к концу - только улучшения.
    """

    def __init__(
        self,
        n_iter: int = 120,        # Кол-во итераций (больше — лучше поиск, дороже по времени)
        T0: float = 3.0,         # Начальная температура
        alpha: float = 0.93,     # Коэф. охлаждения (T *= alpha каждую итерацию)
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_iter = n_iter
        self.T0 = T0
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.best_params_ = None
        self.history_ = []       # История RMSE по итерациям
        self._rng = np.random.RandomState(random_state)

    def _random_params(self) -> Dict[str, Any]:
        """Генерирует случайный набор гиперпараметров RF."""
        return {
            "n_estimators": int(self._rng.randint(80, 400)),
            "max_depth": int(self._rng.randint(3, 22)) if self._rng.random() > 0.1 else None,
            "min_samples_split": int(self._rng.randint(2, 12)),
            "min_samples_leaf": int(self._rng.randint(1, 8)),
            "max_features": self._rng.choice(["sqrt", "log2", None]),
        }

    def _score(
        self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> Tuple[float, RandomForestRegressor]:
        """Обучает RF с данными параметрами и возвращает RMSE на валидации."""
        model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return rmse(y_val, y_pred), model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "SimulatedAnnealingRF":
        """
        Оптимизирует гиперпараметры RF через Simulated Annealing.
        """
        # Начинаем со случайных параметров
        cur_params = self._random_params()
        cur_score, cur_model = self._score(cur_params, X_train, y_train, X_val, y_val)

        best_params = cur_params.copy()
        best_score = cur_score
        best_model = cur_model

        T = self.T0  # Начальная температура
        self.history_ = [(0, cur_score)]

        for t in range(1, self.n_iter + 1):
            # Генерируем случайного кандидата
            cand_params = self._random_params()
            cand_score, cand_model = self._score(cand_params, X_train, y_train, X_val, y_val)

            # Критерий Метрополиса: принимаем если лучше ИЛИ с вероятностью exp(-delta/T)
            delta = cand_score - cur_score
            if delta < 0 or self._rng.random() < np.exp(-delta / max(T, 1e-8)):
                cur_params = cand_params
                cur_score = cand_score
                cur_model = cand_model

            # Запоминаем лучшее найденное решение
            if cur_score < best_score:
                best_params = cur_params.copy()
                best_score = cur_score
                best_model = cur_model

            self.history_.append((t, cur_score))
            T *= self.alpha  # Охлаждаем температуру

        self.model = best_model
        self.best_params_ = best_params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelResult:
        """Обучает и оценивает модель на тестовых данных."""
        self.fit(X_train, y_train, X_val, y_val)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(
            name="RandomForest_SimulatedAnnealing",
            model=self.model,
            metrics=metrics,
            best_params=self.best_params_,
            history=self.history_,
        )


# === Модель 4: MLP (нейросеть) ===
class MLPBaseline:
    """
    MLP (Multi-Layer Perceptron) - простая нейросеть.
    Архитектура по умолчанию: вход → 128 нейронов → 64 нейрона → 1 выход
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),    # Более компактная сеть для малого датасета
        activation: str = "relu",                          # Функция активации
        learning_rate_init: float = 0.003,                 # Ускоренный learning rate
        max_iter: int = 1200,                              # Больше эпох
        early_stopping: bool = True,                       # Остановка если нет улучшений
        n_iter_no_change: int = 40,                        # Ждём дольше перед остановкой
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        """Обучает нейросеть."""
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",  # Оптимизатор Adam
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def loss_curve(self) -> List[float]:
        """История loss по эпохам (для графиков)."""
        return getattr(self.model, "loss_curve_", [])

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelResult:
        """Обучает и оценивает модель на тестовых данных."""
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(
            name="MLP",
            model=self.model,
            metrics=metrics,
            history=[(i, v) for i, v in enumerate(self.loss_curve)],
        )


# === Функция для обучения всех baseline моделей ===

def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    include_dummy: bool = True,
    rf_search_iters: int = 40,
    sa_iters: int = 120,
    mlp_hidden: Tuple[int, ...] = (64, 32),
    mlp_lr: float = 0.003,
    mlp_max_iter: int = 1200,
    mlp_patience: int = 40,
) -> List[ModelResult]:
    """
    Обучает все модели из статьи и возвращает результаты.
    Модели: LR, SVR, DTR, ETR, RFR, DNN (MLP)
    """
    results = []
    total = 7 if not include_dummy else 8
    idx = 1

    # 1. Dummy - предсказывает медиану (baseline для сравнения)
    if include_dummy:
        print(f"[{idx}/{total}] Training Dummy (mean)...")
        dummy = DummyBaseline(strategy="mean")
        results.append(dummy.evaluate(X_train, y_train, X_test, y_test))
        print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")
        idx += 1

    # 2. Linear Regression (LR)
    print(f"[{idx}/{total}] Training Linear Regression...")
    lr = LinearRegressionBaseline()
    results.append(lr.evaluate(X_train, y_train, X_test, y_test))
    print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")
    idx += 1

    # 3. Support Vector Regression (SVR)
    print(f"[{idx}/{total}] Training SVR...")
    svr = SVRBaseline(kernel="rbf", C=10.0, epsilon=0.1)
    results.append(svr.evaluate(X_train, y_train, X_test, y_test))
    print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")
    idx += 1

    # 4. Decision Tree Regression (DTR)
    print(f"[{idx}/{total}] Training Decision Tree...")
    dtr = DecisionTreeBaseline(max_depth=10, random_state=random_state)
    results.append(dtr.evaluate(X_train, y_train, X_test, y_test))
    print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")
    idx += 1

    # 5. Extra Trees Regression (ETR)
    print(f"[{idx}/{total}] Training Extra Trees...")
    etr = ExtraTreesBaseline(n_estimators=100, random_state=random_state)
    results.append(etr.evaluate(X_train, y_train, X_test, y_test))
    print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")
    idx += 1

    # 6. Random Forest Regression (RFR) + RandomizedSearchCV
    print(f"[{idx}/{total}] Training RandomForest + RandomizedSearchCV (n_iter={rf_search_iters})...")
    rf_rs = RandomForestBaseline(n_iter=rf_search_iters, random_state=random_state)
    results.append(rf_rs.evaluate(X_train, y_train, X_test, y_test))
    print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")
    idx += 1

    # 7. Random Forest + Simulated Annealing
    print(f"[{idx}/{total}] Training RandomForest + Simulated Annealing (n_iter={sa_iters})...")
    rf_sa = SimulatedAnnealingRF(n_iter=sa_iters, random_state=random_state)
    results.append(rf_sa.evaluate(X_train, y_train, X_val, y_val, X_test, y_test))
    print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")
    idx += 1

    # 8. MLP нейросеть (DNN) - 128 → 64 → 1
    print(f"[{idx}/{total}] Training MLP/DNN ({'→'.join(map(str, mlp_hidden))})...")
    mlp = MLPBaseline(
        hidden_layer_sizes=mlp_hidden,
        learning_rate_init=mlp_lr,
        max_iter=mlp_max_iter,
        n_iter_no_change=mlp_patience,
        random_state=random_state,
    )
    results.append(mlp.evaluate(X_train, y_train, X_test, y_test))
    print(f"      → MAE={results[-1].metrics['MAE']:.4f}, R²={results[-1].metrics['R2']:.4f}")

    return results


# === Модель 5: Linear Regression ===
class LinearRegressionBaseline:
    """Линейная регрессия - простейшая параметрическая модель."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(name="LinearRegression", model=self.model, metrics=metrics)


# === Модель 6: Support Vector Regression ===
class SVRBaseline:
    """SVR - Support Vector Regression."""

    def __init__(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRBaseline":
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(name="SVR", model=self.model, metrics=metrics)


# === Модель 7: Decision Tree Regression ===
class DecisionTreeBaseline:
    """Decision Tree Regressor."""

    def __init__(self, max_depth: int = None, random_state: int = 42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeBaseline":
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(name="DecisionTree", model=self.model, metrics=metrics)


# === Модель 8: Extra Trees Regression ===
class ExtraTreesBaseline:
    """Extra Trees Regressor - ансамбль случайных деревьев."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesBaseline":
        self.model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        return ModelResult(name="ExtraTrees", model=self.model, metrics=metrics)


def results_to_dataframe(results: List[ModelResult]) -> "pd.DataFrame":
    """Конвертирует результаты в таблицу для удобного просмотра."""
    import pandas as pd

    rows = []
    for r in results:
        row = {"model": r.name, **r.metrics}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
