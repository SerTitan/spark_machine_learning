"""
DNN Performance Predictor для предсказания времени выполнения Spark.

Архитектура (по статье Huang et al.):
    Input(n_features) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1)

Особенности:
- Обучение на log1p(duration) для стабилизации
- Early stopping по валидационной метрике
- Learning rate scheduling
- Поддержка PyTorch и sklearn fallback
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

# PyTorch импорт - если не установлен, используем sklearn как fallback
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class DNNConfig:
    """Конфигурация DNN - все гиперпараметры в одном месте."""
    hidden_sizes: Tuple[int, ...] = (128, 64)  # Размеры скрытых слоёв
    dropout: float = 0.1                        # Dropout для регуляризации
    learning_rate: float = 0.001                # Learning rate для Adam
    batch_size: int = 32                        # Размер батча
    max_epochs: int = 500                       # Максимум эпох
    patience: int = 20                          # Early stopping - сколько эпох ждать улучшения
    weight_decay: float = 1e-5                  # L2 регуляризация
    use_log_target: bool = True                 # Логарифм таргета (стабилизирует обучение)
    random_state: int = 42


# === PyTorch нейросеть ===
# Определяем класс только если PyTorch доступен
if TORCH_AVAILABLE:
    class TorchDNN(nn.Module):
        """
        PyTorch DNN для предсказания времени выполнения.
        Архитектура: вход → 128 нейронов → ReLU → Dropout → 64 нейрона → ReLU → Dropout → 1 выход
        """

        def __init__(
            self,
            input_dim: int,                           # Кол-во входных фич
            hidden_sizes: Tuple[int, ...] = (128, 64),  # Размеры скрытых слоёв
            dropout: float = 0.1,                     # Dropout (выключает часть нейронов при обучении)
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_sizes = hidden_sizes

            # Строим сеть слой за слоем
            layers = []
            prev_size = input_dim

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))  # Полносвязный слой
                layers.append(nn.ReLU())                          # Функция активации
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))            # Dropout для регуляризации
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))  # Выходной слой (1 число = время)
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Прямой проход через сеть."""
            return self.net(x)
else:
    TorchDNN = None  # Placeholder когда PyTorch недоступен


@dataclass
class TrainingHistory:
    """История обучения - для построения графиков."""
    train_loss: List[float] = field(default_factory=list)   # Loss на train по эпохам
    val_loss: List[float] = field(default_factory=list)     # Loss на val по эпохам
    val_rmse: List[float] = field(default_factory=list)     # RMSE на val по эпохам
    best_epoch: int = 0                                      # Эпоха с лучшим результатом
    best_val_rmse: float = float("inf")                      # Лучший RMSE


# === Главный класс предиктора ===
class DNNPredictor:
    """
    DNN Predictor для предсказания времени выполнения Spark.

    Если установлен PyTorch - использует его (быстрее).
    Если нет - fallback на sklearn MLPRegressor.
    """

    def __init__(self, config: Optional[DNNConfig] = None):
        self.config = config or DNNConfig()
        self.model = None
        self.use_torch = TORCH_AVAILABLE  # Используем PyTorch если доступен
        self.history = TrainingHistory()
        self._fitted = False
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> "DNNPredictor":
        """
        Обучает DNN на тренировочных данных.

        Args:
            X_train, y_train: тренировочные данные
            X_val, y_val: валидационные данные (для early stopping)
            verbose: выводить прогресс обучения
        """
        # log1p трансформация: log(1+y) - стабилизирует обучение при больших значениях
        if self.config.use_log_target:
            y_train_t = np.log1p(y_train)
            y_val_t = np.log1p(y_val)
        else:
            y_train_t = y_train
            y_val_t = y_val

        # Выбираем backend: PyTorch или sklearn
        if self.use_torch:
            self._fit_torch(X_train, y_train_t, X_val, y_val, y_val_t, verbose)
        else:
            self._fit_sklearn(X_train, y_train_t, X_val, y_val, y_val_t, verbose)

        self._fitted = True
        return self

    def _fit_torch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val_orig: np.ndarray,
        y_val_t: np.ndarray,
        verbose: bool,
    ):
        """Обучение с PyTorch - основной метод."""
        torch.manual_seed(self.config.random_state)

        # Создаём сеть
        input_dim = X_train.shape[1]
        self.model = TorchDNN(
            input_dim=input_dim,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
        )
        self.model.to(self.device)

        # Adam оптимизатор с L2 регуляризацией
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # Уменьшаем lr если нет прогресса
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        loss_fn = nn.MSELoss()  # Mean Squared Error

        # DataLoader - выдаёт батчи данных
        train_ds = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)).view(-1, 1),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)

        best_state = None
        best_val_rmse = float("inf")
        patience_counter = 0

        # Цикл обучения по эпохам
        for epoch in range(1, self.config.max_epochs + 1):
            # === Фаза обучения ===
            self.model.train()  # Включаем dropout
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()        # Обнуляем градиенты
                pred = self.model(xb)        # Прямой проход
                loss = loss_fn(pred, yb)     # Считаем loss
                loss.backward()              # Обратное распространение
                optimizer.step()             # Обновляем веса
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)

            # === Фаза валидации ===
            self.model.eval()  # Выключаем dropout
            with torch.no_grad():
                y_val_pred_log = self.model(X_val_t).cpu().numpy().reshape(-1)

            # Обратная трансформация: expm1 = exp(x) - 1
            if self.config.use_log_target:
                y_val_pred = np.expm1(y_val_pred_log)
                y_val_pred = np.clip(y_val_pred, 0, None)  # Время не может быть < 0
            else:
                y_val_pred = y_val_pred_log

            val_rmse = float(np.sqrt(np.mean((y_val_orig - y_val_pred) ** 2)))
            val_loss = float(np.mean((y_val_t - y_val_pred_log) ** 2))

            # Записываем историю
            self.history.train_loss.append(epoch_loss)
            self.history.val_loss.append(val_loss)
            self.history.val_rmse.append(val_rmse)

            scheduler.step(val_rmse)

            # === Early stopping ===
            # Если улучшение - сохраняем модель, иначе увеличиваем счётчик
            if val_rmse < best_val_rmse - 1e-6:
                best_val_rmse = val_rmse
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.history.best_epoch = epoch
                self.history.best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(f"[Epoch {epoch:03d}] train_loss={epoch_loss:.4f}, val_RMSE={val_rmse:.4f}")

            # Останавливаемся если patience эпох без улучшения
            if patience_counter >= self.config.patience:
                if verbose:
                    print(f"[Early Stop] epoch {epoch}, best val_RMSE={best_val_rmse:.4f}")
                break

        # Загружаем лучшие веса
        if best_state:
            self.model.load_state_dict(best_state)

    def _fit_sklearn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val_orig: np.ndarray,
        y_val_t: np.ndarray,
        verbose: bool,
    ):
        """Fallback на sklearn MLPRegressor."""
        from sklearn.neural_network import MLPRegressor

        if verbose:
            print("[WARN] PyTorch not available, using sklearn MLPRegressor")

        self.model = MLPRegressor(
            hidden_layer_sizes=self.config.hidden_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=self.config.learning_rate,
            batch_size=self.config.batch_size,
            max_iter=self.config.max_epochs,
            early_stopping=True,
            n_iter_no_change=self.config.patience,
            random_state=self.config.random_state,
            verbose=verbose,
        )
        self.model.fit(X_train, y_train.ravel())

        # История
        if hasattr(self.model, "loss_curve_"):
            self.history.train_loss = list(self.model.loss_curve_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает время выполнения для заданных конфигураций.

        Args:
            X: фичи (n_samples, n_features) - препроцессированные данные

        Returns:
            Предсказанное время в секундах
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.use_torch:
            self.model.eval()  # Выключаем dropout
            with torch.no_grad():  # Не считаем градиенты
                X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
                y_log = self.model(X_t).cpu().numpy().reshape(-1)
        else:
            y_log = self.model.predict(X).reshape(-1)

        # Обратная трансформация из log-пространства
        if self.config.use_log_target:
            y_sec = np.expm1(y_log)  # exp(y) - 1
            y_sec = np.clip(y_sec, 0, None)  # Время >= 0
        else:
            y_sec = y_log

        return y_sec

    def save(self, path: str | Path):
        """Сохраняет модель на диск."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Сохраняем конфигурацию
        config_dict = {
            "hidden_sizes": self.config.hidden_sizes,
            "dropout": self.config.dropout,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "use_log_target": self.config.use_log_target,
            "use_torch": self.use_torch,
        }
        joblib.dump(config_dict, path / "config.joblib")

        # Сохраняем веса модели
        if self.use_torch:
            torch.save(self.model.state_dict(), path / "model.pt")
            # Сохраняем архитектуру для восстановления
            arch = {"input_dim": self.model.input_dim, "hidden_sizes": self.model.hidden_sizes}
            joblib.dump(arch, path / "architecture.joblib")
        else:
            joblib.dump(self.model, path / "model_sklearn.joblib")

        # Сохраняем историю обучения
        history_dict = {
            "train_loss": self.history.train_loss,
            "val_loss": self.history.val_loss,
            "val_rmse": self.history.val_rmse,
            "best_epoch": self.history.best_epoch,
            "best_val_rmse": self.history.best_val_rmse,
        }
        joblib.dump(history_dict, path / "history.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "DNNPredictor":
        """Загружает модель."""
        path = Path(path)

        config_dict = joblib.load(path / "config.joblib")
        config = DNNConfig(
            hidden_sizes=tuple(config_dict["hidden_sizes"]),
            dropout=config_dict.get("dropout", 0.1),
            learning_rate=config_dict["learning_rate"],
            batch_size=config_dict["batch_size"],
            use_log_target=config_dict["use_log_target"],
        )

        predictor = cls(config)
        predictor.use_torch = config_dict.get("use_torch", TORCH_AVAILABLE)

        if predictor.use_torch and (path / "model.pt").exists():
            arch = joblib.load(path / "architecture.joblib")
            predictor.model = TorchDNN(
                input_dim=arch["input_dim"],
                hidden_sizes=tuple(arch["hidden_sizes"]),
                dropout=config.dropout,
            )
            predictor.model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
            predictor.model.eval()
        else:
            predictor.model = joblib.load(path / "model_sklearn.joblib")
            predictor.use_torch = False

        if (path / "history.joblib").exists():
            history_dict = joblib.load(path / "history.joblib")
            predictor.history = TrainingHistory(**history_dict)

        predictor._fitted = True
        return predictor

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Вычисляет метрики."""
        from .baseline import compute_metrics
        return compute_metrics(y_true, y_pred)


def create_dnn_predictor(
    hidden_sizes: Tuple[int, ...] = (128, 64),
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience: int = 20,
    use_log_target: bool = True,
    random_state: int = 42,
) -> DNNPredictor:
    """
    Factory function для создания DNN Predictor.

    Args:
        hidden_sizes: размеры скрытых слоёв (по умолчанию 128→64)
        learning_rate: learning rate для Adam
        batch_size: размер batch
        max_epochs: максимум эпох
        patience: early stopping patience
        use_log_target: использовать log1p трансформацию таргета
        random_state: seed

    Returns:
        DNNPredictor instance
    """
    config = DNNConfig(
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        use_log_target=use_log_target,
        random_state=random_state,
    )
    return DNNPredictor(config)
