"""
Reinforcement Learning Optimizer для поиска оптимальных параметров Spark.

Методы:
- TabularQLearning: классический Q-Learning с дискретным пространством состояний
- DQNOptimizer: Deep Q-Network для больших/непрерывных пространств
- StableBaselinesOptimizer: PPO, A2C через stable-baselines3
- BayesianOptimizer: Bayesian Optimization через Optuna

Использование:
    # Q-Learning
    optimizer = TabularQLearning(param_grid, predictor)
    best_config, history = optimizer.optimize(topology, profile, n_episodes=100)

    # DQN
    optimizer = DQNOptimizer(param_grid, predictor)
    best_config, history = optimizer.optimize(topology, profile, n_episodes=1000)

    # Stable-Baselines3 (PPO/A2C)
    optimizer = StableBaselinesOptimizer(param_grid, predictor, algorithm="PPO")
    best_config, history = optimizer.optimize(topology, profile, total_timesteps=10000)

    # Bayesian Optimization
    optimizer = BayesianOptimizer(param_grid, predictor)
    best_config, history = optimizer.optimize(topology, profile, n_trials=100)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# === Результат оптимизации ===
@dataclass
class OptimizationResult:
    """Результат оптимизации - лучшая конфигурация и история поиска."""
    best_config: Dict[str, Any]       # Лучшие найденные параметры Spark
    best_predicted_time: float        # Предсказанное время для лучшей конфигурации
    history: pd.DataFrame             # История всех испробованных конфигураций
    algorithm: str                    # Название алгоритма
    n_iterations: int                 # Сколько итераций потребовалось

    def speedup_vs_default(self, default_time: float) -> float:
        """Вычисляет ускорение относительно дефолтной конфигурации."""
        if self.best_predicted_time <= 0:
            return 0.0
        return default_time / self.best_predicted_time  # Например, 2.0 = в 2 раза быстрее

    def improvement_pct(self, baseline_time: float) -> float:
        """Вычисляет улучшение в процентах."""
        if baseline_time <= 0:
            return 0.0
        return (baseline_time - self.best_predicted_time) / baseline_time * 100.0


# === Базовый класс оптимизатора ===
class BaseOptimizer(ABC):
    """
    Базовый класс для всех оптимизаторов.
    Задача оптимизатора: найти такие параметры Spark, которые минимизируют время.
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        predictor: Callable[[pd.DataFrame], np.ndarray],
        random_state: int = 42,
    ):
        """
        Args:
            param_grid: пространство параметров {param_name: [возможные значения]}
            predictor: обученная модель (принимает DataFrame, возвращает время)
            random_state: seed для воспроизводимости
        """
        self.param_grid = param_grid
        self.predictor = predictor
        self.random_state = random_state
        self.param_names = list(param_grid.keys())

    @abstractmethod
    def optimize(
        self,
        topology: Dict[str, Any],
        profile: str,
        **kwargs,
    ) -> OptimizationResult:
        """Запускает оптимизацию - реализуется в подклассах."""
        pass

    def _create_config_df(
        self,
        config: Dict[str, Any],
        topology: Dict[str, Any],
        profile: str,
    ) -> pd.DataFrame:
        """Собирает конфигурацию в DataFrame для предсказания."""
        row = {**topology, "profile": profile, **config}
        return pd.DataFrame([row])

    def _predict_time(
        self,
        config: Dict[str, Any],
        topology: Dict[str, Any],
        profile: str,
    ) -> float:
        """Предсказывает время выполнения для данной конфигурации."""
        snapped = self._snap_to_grid(config)
        df = self._create_config_df(snapped, topology, profile)
        return float(self.predictor(df)[0])

    def _snap_to_grid(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Приводит значения к ближайшим из param_grid (избегаем дробных/недопустимых)."""
        snapped = {}
        for name, values in self.param_grid.items():
            if name not in config:
                snapped[name] = values[0]
                continue
            val = config[name]
            # если в сетке есть точное значение — берём его
            if val in values:
                snapped[name] = val
                continue
            # для чисел — ближайшее по модулю (используем default param для замыкания)
            if isinstance(values[0], (int, float)):
                snapped[name] = min(values, key=lambda v, target=val: abs(v - float(target)))
            else:
                # категориальные/булевы — первая в списке
                snapped[name] = values[0]
        return snapped


# === Алгоритм 1: Табличный Q-Learning ===
class TabularQLearning(BaseOptimizer):
    """
    Классический Q-Learning для дискретного пространства параметров.

    Идея: агент находится в "состоянии" (текущая конфигурация) и может
    выполнять "действия" (изменить один параметр на +1 или -1).
    За каждое действие получает "награду" (улучшение времени).

    Q-таблица хранит оценки полезности каждого действия в каждом состоянии.
    Формула обновления: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        predictor: Callable[[pd.DataFrame], np.ndarray],
        alpha: float = 0.1,           # Скорость обучения
        gamma: float = 0.95,          # Дисконт будущих наград
        epsilon: float = 0.3,         # Вероятность случайного действия
        epsilon_decay: float = 0.995, # Уменьшение epsilon после каждого эпизода
        epsilon_min: float = 0.05,    # Минимальный epsilon
        random_state: int = 42,
        initial_config: Optional[Dict[str, Any]] = None,  # Начальная конфигурация
    ):
        super().__init__(param_grid, predictor, random_state)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_config = initial_config
        # Q-таблица: {состояние -> {действие -> Q-значение}}
        self.Q: Dict[Tuple[int, ...], Dict[Tuple[str, int], float]] = {}
        self._rng = random.Random(random_state)

    def _state_to_indices(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Конвертирует конфигурацию в tuple индексов (состояние)."""
        indices = []
        for name in self.param_names:
            values = self.param_grid[name]
            val = config[name]
            try:
                idx = values.index(val)
            except ValueError:
                # Если точного значения нет - берём ближайшее (используем default params для замыкания)
                idx = min(range(len(values)), key=lambda i, v=val, vs=values: abs(vs[i] - v) if isinstance(v, (int, float)) else 0)
            indices.append(idx)
        return tuple(indices)

    def _indices_to_config(self, indices: Tuple[int, ...]) -> Dict[str, Any]:
        """Конвертирует индексы обратно в конфигурацию."""
        return {
            name: self.param_grid[name][idx]
            for name, idx in zip(self.param_names, indices)
        }

    def _get_actions(self, state: Tuple[int, ...]) -> List[Tuple[str, int]]:
        """Возвращает возможные действия: (имя_параметра, направление +1/-1)."""
        actions = []
        for i, name in enumerate(self.param_names):
            n = len(self.param_grid[name])
            if n <= 1:
                continue  # Параметр с одним значением нельзя менять
            if state[i] > 0:
                actions.append((name, -1))  # Уменьшить параметр
            if state[i] < n - 1:
                actions.append((name, +1))  # Увеличить параметр
        return actions

    def _apply_action(self, state: Tuple[int, ...], action: Tuple[str, int]) -> Tuple[int, ...]:
        """Применяет действие и возвращает новое состояние."""
        name, delta = action
        i = self.param_names.index(name)
        n = len(self.param_grid[name])
        new_state = list(state)
        new_state[i] = max(0, min(n - 1, state[i] + delta))  # Не выходим за границы
        return tuple(new_state)

    def _get_q_row(self, state: Tuple[int, ...]) -> Dict[Tuple[str, int], float]:
        """Возвращает Q-значения всех действий для данного состояния."""
        if state not in self.Q:
            # Инициализируем нулями
            self.Q[state] = {a: 0.0 for a in self._get_actions(state)}
        return self.Q[state]

    def _select_action(self, state: Tuple[int, ...], epsilon: float) -> Optional[Tuple[str, int]]:
        """ε-greedy выбор действия: с вероятностью ε - случайное, иначе лучшее."""
        q_row = self._get_q_row(state)
        if not q_row:
            return None
        if self._rng.random() < epsilon:
            return self._rng.choice(list(q_row.keys()))  # Случайное действие
        return max(q_row.items(), key=lambda x: x[1])[0]  # Лучшее по Q

    def _config_to_indices(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Конвертирует конфигурацию в индексы (ближайшие значения в grid)."""
        indices = []
        for name in self.param_names:
            values = self.param_grid[name]
            val = config.get(name, values[len(values) // 2])
            # Ищем ближайшее значение в grid
            if val in values:
                idx = values.index(val)
            elif isinstance(val, (int, float)):
                # Используем default params чтобы избежать проблем с замыканием
                idx = min(range(len(values)), key=lambda i, v=val, vs=values: abs(vs[i] - v) if isinstance(vs[i], (int, float)) else float('inf'))
            else:
                idx = len(values) // 2
            indices.append(idx)
        return tuple(indices)

    def _initial_state(self) -> Tuple[int, ...]:
        """Начальное состояние - из initial_config или середина каждого диапазона."""
        if self.initial_config:
            return self._config_to_indices(self.initial_config)
        return tuple(len(vals) // 2 for vals in self.param_grid.values())

    def optimize(
        self,
        topology: Dict[str, Any],
        profile: str,
        n_episodes: int = 100,
        max_steps: int = 50,
    ) -> OptimizationResult:
        """
        Запускает Q-Learning оптимизацию.

        Args:
            topology: параметры кластера (workers, cores, memory)
            profile: профиль задачи (small, medium, large)
            n_episodes: количество эпизодов обучения
            max_steps: максимум шагов в одном эпизоде
        """
        history = []
        best_state = self._initial_state()
        best_config = self._indices_to_config(best_state)
        best_time = self._predict_time(best_config, topology, profile)

        epsilon = self.epsilon

        # Цикл по эпизодам
        for episode in range(n_episodes):
            state = best_state  # Начинаем с лучшего найденного
            config = self._indices_to_config(state)
            t_prev = self._predict_time(config, topology, profile)

            # Цикл по шагам в эпизоде
            for step in range(max_steps):
                # Выбираем действие (ε-greedy)
                action = self._select_action(state, epsilon)
                if action is None:
                    break

                # Применяем действие и получаем новое состояние
                next_state = self._apply_action(state, action)
                next_config = self._indices_to_config(next_state)
                t_new = self._predict_time(next_config, topology, profile)

                # Награда = относительное улучшение времени
                reward = (t_prev - t_new) / max(t_prev, 1e-9)

                # Q-learning update: Q(s,a) += α * (r + γ*max_Q(s') - Q(s,a))
                q_row = self._get_q_row(state)
                next_q_row = self._get_q_row(next_state)
                max_next_q = max(next_q_row.values()) if next_q_row else 0.0

                q_row[action] = q_row[action] + self.alpha * (
                    reward + self.gamma * max_next_q - q_row[action]
                )

                # Записываем историю
                history.append({
                    "episode": episode,
                    "step": step,
                    "predicted_time": t_new,
                    "reward": reward,
                    "epsilon": epsilon,
                    **next_config,
                })

                # Обновляем лучший результат
                if t_new < best_time:
                    best_time = t_new
                    best_state = next_state
                    best_config = next_config

                state = next_state
                t_prev = t_new

            # Уменьшаем epsilon (меньше случайных действий со временем)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        return OptimizationResult(
            best_config=best_config,
            best_predicted_time=best_time,
            history=pd.DataFrame(history),
            algorithm="TabularQLearning",
            n_iterations=n_episodes * max_steps,
        )


# === Алгоритм 2: Deep Q-Network (DQN) ===
class DQNOptimizer(BaseOptimizer):
    """
    Deep Q-Network - использует нейросеть вместо таблицы для хранения Q-значений.

    Преимущества перед табличным Q-Learning:
    - Работает с большими пространствами состояний
    - Обобщает на похожие состояния

    Ключевые компоненты:
    - Experience Replay: буфер прошлого опыта для стабильного обучения
    - Target Network: отдельная сеть для вычисления target Q-значений
    - ε-greedy: баланс исследования и эксплуатации
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        predictor: Callable[[pd.DataFrame], np.ndarray],
        hidden_sizes: Tuple[int, ...] = (64, 32),  # Архитектура Q-сети
        learning_rate: float = 0.001,
        gamma: float = 0.95,                       # Дисконт будущих наград
        epsilon: float = 1.0,                      # Начальная вероятность случайного действия
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        batch_size: int = 32,                      # Размер батча для обучения
        buffer_size: int = 10000,                  # Размер replay buffer
        target_update_freq: int = 10,              # Как часто обновлять target сеть
        random_state: int = 42,
        initial_config: Optional[Dict[str, Any]] = None,  # Начальная конфигурация
    ):
        super().__init__(param_grid, predictor, random_state)
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.initial_config = initial_config

        # Размерности: состояние = кол-во параметров, действие = 2 на каждый параметр (+1/-1)
        self.state_dim = len(param_grid)
        self.action_dim = sum(2 for vals in param_grid.values() if len(vals) > 1)

        self._check_torch()  # DQN требует PyTorch

    def _check_torch(self):
        """Проверяет доступность PyTorch."""
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ImportError("DQNOptimizer requires PyTorch. Install with: pip install torch")

    def _build_network(self):
        """Создаёт Q-network."""
        import torch.nn as nn

        layers = []
        prev_dim = self.state_dim
        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, self.action_dim))

        return nn.Sequential(*layers)

    def _state_to_tensor(self, state: Tuple[int, ...]) -> "torch.Tensor":
        """Конвертирует состояние в тензор."""
        # Нормализуем индексы
        normalized = []
        for i, name in enumerate(self.param_names):
            n = len(self.param_grid[name])
            normalized.append(state[i] / max(n - 1, 1))
        return self._torch.tensor(normalized, dtype=self._torch.float32).unsqueeze(0)

    def _action_to_index(self, action: Tuple[str, int]) -> int:
        """Конвертирует действие в индекс."""
        idx = 0
        for name in self.param_names:
            if len(self.param_grid[name]) <= 1:
                continue
            if action[0] == name:
                return idx if action[1] == -1 else idx + 1
            idx += 2
        return 0

    def _index_to_action(self, index: int, state: Tuple[int, ...]) -> Optional[Tuple[str, int]]:
        """Конвертирует индекс в действие."""
        idx = 0
        for i, name in enumerate(self.param_names):
            n = len(self.param_grid[name])
            if n <= 1:
                continue
            if index == idx:
                if state[i] > 0:
                    return (name, -1)
            elif index == idx + 1:
                if state[i] < n - 1:
                    return (name, +1)
            idx += 2
        return None

    def _config_to_indices(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Конвертирует конфигурацию в индексы (ближайшие значения в grid)."""
        indices = []
        for name in self.param_names:
            values = self.param_grid[name]
            val = config.get(name, values[len(values) // 2])
            if val in values:
                idx = values.index(val)
            elif isinstance(val, (int, float)):
                idx = min(range(len(values)), key=lambda i, v=val, vs=values: abs(vs[i] - v) if isinstance(vs[i], (int, float)) else float('inf'))
            else:
                idx = len(values) // 2
            indices.append(idx)
        return tuple(indices)

    def _initial_state(self) -> Tuple[int, ...]:
        """Начальное состояние - из initial_config или середина grid."""
        if self.initial_config:
            return self._config_to_indices(self.initial_config)
        return tuple(len(vals) // 2 for vals in self.param_grid.values())

    def optimize(
        self,
        topology: Dict[str, Any],
        profile: str,
        n_episodes: int = 500,
        max_steps: int = 50,
    ) -> OptimizationResult:
        """Запускает DQN оптимизацию."""
        import torch
        import torch.nn.functional as F
        from collections import deque

        # Networks
        q_net = self._build_network()
        target_net = self._build_network()
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters(), lr=self.learning_rate)

        # Replay buffer
        buffer = deque(maxlen=self.buffer_size)

        history = []
        best_state = self._initial_state()
        best_config = {name: self.param_grid[name][idx] for name, idx in zip(self.param_names, best_state)}
        best_time = self._predict_time(best_config, topology, profile)

        epsilon = self.epsilon
        total_steps = 0

        for episode in range(n_episodes):
            state = best_state
            config = {name: self.param_grid[name][idx] for name, idx in zip(self.param_names, state)}
            t_prev = self._predict_time(config, topology, profile)

            for step in range(max_steps):
                # ε-greedy action selection
                if random.random() < epsilon:
                    valid_actions = []
                    for i, name in enumerate(self.param_names):
                        n = len(self.param_grid[name])
                        if n <= 1:
                            continue
                        if state[i] > 0:
                            valid_actions.append((name, -1))
                        if state[i] < n - 1:
                            valid_actions.append((name, +1))
                    if not valid_actions:
                        break
                    action = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        q_values = q_net(self._state_to_tensor(state))
                    action_idx = q_values.argmax().item()
                    action = self._index_to_action(action_idx, state)
                    if action is None:
                        break

                # Apply action
                name, delta = action
                i = self.param_names.index(name)
                n = len(self.param_grid[name])
                next_state = list(state)
                next_state[i] = max(0, min(n - 1, state[i] + delta))
                next_state = tuple(next_state)

                next_config = {nm: self.param_grid[nm][idx] for nm, idx in zip(self.param_names, next_state)}
                t_new = self._predict_time(next_config, topology, profile)

                reward = (t_prev - t_new) / max(t_prev, 1e-9)

                # Store transition
                buffer.append((state, action, reward, next_state, step == max_steps - 1))

                # Train
                if len(buffer) >= self.batch_size:
                    batch = random.sample(buffer, self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    state_tensors = torch.cat([self._state_to_tensor(s) for s in states])
                    next_state_tensors = torch.cat([self._state_to_tensor(s) for s in next_states])
                    action_indices = torch.tensor([self._action_to_index(a) for a in actions])
                    rewards_t = torch.tensor(rewards, dtype=torch.float32)
                    dones_t = torch.tensor(dones, dtype=torch.float32)

                    q_values = q_net(state_tensors).gather(1, action_indices.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        next_q_values = target_net(next_state_tensors).max(1)[0]
                    targets = rewards_t + self.gamma * next_q_values * (1 - dones_t)

                    loss = F.mse_loss(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Update target network
                total_steps += 1
                if total_steps % self.target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

                # Record
                history.append({
                    "episode": episode,
                    "step": step,
                    "predicted_time": t_new,
                    "reward": reward,
                    "epsilon": epsilon,
                    **next_config,
                })

                # Update best
                if t_new < best_time:
                    best_time = t_new
                    best_state = next_state
                    best_config = next_config

                state = next_state
                t_prev = t_new

            # Decay epsilon
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        return OptimizationResult(
            best_config=best_config,
            best_predicted_time=best_time,
            history=pd.DataFrame(history),
            algorithm="DQN",
            n_iterations=total_steps,
        )


# === Алгоритм 3: Bayesian Optimization (через Optuna) ===
class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization - умный поиск гиперпараметров.

    В отличие от случайного поиска, строит модель зависимости
    "параметры → результат" и предлагает следующие точки для проверки
    на основе этой модели.

    Хорош когда:
    - Мало итераций (каждая оценка дорогая)
    - Параметры непрерывные или их много
    - Нужен глобальный оптимум
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        predictor: Callable[[pd.DataFrame], np.ndarray],
        sampler: str = "tpe",  # "tpe" (Tree-Parzen), "cmaes", "random"
        random_state: int = 42,
    ):
        super().__init__(param_grid, predictor, random_state)
        self.sampler = sampler
        self._check_optuna()

    def _check_optuna(self):
        """Проверяет что Optuna установлена."""
        try:
            import optuna
            self._optuna = optuna
        except ImportError:
            raise ImportError("BayesianOptimizer requires Optuna. Install with: pip install optuna")

    def optimize(
        self,
        topology: Dict[str, Any],
        profile: str,
        n_trials: int = 100,
    ) -> OptimizationResult:
        """Запускает Bayesian Optimization."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        history = []

        def objective(trial):
            config = {}
            for name, values in self.param_grid.items():
                if len(values) <= 1:
                    config[name] = values[0]
                elif isinstance(values[0], bool) or set(values) == {0, 1}:
                    config[name] = trial.suggest_categorical(name, values)
                elif isinstance(values[0], int):
                    config[name] = trial.suggest_int(name, min(values), max(values))
                elif isinstance(values[0], float):
                    # значения дискретны — берём категориально, чтобы не появлялись дробные вне сетки
                    config[name] = trial.suggest_categorical(name, values)
                else:
                    config[name] = trial.suggest_categorical(name, values)

            t_pred = self._predict_time(config, topology, profile)
            history.append({
                "trial": trial.number,
                "predicted_time": t_pred,
                **config,
            })
            return t_pred

        # Create sampler
        if self.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        elif self.sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.random_state)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return OptimizationResult(
            best_config=study.best_params,
            best_predicted_time=study.best_value,
            history=pd.DataFrame(history),
            algorithm=f"Bayesian_{self.sampler}",
            n_iterations=n_trials,
        )


# === Алгоритм 4: Stable-Baselines3 (PPO, A2C) ===
class StableBaselinesOptimizer(BaseOptimizer):
    """
    Оптимизатор на базе stable-baselines3 - библиотеки RL алгоритмов.

    Поддерживаемые алгоритмы:
    - PPO (Proximal Policy Optimization) - самый популярный, стабильный
    - A2C (Advantage Actor-Critic) - быстрее, но менее стабильный
    - DQN (Deep Q-Network) - для дискретных действий

    Создаёт Gym-совместимую среду где агент учится оптимизировать параметры.
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        predictor: Callable[[pd.DataFrame], np.ndarray],
        algorithm: str = "PPO",  # "PPO", "A2C", "DQN"
        random_state: int = 42,
    ):
        super().__init__(param_grid, predictor, random_state)
        self.algorithm = algorithm
        self._check_sb3()

    def _check_sb3(self):
        """Проверяет что stable-baselines3 установлена."""
        try:
            import gymnasium as gym
            import stable_baselines3
            self._gym = gym
            self._sb3 = stable_baselines3
        except ImportError:
            raise ImportError(
                "StableBaselinesOptimizer requires gymnasium and stable-baselines3. "
                "Install with: pip install gymnasium stable-baselines3"
            )

    def _create_env(self, topology: Dict[str, Any], profile: str):
        """Создаёт Gym среду для оптимизации."""
        import gymnasium as gym
        from gymnasium import spaces

        param_grid = self.param_grid
        param_names = self.param_names
        predictor = self.predictor

        class SparkConfigEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.param_grid = param_grid
                self.param_names = param_names
                self.topology = topology
                self.profile = profile
                self.predictor = predictor

                # State: normalized indices
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(len(param_names),), dtype=np.float32
                )

                # Action: discrete (param_idx * 2 + direction)
                n_actions = sum(2 for vals in param_grid.values() if len(vals) > 1)
                self.action_space = spaces.Discrete(max(n_actions, 1))

                self.state = None
                self.best_time = float("inf")
                self.step_count = 0
                self.max_steps = 50

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.state = [len(vals) // 2 for vals in self.param_grid.values()]
                self.step_count = 0
                config = self._state_to_config()
                self.best_time = self._predict(config)
                return self._get_obs(), {}

            def step(self, action):
                self.step_count += 1

                # Decode action
                action_applied = False
                idx = 0
                for i, name in enumerate(self.param_names):
                    n = len(self.param_grid[name])
                    if n <= 1:
                        continue
                    if action == idx:  # decrease
                        if self.state[i] > 0:
                            self.state[i] -= 1
                            action_applied = True
                        break
                    elif action == idx + 1:  # increase
                        if self.state[i] < n - 1:
                            self.state[i] += 1
                            action_applied = True
                        break
                    idx += 2

                config = self._state_to_config()
                t_new = self._predict(config)

                # Reward: normalized improvement
                reward = (self.best_time - t_new) / max(self.best_time, 1e-9)
                if t_new < self.best_time:
                    self.best_time = t_new

                done = self.step_count >= self.max_steps
                truncated = False

                return self._get_obs(), reward, done, truncated, {"time": t_new, "config": config}

            def _get_obs(self):
                obs = []
                for i, name in enumerate(self.param_names):
                    n = len(self.param_grid[name])
                    obs.append(self.state[i] / max(n - 1, 1))
                return np.array(obs, dtype=np.float32)

            def _state_to_config(self):
                return {
                    name: self.param_grid[name][self.state[i]]
                    for i, name in enumerate(self.param_names)
                }

            def _predict(self, config):
                row = {**self.topology, "profile": self.profile, **config}
                df = pd.DataFrame([row])
                return float(self.predictor(df)[0])

        return SparkConfigEnv()

    def optimize(
        self,
        topology: Dict[str, Any],
        profile: str,
        total_timesteps: int = 10000,
    ) -> OptimizationResult:
        """Запускает SB3 оптимизацию."""
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.callbacks import BaseCallback

        env = self._create_env(topology, profile)

        # History callback
        history = []

        class HistoryCallback(BaseCallback):
            def _on_step(self):
                if "time" in self.locals.get("infos", [{}])[0]:
                    info = self.locals["infos"][0]
                    history.append({
                        "timestep": self.num_timesteps,
                        "predicted_time": info["time"],
                        **info["config"],
                    })
                return True

        # Select algorithm
        algos = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        AlgoClass = algos.get(self.algorithm, PPO)

        model = AlgoClass("MlpPolicy", env, verbose=0, seed=self.random_state)
        model.learn(total_timesteps=total_timesteps, callback=HistoryCallback())

        # Get best from history
        if history:
            best_idx = min(range(len(history)), key=lambda i: history[i]["predicted_time"])
            best_config = {k: v for k, v in history[best_idx].items() if k not in ["timestep", "predicted_time"]}
            best_time = history[best_idx]["predicted_time"]
        else:
            best_config = {name: self.param_grid[name][len(vals) // 2] for name, vals in self.param_grid.items()}
            best_time = self._predict_time(best_config, topology, profile)

        return OptimizationResult(
            best_config=best_config,
            best_predicted_time=best_time,
            history=pd.DataFrame(history),
            algorithm=f"SB3_{self.algorithm}",
            n_iterations=total_timesteps,
        )


# === Функция для запуска всех оптимизаторов ===

def run_all_optimizers(
    param_grid: Dict[str, List[Any]],
    predictor: Callable[[pd.DataFrame], np.ndarray],
    topology: Dict[str, Any],
    profile: str,
    random_state: int = 42,
) -> Dict[str, OptimizationResult]:
    """
    Запускает все доступные RL оптимизаторы и сравнивает результаты.
    Пропускает алгоритмы если не установлены нужные библиотеки.

    Returns:
        Словарь {название_алгоритма: OptimizationResult}
    """
    results = {}

    # 1. Q-Learning - всегда доступен (без зависимостей)
    print("[1/4] Running Q-Learning...")
    ql = TabularQLearning(param_grid, predictor, random_state=random_state)
    results["QLearning"] = ql.optimize(topology, profile)
    print(f"      Best time: {results['QLearning'].best_predicted_time:.2f}s")

    # 2. DQN - требует PyTorch
    try:
        print("[2/4] Running DQN...")
        dqn = DQNOptimizer(param_grid, predictor, random_state=random_state)
        results["DQN"] = dqn.optimize(topology, profile)
        print(f"      Best time: {results['DQN'].best_predicted_time:.2f}s")
    except ImportError as e:
        print(f"[2/4] Skipping DQN: {e}")

    # 3. Bayesian Optimization - требует Optuna
    try:
        print("[3/4] Running Bayesian Optimization...")
        bo = BayesianOptimizer(param_grid, predictor, random_state=random_state)
        results["Bayesian"] = bo.optimize(topology, profile)
        print(f"      Best time: {results['Bayesian'].best_predicted_time:.2f}s")
    except ImportError as e:
        print(f"[3/4] Skipping Bayesian: {e}")

    # 4. PPO - требует stable-baselines3
    try:
        print("[4/4] Running PPO...")
        ppo = StableBaselinesOptimizer(param_grid, predictor, algorithm="PPO", random_state=random_state)
        results["PPO"] = ppo.optimize(topology, profile)
        print(f"      Best time: {results['PPO'].best_predicted_time:.2f}s")
    except ImportError as e:
        print(f"[4/4] Skipping PPO: {e}")

    return results
