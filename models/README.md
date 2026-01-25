# Модуль models/

Этот модуль содержит все ML-модели для оптимизации конфигурации Spark.

## Общая архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                         Входные данные                          │
│  CSV: топология кластера + 16 параметров Spark + время          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      data.py (препроцессинг)                    │
│  • Парсинг памяти (1g → 1024 MB)                               │
│  • Конвертация bool (true/false → 1/0)                         │
│  • StandardScaler для числовых                                  │
│  • OneHotEncoder для категориальных                            │
│  • Train/Val/Test split с стратификацией                       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   baseline.py    │ │ dnn_predictor.py │ │  rl_optimizer.py │
│                  │ │                  │ │                  │
│ • Dummy (медиана)│ │ • PyTorch DNN    │ │ • Q-Learning     │
│ • RF + Search    │ │ • 128 → 64 → 1   │ │ • DQN            │
│ • RF + SA        │ │ • Early stopping │ │ • PPO (SB3)      │
│ • MLP (sklearn)  │ │ • log1p(target)  │ │ • Bayesian       │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                     │                    │
        │                     ▼                    │
        │            ┌──────────────────┐          │
        │            │    Предиктор     │◄─────────┘
        │            │  времени Spark   │  (использует для
        │            └──────────────────┘   поиска оптимума)
        │                     │
        └─────────────────────┼─────────────────────
                              ▼
                    Оптимальная конфигурация
```

---

## Файлы модуля

### 1. `data.py` — Загрузка и препроцессинг

**Что делает:**
- Читает CSV с результатами бенчмарков
- Парсит строки памяти (`"4g"` → `4096.0` MB)
- Конвертирует булевы значения (`"true"` → `1`)
- Нормализует числовые фичи (StandardScaler)
- Кодирует категориальные (OneHotEncoder)
- Разбивает на train/val/test с сохранением пропорций по `profile`

**Ключевые функции:**
```python
# Загрузить датасет и получить готовые splits
dataset = create_dataset("./out/wc_train_all.csv")

# Получить numpy массивы для обучения
X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits()

# Получить сетку параметров для RL
param_grid = dataset.get_param_grid()
```

**Формат CSV:**
```
topology_workers,topology_worker_cores,topology_worker_mem_gb,profile,
executor_cores,executor_memory,executor_instances,...,median_duration_s,exit_code
4,2,4,large,2,2g,4,1,1g,0.6,0.5,true,true,64k,8m,true,72m,lz4,128,false,45.2,0
```

---

### 2. `baseline.py` — Baseline модели

**Что делает:**
Реализует простые модели для сравнения с более сложными (DNN, RL).

**Модели:**

| Класс | Описание | Когда использовать |
|-------|----------|-------------------|
| `DummyBaseline` | Предсказывает медиану | Нижняя граница качества |
| `RandomForestBaseline` | RF + RandomizedSearchCV | Хорошо на табличных данных |
| `SimulatedAnnealingRF` | RF + имитация отжига | Поиск глобального оптимума |
| `MLPBaseline` | Нейросеть sklearn | Простая альтернатива PyTorch |

**Simulated Annealing:**
```
Алгоритм:
1. Начинаем со случайных гиперпараметров RF
2. На каждой итерации:
   - Генерируем соседнее решение
   - Если лучше — принимаем
   - Если хуже — принимаем с вероятностью exp(-Δ/T)
3. Температура T постепенно снижается (T *= 0.93)
4. Это позволяет выходить из локальных минимумов
```

**Пример:**
```python
from models import train_all_baselines

results = train_all_baselines(X_train, y_train, X_val, y_val, X_test, y_test)
for r in results:
    print(f"{r.name}: MAE={r.metrics['MAE']:.2f}")
```

---

### 3. `dnn_predictor.py` — DNN предиктор

**Что делает:**
Нейросеть для предсказания времени выполнения Spark по конфигурации.

**Архитектура:**
```
Input(n_features)
    │
    ▼
Linear(128) → ReLU → Dropout(0.1)
    │
    ▼
Linear(64) → ReLU → Dropout(0.1)
    │
    ▼
Linear(1) → выход (log-время)
```

**Особенности:**
- **log1p transform**: обучаемся на `log(1 + время)`, это стабилизирует loss
- **Early stopping**: останавливаемся если val_RMSE не улучшается 20 эпох
- **LR scheduling**: уменьшаем lr если застряли на плато

**Пример:**
```python
from models import DNNPredictor, DNNConfig

# Создание с кастомными параметрами
config = DNNConfig(
    hidden_sizes=(128, 64),
    learning_rate=0.001,
    batch_size=32,
    patience=20
)
predictor = DNNPredictor(config)

# Обучение
predictor.fit(X_train, y_train, X_val, y_val, verbose=True)

# Предсказание (возвращает секунды, не log)
y_pred = predictor.predict(X_test)

# Сохранение/загрузка
predictor.save("./out/dnn/model")
loaded = DNNPredictor.load("./out/dnn/model")
```

---

### 4. `rl_optimizer.py` — RL оптимизаторы

**Что делает:**
Ищет оптимальную конфигурацию Spark используя RL-алгоритмы.

**Алгоритмы:**

#### Q-Learning (табличный)
```
Идея: храним таблицу Q(состояние, действие) → ожидаемая награда

Состояние = текущие индексы параметров (например, executor_cores=2 → индекс 1)
Действие = изменить один параметр на +1 или -1
Награда = (старое_время - новое_время) / старое_время

Обновление Q-таблицы:
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

где:
  α = 0.1 (learning rate)
  γ = 0.95 (discount factor)
  ε = 0.3→0.05 (exploration, постепенно снижается)
```

#### DQN (Deep Q-Network)
```
То же что Q-Learning, но Q-функция — нейросеть.
Позволяет работать с большими пространствами состояний.

Ключевые техники:
- Experience Replay: сохраняем переходы, учимся на случайных batch'ах
- Target Network: отдельная сеть для стабильности
```

#### PPO/A2C (stable-baselines3)
```
Policy Gradient методы — напрямую оптимизируют стратегию.
PPO добавляет ограничение на размер обновления для стабильности.

Используем готовую реализацию из stable-baselines3.
```

#### Bayesian Optimization (Optuna)
```
Строит surrogate model (обычно Gaussian Process) для предсказания
качества конфигурации. Выбирает следующую точку максимизируя
acquisition function (баланс exploration/exploitation).

Эффективен когда мало итераций (дорогие вычисления).
```

**Пример:**
```python
from models import TabularQLearning, DNNPredictor

# Загружаем предиктор
predictor = DNNPredictor.load("./out/dnn/model")

# Создаём wrapper для предсказания
def predict_fn(df):
    X = preprocess(df)  # препроцессинг
    return predictor.predict(X)

# Запускаем Q-Learning
optimizer = TabularQLearning(
    param_grid=dataset.get_param_grid(),
    predictor=predict_fn,
    alpha=0.1,      # learning rate
    gamma=0.95,     # discount
    epsilon=0.3,    # exploration
)

result = optimizer.optimize(
    topology={"topology_workers": 4, "topology_worker_cores": 2, "topology_worker_mem_gb": 4},
    profile="large",
    n_episodes=100,
)

print(f"Лучшее время: {result.best_predicted_time:.2f}s")
print(f"Конфигурация: {result.best_config}")
```

---

## Метрики

| Метрика | Формула | Что показывает |
|---------|---------|----------------|
| MAE | `mean(\|y - ŷ\|)` | Средняя ошибка в секундах |
| RMSE | `sqrt(mean((y - ŷ)²))` | Штраф за большие ошибки |
| MAPE | `mean(\|y - ŷ\| / y) × 100%` | Относительная ошибка |
| R² | `1 - SS_res/SS_tot` | Доля объяснённой дисперсии |
| Speedup | `time_default / time_optimized` | Ускорение vs baseline |

---

## Зависимости

```
# Обязательные
numpy, pandas, scikit-learn, matplotlib, joblib

# Для DNN
torch  # или используется sklearn.MLPRegressor как fallback

# Для Bayesian Optimization
optuna

# Для PPO/A2C
gymnasium, stable-baselines3
```

---

## Быстрый старт

```python
from models import create_dataset, train_all_baselines, DNNPredictor, TabularQLearning

# 1. Загружаем данные
dataset = create_dataset("./out/wc_train_all.csv")
X_tr, X_val, X_te, y_tr, y_val, y_te = dataset.get_splits()

# 2. Обучаем baseline
results = train_all_baselines(X_tr, y_tr, X_val, y_val, X_te, y_te)

# 3. Обучаем DNN
dnn = DNNPredictor()
dnn.fit(X_tr, y_tr, X_val, y_val)
dnn.save("./model")

# 4. Ищем оптимум через Q-Learning
ql = TabularQLearning(dataset.get_param_grid(), lambda df: dnn.predict(preprocess(df)))
result = ql.optimize(topology, profile="large")
print(result.best_config)
```
