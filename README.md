# Spark Configuration Parameter Optimization

Система автоматической оптимизации конфигурационных параметров Apache Spark на основе машинного обучения.

## Цель проекта

Разработать рекомендательную систему, которая по заданной топологии кластера (количество воркеров, ядер, памяти) и характеристикам задачи предсказывает оптимальные значения 16 ключевых параметров Spark для минимизации времени выполнения.

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                    Рекомендательная система                      │
├─────────────────────────────────────────────────────────────────┤
│  Input:                                                          │
│  - Топология кластера (workers, cores, memory)                  │
│  - Тип задачи (wordcount, pagerank, kmeans, terasort)           │
│  - Размер входных данных                                         │
├─────────────────────────────────────────────────────────────────┤
│  Processing:                                                     │
│  1. DNN Performance Predictor → предсказание времени выполнения │
│  2. RL-based Parameter Search → поиск оптимальных параметров    │
├─────────────────────────────────────────────────────────────────┤
│  Output:                                                         │
│  - Оптимальная конфигурация Spark (16 параметров)               │
│  - Предсказанное время выполнения                                │
│  - Ожидаемое ускорение vs default                               │
└─────────────────────────────────────────────────────────────────┘
```

## Оптимизируемые параметры Spark (16 шт.)

| # | Параметр | Диапазон | Описание |
|---|----------|----------|----------|
| 1 | `executor.cores` | 1-8 | Ядра на executor |
| 2 | `executor.memory` | 1g-8g | Память executor |
| 3 | `executor.instances` | 1-8 | Количество executors |
| 4 | `driver.cores` | 1-4 | Ядра driver |
| 5 | `driver.memory` | 1g-4g | Память driver |
| 6 | `memory.fraction` | 0.3-0.8 | Доля памяти для execution/storage |
| 7 | `memory.storageFraction` | 0.3-0.8 | Доля storage в memory.fraction |
| 8 | `shuffle.compress` | true/false | Сжатие shuffle данных |
| 9 | `shuffle.spill.compress` | true/false | Сжатие spill данных |
| 10 | `shuffle.file.buffer` | 32k-128k | Буфер shuffle файлов |
| 11 | `broadcast.blockSize` | 4m-24m | Размер блока broadcast |
| 12 | `broadcast.compress` | true/false | Сжатие broadcast |
| 13 | `reducer.maxSizeInFlight` | 48m-96m | Макс. размер данных в полёте |
| 14 | `io.compression.codec` | lz4/snappy | Кодек сжатия |
| 15 | `rpc.message.maxSize` | 128-256 | Макс. размер RPC сообщения |
| 16 | `rdd.compress` | true/false | Сжатие RDD |

## Структура проекта

```
spark_machine_learning/
├── models/                             # ML модели
│   ├── __init__.py                     # Экспорты модуля
│   ├── data.py                         # Загрузка и препроцессинг данных
│   ├── baseline.py                     # Baseline: Dummy, RF, SA, MLP
│   ├── dnn_predictor.py                # DNN предиктор (PyTorch)
│   ├── rl_optimizer.py                 # RL: Q-Learning, DQN, PPO, Bayesian
│   └── README.md                       # Документация модуля
├── training/                           # Скрипты обучения
│   ├── train_baseline.py               # Обучение baseline с MLflow
│   ├── train_dnn.py                    # Обучение DNN предиктора
│   ├── train_rl.py                     # Обучение RL оптимизаторов
│   └── collect_agg.py                  # Агрегация CSV файлов
├── scripts/                            # Скрипты сбора данных
│   ├── run_wordcount_experiments.sh    # Оркестратор сбора данных
│   ├── collect_wordcount_data.sh       # Сбор датасета внутри HiBench
│   └── utils/                          # Вспомогательные скрипты
├── recommender/                        # Рекомендательная система (TODO)
│   ├── api.py                          # REST API (FastAPI)
│   ├── inference.py                    # Инференс моделей
│   └── config_generator.py             # Генерация spark.conf
├── docker/                             # Docker конфиги
│   ├── hibench/                        # HiBench образ
│   └── mlflow/                         # MLflow образ
├── config/                             # Конфигурации Hadoop/Spark
├── out/                                # Результаты экспериментов
├── docker-compose.yml                  # Spark + HDFS + MLflow кластер
└── README.md
```

## План разработки

### Фаза 1: Сбор данных
- [x] Настройка HiBench + Spark кластера
- [x] Скрипты сбора данных с медианой по N прогонов
- [x] Исправление docker-compose (namenode formatting, hibench healthcheck)
- [ ] **В процессе**: Сбор 150+ сэмплов WordCount (3 топологии × 50 сэмплов)
- [ ] Валидация и очистка датасета

### Фаза 2: Baseline модели ✅
- [x] Модуль загрузки данных (`models/data.py`)
- [x] DummyRegressor (median baseline)
- [x] RandomForestRegressor + RandomizedSearchCV
- [x] RandomForestRegressor + Simulated Annealing
- [x] MLP (sklearn)
- [x] Скрипт обучения с MLflow (`training/train_baseline.py`)
- [ ] Сравнительный анализ MAE/RMSE/MAPE/R² (после получения датасета)

### Фаза 3: DNN Performance Predictor ✅
- [x] Архитектура: Input(n) → Dense(128) → Dense(64) → Output(1)
- [x] Препроцессинг: OneHotEncoder + StandardScaler
- [x] Early stopping, learning rate scheduling
- [x] PyTorch реализация с sklearn fallback
- [x] Скрипт обучения (`training/train_dnn.py`)
- [ ] Обучение и оценка (после получения датасета)

### Фаза 4: Reinforcement Learning Optimizer ✅
- [x] **Q-Learning** (табличный, как в статье)
- [x] **Deep Q-Network (DQN)** с Experience Replay
- [x] **PPO/A2C** через stable-baselines3
- [x] **Bayesian Optimization** через Optuna
- [x] Скрипт оптимизации (`training/train_rl.py`)
- [ ] Обучение и сравнение алгоритмов (после получения датасета)

### Фаза 5: Рекомендательная система
- [ ] REST API (FastAPI)
- [ ] CLI интерфейс
- [ ] Генерация spark.conf файла
- [ ] Web UI (опционально)

### Фаза 6: Расширение на другие бенчмарки
- [ ] PageRank, K-Means, TeraSort
- [ ] Multi-task learning / Transfer learning

---

## Методы машинного обучения

### Baseline методы

#### Random Forest
Ансамбль деревьев решений, хорошо работает на табличных данных без нормализации.

**Где читать:**
- [Scikit-learn: Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Статья Breiman (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Towards Data Science: Understanding Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)

#### Simulated Annealing (Имитация отжига)
Метаэвристический алгоритм оптимизации, вдохновлённый процессом отжига металлов.

**Где читать:**
- [Wikipedia: Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
- [Статья Kirkpatrick et al. (1983)](https://www.science.org/doi/10.1126/science.220.4598.671)
- [Tutorials Point: Simulated Annealing](https://www.tutorialspoint.com/simulated-annealing-algorithm)

---

### Deep Neural Networks (DNN)

#### Многослойный персептрон (MLP)
Полносвязная нейросеть для регрессии. Используется как предиктор времени выполнения.

**Где читать:**
- [PyTorch Tutorial: Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [Deep Learning Book (Goodfellow): Ch. 6 Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
- [Scikit-learn: MLPRegressor](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

**Ключевые концепции:**
- Функции активации (ReLU, Tanh, Sigmoid)
- Backpropagation
- Регуляризация (Dropout, L2)
- Early Stopping

---

### Reinforcement Learning (Обучение с подкреплением)

#### Q-Learning (Tabular)
Классический RL-алгоритм, хранящий Q-значения в таблице. Используется в оригинальной статье.

**Где читать:**
- [Sutton & Barto: Ch. 6 Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — **главная книга по RL**
- [Hugging Face: Q-Learning](https://huggingface.co/learn/deep-rl-course/unit2/q-learning)
- [Wikipedia: Q-Learning](https://en.wikipedia.org/wiki/Q-learning)

**Формула:**
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

#### Deep Q-Network (DQN)
Q-Learning с нейросетью вместо таблицы. Позволяет работать с непрерывными/большими пространствами состояний.

**Где читать:**
- [Статья DeepMind (2015): Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602)
- [PyTorch Tutorial: DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Hugging Face: Deep Q-Learning](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-network)

**Ключевые концепции:**
- Experience Replay
- Target Network
- ε-greedy exploration

#### Policy Gradient: REINFORCE
Прямая оптимизация policy (стратегии) через градиентный спуск.

**Где читать:**
- [Sutton & Barto: Ch. 13 Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf)
- [Spinning Up: Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Lilian Weng: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

#### Proximal Policy Optimization (PPO)
Современный алгоритм policy gradient с ограничением изменения policy. Стабильный и эффективный.

**Где читать:**
- [Статья OpenAI (2017): PPO](https://arxiv.org/abs/1707.06347)
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Stable-Baselines3: PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

**Почему PPO популярен:**
- Проще чем TRPO, но такая же стабильность
- Хорошо масштабируется
- Работает как для дискретных, так и для непрерывных действий

#### Advantage Actor-Critic (A2C/A3C)
Комбинация policy gradient и value function. A3C — асинхронная версия.

**Где читать:**
- [Статья DeepMind (2016): A3C](https://arxiv.org/abs/1602.01783)
- [Stable-Baselines3: A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
- [Lilian Weng: Actor-Critic](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#actor-critic)

---

### Bayesian Optimization

Глобальная оптимизация чёрного ящика через построение surrogate model (обычно Gaussian Process).

**Где читать:**
- [A Tutorial on Bayesian Optimization (Frazier, 2018)](https://arxiv.org/abs/1807.02811)
- [Scikit-optimize (skopt)](https://scikit-optimize.github.io/stable/)
- [BoTorch (PyTorch-based)](https://botorch.org/docs/introduction)
- [Optuna: Hyperparameter Optimization](https://optuna.org/)

**Когда использовать:**
- Дорогие вычисления функции (как у нас — реальный запуск Spark)
- Малое количество итераций
- Непрерывные параметры

---

### Дополнительные методы (опционально)

#### Gradient-based Optimization
- **CMA-ES**: эволюционная стратегия для непрерывной оптимизации
- [CMA-ES Tutorial](https://arxiv.org/abs/1604.00772)

#### Meta-Learning
- Обучение на нескольких задачах для быстрой адаптации к новым
- [MAML: Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400)

#### Transfer Learning для RL
- Перенос знаний между похожими задачами (WordCount → PageRank)
- [Survey: Transfer Learning in RL](https://arxiv.org/abs/2009.07888)

---

## Рекомендуемый порядок изучения

1. **Основы RL** → Sutton & Barto, главы 1-6
2. **Q-Learning** → Реализовать tabular Q-learning для простой среды
3. **DQN** → PyTorch tutorial + Hugging Face курс
4. **Policy Gradient** → Spinning Up: REINFORCE → PPO
5. **Практика** → Stable-Baselines3 для готовых реализаций

---

## Методология (по статье Huang et al.)

### DNN Predictor
```
Input Features (22):
- Топология: workers, worker_cores, worker_mem_gb (3)
- Параметры Spark: 16 параметров
- Задача: profile, input_size (3)

Architecture:
Input(22) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1)

Training:
- Loss: MSE
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Early stopping: patience=20
```

### Q-Learning Optimizer
```
State Space: дискретизированные значения 16 параметров
Action Space: {увеличить, уменьшить, не менять} × 16 параметров = 48 actions

Q-table update:
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]

Parameters:
- α (learning rate): 0.1
- γ (discount): 0.95
- ε (exploration): 0.3 → 0.05 (decay)
```

---

## Запуск

### 0. Установка зависимостей
```bash
pip install numpy pandas scikit-learn matplotlib joblib mlflow
pip install torch                        # для DNN и DQN
pip install optuna                       # для Bayesian Optimization
pip install gymnasium stable-baselines3  # для PPO/A2C
```

### 1. Запуск кластера
```bash
docker-compose up -d
# Проверить статус
docker-compose ps
```

### 2. Сбор датасета
```bash
# Запустить сбор (выполняется внутри hibench контейнера)
docker exec -it hibench bash
cd /opt/hibench
./scripts/collect_wordcount_data.sh

# Забрать результат
docker cp hibench:/opt/hibench/report/wc_train_all.csv ./out/
```

### 3. Обучение baseline моделей
```bash
python training/train_baseline.py \
    --csv ./out/wc_train_all.csv \
    --outdir ./out/baseline \
    --mlflow
```

### 4. Обучение DNN предиктора
```bash
python training/train_dnn.py \
    --csv ./out/wc_train_all.csv \
    --outdir ./out/dnn \
    --hidden-sizes 128,64 \
    --epochs 500 \
    --mlflow
```

### 5. Запуск RL оптимизации
```bash
python training/train_rl.py \
    --csv ./out/wc_train_all.csv \
    --dnn-model ./out/dnn/model \
    --outdir ./out/rl \
    --workers 4 --worker-cores 2 --worker-mem 4 \
    --profile large \
    --algorithm all  # или: qlearning, dqn, ppo, bayesian
```

### 6. Просмотр результатов в MLflow
```bash
# MLflow UI доступен по адресу http://localhost:5000
```

### Запуск рекомендательной системы (TODO)
```bash
python recommender/api.py --port 8080
# POST http://localhost:8080/recommend
# {"workers": 4, "worker_cores": 2, "worker_mem_gb": 4, "task": "wordcount", "input_gb": 10}
```

---

## Метрики качества

| Метрика | Формула | Цель |
|---------|---------|------|
| MAE | mean(\|y - ŷ\|) | < 5 сек |
| RMSE | sqrt(mean((y - ŷ)²)) | < 10 сек |
| MAPE | mean(\|y - ŷ\| / y) × 100% | < 10% |
| R² | 1 - SS_res/SS_tot | > 0.9 |
| Speedup | time_default / time_optimized | > 1.5x |

---

## Зависимости

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.4
mlflow>=2.0
torch>=2.0          # для DNN
gymnasium>=0.28     # для RL (gym successor)
stable-baselines3   # для PPO, A2C, DQN
fastapi>=0.100      # для API
uvicorn>=0.22       # ASGI server
optuna>=3.0         # для Bayesian Optimization
```

---

## Ссылки

### Основные
- [Статья: Spark Configuration Parameter Optimization](./docs/spark_config_optimization.pdf)
- [HiBench Benchmark Suite](https://github.com/Intel-bigdata/HiBench)
- [Spark Configuration Guide](https://spark.apache.org/docs/latest/configuration.html)

### Книги
- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/RLbook2020.pdf) — **бесплатно онлайн**
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/) — **бесплатно онлайн**

### Курсы
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) — **бесплатно**
- [OpenAI Spinning Up](https://spinningup.openai.com/) — **бесплатно**
- [DeepMind x UCL RL Lectures](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm) — **YouTube**

### Библиотеки
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — готовые RL алгоритмы
- [Gymnasium](https://gymnasium.farama.org/) — среды для RL
- [Optuna](https://optuna.org/) — Bayesian Optimization
