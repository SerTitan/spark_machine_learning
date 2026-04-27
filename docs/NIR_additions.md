# РАЗДЕЛ 3. АНАЛИЗ И ПРИМЕНЕНИЕ МЕТОДОВ МАШИННОГО ОБУЧЕНИЯ

## 3.0. Актуальное состояние (дата: 2026-04-24)

- **Текущий обучающий датасет**: `data/wc_train_merged.csv` — 401 успешная строка, профиль `large`, 3 топологии (`2×4×8`, `3×3×6`, `4×2×4`). Сырые снапшоты — `data/snapshots/`, устаревшие версии перенесены в `data/archive/`.
- **Расширение датасета** в процессе: переход к новой методике сбора (generic collector `scripts/collect_hibench_data.sh`, CSV со схемой `schema_version=2`). Планируется собрать три нагрузки: TeraSort, PageRank, WordCount; опционально KMeans.
- **Лучший baseline** на текущем WordCount-датасете: `RandomForest_SimulatedAnnealing` — **MAE 0.746 s, RMSE 1.215 s, R² 0.961, MAPE 3.19%** (отчёт: `out/final_best/baseline/report.json`).
- **DNN (PyTorch)** на том же датасете — **MAE 3.038 s, RMSE 3.949 s, R² 0.591, MAPE 13.18%** (отчёт: `out/final_best/dnn/report.json`). В продакшне не используется.
- **API**: `api/main.py` на FastAPI, эндпоинты `/health` и `/predict`. Модель и препроцессор загружаются из `out/final_best/baseline`. Текущая версия API рассчитана только на WordCount-схему одной топологии; расширение до multi-workload запланировано после сбора нового датасета.
- **RL-оптимизация** (последний прогон: `out/rl_topo336_constrained/report.json`, профиль `large`, топология `workers=3, cores=3, mem=6 GB`):
  - Q-Learning: 15.84 s (speedup 1.17×, +14.6% к дефолту);
  - DQN: 15.85 s (speedup 1.17×, +14.6%);
  - Bayesian (Optuna): 15.87 s (speedup 1.17×, +14.5%);
  - PPO: 15.88 s (speedup 1.17×, +14.4%).
  - Пример лучшей конфигурации (Q-Learning): `executor_cores=4, executor_instances=3, executor_memory=3g, driver_cores=1, driver_memory=4g, memory_fraction=0.5, memory_storageFraction=0.4, shuffle_file_buffer=128k, broadcast_block=20m, maxSizeInFlight=88m, shuffle_compress=false, spill_compress=false, broadcast_compress=false, rdd_compress=true, io_codec=lz4, rpc_message_maxSize=160`.
- **Главное ограничение**: текущая модель обучена только на WordCount и только на `large`; после расширения датасета постановка ML-задачи меняется на multi-workload.

## 3.1. Предобработка и анализ собранных данных

### 3.1.1. Обоснование необходимости предобработки

После сбора экспериментальных данных я приступил к этапу предобработки, который является критически важным для успешного обучения моделей машинного обучения [1]. Собранный датасет содержал параметры конфигурации Spark различных типов (числовые, категориальные, параметры памяти в виде строк типа "4g", "512m") и требовал приведения к единому формату для корректной работы алгоритмов обучения.

Основные задачи предобработки включали:
- Парсинг параметров памяти из строкового формата в числовой
- Нормализацию числовых признаков для устранения влияния масштаба
- Кодирование категориальных признаков
- Трансформацию целевой переменной для стабилизации обучения

### 3.1.2. Парсинг параметров памяти

Параметры Spark, связанные с памятью (например, `spark.executor.memory`, `spark.driver.memory`), записываются в формате строк с суффиксами "g" (гигабайты), "m" (мегабайты), "k" (килобайты). Я реализовал функцию парсинга, конвертирующую эти значения в мегабайты для унификации:

```python
def parse_memory(mem_str):
    if isinstance(mem_str, (int, float)):
        return float(mem_str)
    mem_str = str(mem_str).strip().lower()
    if mem_str.endswith('g'):
        return float(mem_str[:-1]) * 1024
    elif mem_str.endswith('m'):
        return float(mem_str[:-1])
    elif mem_str.endswith('k'):
        return float(mem_str[:-1]) / 1024
    return float(mem_str)
```

### 3.1.3. Нормализация числовых признаков

Для нормализации числовых признаков я применил StandardScaler из библиотеки scikit-learn [2]. Стандартизация выполняется по формуле:

```
z = (x - μ) / σ
```

где:
- `x` — исходное значение признака
- `μ` — среднее значение признака по обучающей выборке
- `σ` — стандартное отклонение признака
- `z` — нормализованное значение

Стандартизация приводит все числовые признаки к одному масштабу (среднее = 0, стандартное отклонение = 1), что предотвращает доминирование признаков с большими значениями и ускоряет сходимость градиентных методов оптимизации [3].

### 3.1.4. Кодирование категориальных признаков

Категориальные параметры (такие как `spark.io.compression.codec` со значениями "lz4", "snappy") я закодировал с использованием One-Hot Encoding [4]. Этот метод создаёт бинарный вектор для каждого уникального значения категории:

Исходное значение: `codec = "lz4"`
Результат кодирования: `[codec_lz4=1, codec_snappy=0]`

One-Hot Encoding предотвращает ошибочное внесение порядковых отношений между категориями, которое возникает при простом числовом кодировании.

### 3.1.5. Логарифмическая трансформация целевой переменной

Распределение времени выполнения Spark-задач часто имеет правостороннюю асимметрию (большинство значений сконцентрировано в левой части, но есть выбросы с большим временем). Для стабилизации дисперсии и улучшения качества предсказаний я применил логарифмическую трансформацию [5]:

```
y' = log(1 + y)
```

где `y` — исходное время выполнения, `y'` — трансформированная целевая переменная. Добавление единицы (функция `log1p`) предотвращает проблемы с нулевыми значениями.

---

## 3.2. Реализация baseline моделей

**Обоснование необходимости baseline моделей:**

В работе Huang et al. [16] авторы сразу перешли к глубокой нейронной сети для предсказания производительности. Однако я принял решение начать с более простых и интерпретируемых методов по следующим причинам:

1. **Установление точки отсчёта:** Baseline модели позволяют оценить минимальный порог качества и понять, насколько сложные методы оправданы для данной задачи
2. **Интерпретируемость:** Random Forest позволяет анализировать важность признаков (feature importance), что помогает понять какие параметры Spark наиболее критичны
3. **Быстрота экспериментов:** Обучение классических ML моделей занимает секунды, в то время как DNN требует подбора гиперпараметров и может обучаться минуты
4. **Проверка гипотез:** Если простые модели дают хорошее качество, возможно сложная архитектура не нужна

После консультации с научным руководителем было решено протестировать несколько baseline подходов перед переходом к глубоким нейросетям.

---

### 3.2.1. DummyRegressor — базовая оценка

**Обоснование выбора:**
Перед применением сложных моделей важно установить baseline — простейший метод, который служит точкой отсчёта для оценки эффективности более продвинутых подходов [6]. DummyRegressor предсказывает среднее значение целевой переменной по обучающей выборке, игнорируя входные признаки.

**Алгоритм:**
```
ŷ = (1/n) * Σ(yi)
```

где `n` — количество примеров в обучающей выборке, `yi` — значения целевой переменной.

**Этапы реализации:**
1. Импорт библиотеки: `from sklearn.dummy import DummyRegressor`
2. Инициализация модели: `dummy = DummyRegressor(strategy='mean')`
3. Обучение на train-выборке: `dummy.fit(X_train, y_train)`
4. Предсказание на test-выборке: `y_pred = dummy.predict(X_test)`
5. Оценка метрик (MAE, RMSE, R²)

**Результат:**
На актуальном WordCount-датасете DummyRegressor дал ошибку, кратно превышающую ошибку RF-моделей (baseline порядка среднего времени выполнения — ~18–20 s); любая модель с MAE хуже этого значения бесполезна. Точные числа — в `out/final_best/baseline/report.json`.

---

### 3.2.2. Random Forest с RandomizedSearchCV

**Обоснование выбора:**
Random Forest — мощный ансамблевый метод, устойчивый к переобучению и способный выявлять нелинейные зависимости между признаками [7]. Метод основан на построении множества деревьев решений на различных подвыборках данных и усреднении их предсказаний.

**Алгоритм Random Forest:**
1. Создаётся `n_estimators` деревьев решений
2. Каждое дерево обучается на bootstrap-выборке (случайная выборка с возвращением из обучающих данных)
3. При построении каждого узла дерева рассматривается случайное подмножество признаков размера `max_features`
4. Итоговое предсказание: среднее арифметическое предсказаний всех деревьев

```
ŷ = (1/T) * Σ(ht(x))
```

где `T` — количество деревьев, `ht(x)` — предсказание t-го дерева.

**Подбор гиперпараметров через RandomizedSearchCV:**
Для оптимизации гиперпараметров я использовал случайный поиск [8], который эффективнее полного перебора (Grid Search) при большом пространстве параметров:

```python
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)
```

**Этапы реализации:**
1. Определение пространства гиперпараметров
2. Запуск RandomizedSearchCV с 5-fold кросс-валидацией
3. Выбор лучшей комбинации параметров по метрике MAE
4. Обучение финальной модели на полной обучающей выборке
5. Оценка качества на test-выборке

---

### 3.2.3. Random Forest с оптимизацией через Simulated Annealing

**Обоснование выбора:**
Simulated Annealing (метод имитации отжига) — метаэвристический алгоритм глобальной оптимизации, вдохновлённый процессом кристаллизации металлов [9]. В отличие от градиентных методов, он способен избегать локальных минимумов за счёт вероятностного принятия худших решений на ранних итерациях.

**Алгоритм Simulated Annealing:**
1. Инициализация начального решения `s0` и температуры `T0`
2. На каждой итерации:
   - Генерация соседнего решения `s'` путём небольшого изменения текущего `s`
   - Вычисление изменения целевой функции: `ΔE = f(s') - f(s)`
   - Если `ΔE < 0` (улучшение), принять `s'`
   - Если `ΔE ≥ 0` (ухудшение), принять с вероятностью `P = exp(-ΔE / T)`
3. Снижение температуры: `T = α * T`, где `α ∈ (0, 1)` — коэффициент охлаждения
4. Повторение до достижения критерия останова

**Применение к Random Forest:**
Я оптимизировал гиперпараметры Random Forest с помощью Simulated Annealing:

```python
def objective(params):
    """Целевая функция: MAE модели с данными параметрами"""
    rf = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        random_state=42
    )
    scores = cross_val_score(rf, X_train, y_train,
                             cv=5, scoring='neg_mean_absolute_error')
    return -scores.mean()  # Минимизируем MAE

def simulated_annealing(objective, bounds, T0=100, alpha=0.95, max_iter=100):
    # Начальное решение
    current = {k: random.uniform(v[0], v[1]) for k, v in bounds.items()}
    current_cost = objective(current)
    best = current.copy()
    best_cost = current_cost
    T = T0

    for i in range(max_iter):
        # Генерация соседа
        neighbor = generate_neighbor(current, bounds)
        neighbor_cost = objective(neighbor)

        # Решение о принятии
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_cost = neighbor_cost

        # Обновление лучшего
        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost

        # Охлаждение
        T *= alpha

    return best, best_cost
```

**Этапы реализации:**
1. Определение границ гиперпараметров
2. Реализация функции генерации соседних решений
3. Запуск алгоритма Simulated Annealing
4. Обучение Random Forest с найденными параметрами
5. Сравнение с результатами RandomizedSearchCV

---

### 3.2.4. MLP нейросеть (Multi-Layer Perceptron)

**Обоснование выбора:**
Multi-Layer Perceptron — базовый тип искусственной нейронной сети прямого распространения, способный аппроксимировать произвольные нелинейные функции [10]. В отличие от деревьев решений, MLP может улавливать сложные взаимодействия между признаками благодаря нелинейным функциям активации.

**Архитектура MLP:**
Нейросеть состоит из последовательных слоёв нейронов:
- **Входной слой:** 16 нейронов (по количеству параметров Spark)
- **Скрытые слои:** 2 слоя по 64 и 32 нейрона соответственно
- **Выходной слой:** 1 нейрон (предсказание времени выполнения)

**Функция активации ReLU:**
В скрытых слоях я использовал ReLU (Rectified Linear Unit) [11]:

```
ReLU(x) = max(0, x)
```

Преимущества ReLU:
- Устраняет проблему затухающих градиентов
- Вычислительно эффективна
- Обеспечивает разреженность активаций (часть нейронов = 0)

**Функция потерь — Mean Squared Error:**

```
L(θ) = (1/n) * Σ(yi - ŷi)²
```

где `θ` — веса нейросети, `yi` — истинное значение, `ŷi` — предсказание.

**Оптимизатор Adam:**
Для обучения я использовал адаптивный оптимизатор Adam [12], который комбинирует идеи Momentum и RMSProp:

```
mt = β1 * mt-1 + (1 - β1) * gt
vt = β2 * vt-1 + (1 - β2) * gt²
θt = θt-1 - α * mt / (√vt + ε)
```

где:
- `gt` — градиент на шаге t
- `mt` — первый момент (скользящее среднее градиента)
- `vt` — второй момент (скользящее среднее квадрата градиента)
- `α` — learning rate
- `β1, β2` — коэффициенты экспоненциального затухания (обычно 0.9 и 0.999)

**Этапы реализации:**
1. Импорт библиотеки: `from sklearn.neural_network import MLPRegressor`
2. Инициализация модели:
```python
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42
)
```
3. Обучение на train-выборке с мониторингом validation loss
4. Early stopping — остановка обучения при отсутствии улучшения на validation в течение 10 эпох
5. Оценка качества на test-выборке

---

## 3.3. Разработка DNN предиктора на PyTorch

### 3.3.1. Обоснование выбора глубокой нейросети

**Следование методологии Huang et al.:**

В статье [16] авторы использовали глубокую нейронную сеть (DNN) в качестве предиктора производительности Spark-приложений. Они обосновали этот выбор следующим образом:

> "Deep-learning neural networks can gradually learn through multiple networks, extract complex and effective features, and have better prediction accuracy and generalization ability compared to shallow machine-learning methods" [16, стр. 2]

После проверки baseline моделей я также перешёл к DNN предиктору, но с некоторыми адаптациями:

**Архитектура в статье Huang et al.:**
- Input(16) → Dense(12) → Dense(8) → Dense(4) → Output(1)
- Activation: ReLU
- Optimizer: Adam
- Loss: MSE

**Моя адаптированная архитектура:**
- Input(16) → Dense(128) → Dense(64) → Output(1)
- Activation: ReLU
- Optimizer: Adam + learning rate scheduling
- Loss: MSE на логарифмически трансформированных данных

**Обоснование изменений:**
1. **Больше нейронов (128, 64 vs 12, 8, 4):** Увеличенная размерность скрытых слоёв позволяет модели выявлять более сложные нелинейные зависимости между 16 параметрами
2. **Меньше слоёв (2 vs 3):** Упрощение архитектуры снижает риск переобучения при ограниченном размере датасета
3. **Логарифмическая трансформация:** Стабилизирует распределение целевой переменной (время выполнения часто имеет long-tail распределение)
4. **Learning rate scheduling:** Динамическая адаптация скорости обучения улучшает сходимость

Реализация на PyTorch [13] вместо Keras (как в оригинальной статье) обусловлена:
- Большей гибкостью в кастомизации процесса обучения
- Детальным контролем над градиентами и оптимизацией
- Возможностью использования GPU для ускорения вычислений
- Лучшей интеграцией с современными RL библиотеками (stable-baselines3)

### 3.3.2. Архитектура DNN предиктора

Я спроектировал полносвязную нейросеть следующей архитектуры:

```
Вход (16 признаков) → FC(128) → ReLU → FC(64) → ReLU → FC(1) → Выход
```

где FC(n) — fully connected layer с n нейронами.

**Обоснование архитектуры:**
- Первый слой (128 нейронов) — увеличение размерности для выявления сложных паттернов
- Второй слой (64 нейрона) — постепенное сжатие представления
- Выходной слой (1 нейрон) — регрессия времени выполнения

**Реализация на PyTorch:**

```python
import torch
import torch.nn as nn

class DNNPredictor(nn.Module):
    def __init__(self, input_dim=16):
        super(DNNPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 3.3.3. Логарифмическая трансформация целевой переменной

Для стабилизации обучения я применил логарифмическое преобразование к целевой переменной:

```python
y_train_log = np.log1p(y_train)  # log(1 + y)
```

При предсказании выполняется обратное преобразование:

```python
y_pred = np.expm1(model(X_test))  # exp(y') - 1
```

### 3.3.4. Learning Rate Scheduling

Для динамической адаптации скорости обучения я реализовал ReduceLROnPlateau scheduler [14]:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Уменьшение lr в 2 раза
    patience=10,     # Количество эпох без улучшения
    verbose=True
)
```

**Принцип работы:**
Если validation loss не улучшается в течение 10 эпох, learning rate уменьшается в 2 раза. Это помогает модели "дообучиться" в областях, близких к локальному минимуму.

### 3.3.5. Early Stopping

Для предотвращения переобучения я реализовал механизм ранней остановки:

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 20

for epoch in range(num_epochs):
    # ... обучение ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Сохранение лучших весов
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 3.3.6. Этапы обучения DNN

1. **Подготовка данных:**
   - Преобразование pandas DataFrame в PyTorch тензоры
   - Логарифмическая трансформация целевой переменной
   - Разделение на train/validation/test

2. **Инициализация модели и оптимизатора:**
   ```python
   model = DNNPredictor(input_dim=16)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()
   ```

3. **Обучающий цикл:**
   ```python
   for epoch in range(num_epochs):
       model.train()
       for X_batch, y_batch in train_loader:
           optimizer.zero_grad()
           outputs = model(X_batch)
           loss = criterion(outputs, y_batch)
           loss.backward()
           optimizer.step()

       # Валидация
       model.eval()
       with torch.no_grad():
           val_loss = criterion(model(X_val), y_val)

       scheduler.step(val_loss)
   ```

4. **Оценка на test-выборке:**
   - Загрузка лучших весов
   - Предсказание на тестовых данных
   - Обратное log-преобразование
   - Вычисление метрик (MAE, RMSE, R²)

---

## 3.4. Сравнительный анализ моделей и оценка качества

### 3.4.1. Метрики качества

Для оценки моделей я использовал следующие метрики [15]:

**Mean Absolute Error (MAE):**
```
MAE = (1/n) * Σ|yi - ŷi|
```
Интерпретация: средняя абсолютная ошибка предсказания в секундах.

**Root Mean Squared Error (RMSE):**
```
RMSE = √[(1/n) * Σ(yi - ŷi)²]
```
Интерпретация: сильнее штрафует большие ошибки по сравнению с MAE.

**R² Score (коэффициент детерминации):**
```
R² = 1 - [Σ(yi - ŷi)² / Σ(yi - ȳ)²]
```
Интерпретация: доля дисперсии, объяснённая моделью. R² = 1 означает идеальное предсказание.

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (100%/n) * Σ|((yi - ŷi) / yi)|
```
Интерпретация: средняя относительная ошибка в процентах.

### 3.4.2. Результаты сравнения моделей

Итоговые метрики на test split датасета `wc_train_merged.csv` (401 строка, profile=large, 3 топологии):

| Модель | MAE (с) | RMSE (с) | R² | MAPE (%) |
|--------|---------|----------|-----|----------|
| DummyRegressor | ≈18–20 | ≈22 | ~0 | ≈100 |
| Random Forest (RandomSearch) | 0.815 | 1.189 | 0.963 | 3.58 |
| Random Forest (SimAnnealing) | **0.746** | **1.215** | **0.961** | **3.19** |
| MLP (sklearn) | ~1.2 | ~1.6 | ~0.93 | ~5 |
| DNN (PyTorch) | 3.038 | 3.949 | 0.591 | 13.18 |

Полные отчёты: `out/final_best/baseline/report.json`, `out/final_best/dnn/report.json`.

### 3.4.3. Выводы

1. Все модели превзошли DummyRegressor, что подтверждает наличие значимой зависимости между параметрами конфигурации и временем выполнения WordCount.
2. **Random Forest с Simulated Annealing оказался лучшим** по комбинации MAE/MAPE и используется в прод-API (`api/main.py`).
3. **DNN проиграл** на текущем датасете (401 строка — мало для глубокой сети). После расширения датасета до multi-workload имеет смысл переобучить DNN с `job_type`, `input_size_bytes` как признаками и сравнить заново.
4. RandomForest даёт интерпретируемость: feature importance показывает, что наиболее критичны `executor_cores`, `executor_memory`, `shuffle_file_buffer`, `memory_fraction`.
5. Логарифмическая трансформация целевой переменной значительно улучшила стабильность обучения нейросетей, но в RF-моделях разницы не дала.

---

## 3.5. Прототип API рекомендательного сервиса (текущая версия)

Реализован FastAPI-сервис `api/main.py`. На момент 2026-04-24 он работает в режиме single-workload (WordCount) и служит заготовкой для multi-workload расширения после сбора нового датасета.

Эндпоинты:

- `GET /health` — проверка загрузки модели и препроцессора;
- `POST /predict` — принимает конкретную Spark-конфигурацию (16 параметров + топологию) и возвращает предсказанное время выполнения.

Во время старта сервис загружает артефакты из `out/final_best/baseline/`: обученную RF+SA модель, StandardScaler и OneHotEncoder.

После того как будет собран multi-workload датасет, API расширяется до следующих эндпоинтов: `/models`, `/workloads`, `/recommend`, `/compare`, `/validate-result`. Ключевое изменение входной схемы: пользователь будет передавать **свои системные ресурсы и размер данных**, а не HiBench-профиль и не Spark-конфигурацию напрямую. Сервис сам генерирует допустимых кандидатов Spark-config, прогоняет их через предиктор и возвращает топ-K рекомендаций.

---

# РАЗДЕЛ 4. РАЗРАБОТКА ПРОТОТИПА РЕКОМЕНДАТЕЛЬНОГО СЕРВИСА

## 4.1. Проектирование архитектуры прототипа

### 4.1.1. Общий подход и связь с работой Huang et al.

После реализации DNN предиктора я приступил к разработке компонента поиска оптимальных параметров конфигурации. В статье Huang et al. [16] авторы использовали улучшенный алгоритм Q-Learning для решения этой задачи. Я принял решение реализовать не только метод из статьи, но и несколько альтернативных подходов для сравнительного анализа.

**Обоснование множественности подходов:**
1. **Валидация результатов статьи:** Проверить работоспособность улучшенного Q-Learning на нашем датасете
2. **Сравнительный анализ:** Оценить насколько современные методы (DQN, PPO) превосходят классический подход
3. **Масштабируемость:** Проверить как методы работают при увеличении пространства параметров
4. **Практическая применимость:** Выбрать наилучший метод для финального прототипа

### 4.1.2. Формализация задачи как Markov Decision Process (MDP)

Задачу оптимизации параметров Spark я сформулировал как MDP [17]:

**Пространство состояний (S):**
- Состояние `s` = конкретная конфигурация 16 параметров Spark
- Представление: дискретные индексы значений каждого параметра
- Пример: `s = (cores=4, memory=2g, shuffle.compress=true, ...)`

**Пространство действий (A):**
- Действие `a` = изменение одного параметра на +1 или -1 шаг в сетке значений
- Количество действий: ≈ 32 (по 2 действия на каждый из 16 параметров)
- Пример: `a = "увеличить executor.cores на 1"`

**Функция награды (R):**
```
r(s, a, s') = (t_предсказание(s) - t_предсказание(s')) / t_предсказание(s)
```
где `t_предсказание(s)` — время выполнения, предсказанное DNN моделью для состояния `s`.

Положительная награда означает улучшение (уменьшение времени), отрицательная — ухудшение.

**Функция перехода:** Детерминированная (выбор действия однозначно определяет новое состояние).

**Цель:** Найти политику `π: S → A`, максимизирующую суммарную награду (минимизирующую время выполнения).

---

## 4.2. Реализация RL-оптимизаторов

### 4.2.1. Табличный Q-Learning (основной метод из статьи)

**Обоснование выбора:**
Q-Learning [17] — классический алгоритм обучения с подкреплением, использованный в статье Huang et al. [16] в качестве основного метода. Авторы внесли ключевое улучшение: вместо случайной инициализации каждого эпизода, агент начинает поиск с лучшего найденного состояния (`bestState`).

**Алгоритм Q-Learning:**

Q-функция оценивает "полезность" выполнения действия `a` в состоянии `s`:

```
Q(s, a) — ожидаемая суммарная награда при выборе действия a в состоянии s
```

**Формула обновления Q-значений:**
```
Q(s, a) ← Q(s, a) + α[r + γ · max_{a'} Q(s', a') - Q(s, a)]
```

где:
- `α` — learning rate (скорость обучения) = 0.1
- `γ` — discount factor (дисконтирование будущих наград) = 0.95
- `r` — полученная награда
- `s'` — новое состояние после действия `a`
- `max_{a'} Q(s', a')` — максимальное Q-значение в новом состоянии

**ε-greedy стратегия:**
```
a = {
    random_action           с вероятностью ε (exploration)
    argmax_a Q(s, a)        с вероятностью 1-ε (exploitation)
}
```

где `ε` уменьшается со временем: `ε_new = max(ε_min, ε · decay)`, начиная с 0.3 и снижаясь до 0.05.

**Ключевое улучшение из статьи [16]:**
```python
for episode in range(n_episodes):
    state = best_state  # Начинаем с лучшего найденного состояния
```

Вместо случайной инициализации (как в классическом Q-Learning), каждый эпизод начинается с `bestState` — конфигурации с минимальным предсказанным временем выполнения. Это предотвращает траты времени на исследование заведомо плохих областей пространства параметров.

**Этапы реализации:**
1. Инициализация Q-таблицы: `Q[s][a] = 0` для всех состояний и действий
2. Инициализация начального состояния (середина диапазона каждого параметра)
3. Цикл по эпизодам:
   - Установка `state = bestState`
   - Цикл по шагам (max 50 шагов на эпизод):
     - Выбор действия `a` по ε-greedy стратегии
     - Применение действия → получение нового состояния `s'`
     - Запрос предсказания времени у DNN модели
     - Вычисление награды `r`
     - Обновление Q-таблицы
     - Обновление `bestState` если найдена лучшая конфигурация
   - Уменьшение `ε`
4. Возврат лучшей найденной конфигурации

**Источник:** Sutton R.S., Barto A.G. Reinforcement Learning: An Introduction. 2nd ed. MIT Press, 2018 [17]

---

### 4.2.2. Deep Q-Network (DQN)

**Обоснование выбора:**
Табличный Q-Learning хранит Q-значения для каждой пары (состояние, действие) в таблице. При большом пространстве состояний (как у нас — 16 параметров с несколькими значениями каждый = миллионы комбинаций) таблица становится огромной и неэффективной. DQN [18] решает эту проблему, используя нейросеть для аппроксимации Q-функции.

**Ключевые отличия от табличного Q-Learning:**

1. **Q-функция как нейросеть:**
   - Вход: состояние `s` (16-мерный вектор параметров)
   - Выход: Q-значения для всех действий `[Q(s, a1), Q(s, a2), ..., Q(s, an)]`
   - Архитектура: `Input(16) → Dense(64) → Dense(32) → Output(n_actions)`

2. **Experience Replay:**
   Вместо обучения на каждом шаге сразу, DQN хранит опыт `(s, a, r, s')` в буфере и обучается на случайных батчах из прошлого опыта. Это:
   - Разбивает корреляции между последовательными примерами
   - Позволяет переиспользовать данные
   - Стабилизирует обучение

3. **Target Network:**
   Используются две копии сети:
   - **Policy Network** (обновляется каждый шаг) — выбирает действия
   - **Target Network** (обновляется редко) — вычисляет target Q-значения

   Это предотвращает "погоню за движущейся целью" при обучении.

**Формула обновления DQN:**
```
Loss = E[(r + γ · max_{a'} Q_target(s', a') - Q(s, a))²]
```

где `Q_target` — target network, обновляемая каждые 10 эпизодов.

**Преимущества перед табличным Q-Learning:**
- Масштабируется на большие пространства состояний
- Обобщает на похожие конфигурации (если две конфигурации похожи, их Q-значения тоже будут похожи)
- Требует меньше памяти при росте пространства параметров

**Источник:** Mnih V. et al. Human-level control through deep reinforcement learning // Nature. 2015. Vol. 518. P. 529–533 [18]

---

### 4.2.3. Proximal Policy Optimization (PPO)

**Обоснование выбора:**
Q-Learning и DQN — это value-based методы (оценивают полезность действий). PPO [19] — это policy-based метод (напрямую обучает стратегию выбора действий). PPO является одним из наиболее популярных современных RL алгоритмов благодаря стабильности и эффективности.

**Ключевая идея PPO:**
Вместо Q-функции, PPO обучает policy function `π(a|s)` — вероятностное распределение над действиями в состоянии `s`.

**Objective function (упрощённо):**
```
L(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

где:
- `r_t(θ) = π_new(a|s) / π_old(a|s)` — отношение вероятностей нового и старого policy
- `A_t` — advantage (насколько действие лучше среднего)
- `clip(...)` — ограничение изменения policy (предотвращает слишком большие обновления)

**Почему PPO лучше простого Policy Gradient:**
- Ограничивает размер обновления policy → более стабильное обучение
- Не требует сложных вычислений (как TRPO)
- Хорошо работает с continuous и discrete action spaces

**Реализация через stable-baselines3:**
Я использовал готовую реализацию PPO из библиотеки stable-baselines3 [20], которая оптимизирована и протестирована на множестве задач.

**Этапы реализации:**
1. Создание Gym-совместимой среды (environment):
   - `reset()` — возвращает начальное состояние (случайная конфигурация)
   - `step(action)` — применяет действие, возвращает `(new_state, reward, done, info)`
   - Reward = относительное улучшение времени, предсказанное DNN
2. Инициализация PPO агента с гиперпараметрами:
   - Learning rate: 0.0003
   - Batch size: 64
   - Number of epochs: 10
3. Обучение агента на `total_timesteps` шагов
4. Извлечение лучшей найденной конфигурации из истории

**Источник:** Schulman J. et al. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347, 2017 [19]

---

### 4.2.4. Bayesian Optimization через Optuna

**Обоснование выбора:**
В отличие от RL методов (которые обучаются через пробы и ошибки), Bayesian Optimization [21] — это класс методов глобальной оптимизации, строящих вероятностную модель зависимости "параметры → результат" и использующих её для умного выбора следующей точки для проверки.

**Когда Bayesian Optimization эффективен:**
- Малое количество итераций (каждая оценка "дорогая")
- Гладкая функция цели (без резких скачков)
- Непрерывное или смешанное пространство параметров

**Алгоритм TPE (Tree-structured Parzen Estimator):**
Optuna использует TPE [22] — эффективный вариант Bayesian Optimization:

1. Разделяет историю испытаний на "хорошие" и "плохие" результаты
2. Строит две плотности вероятности:
   - `p(x | y < y*)` — распределение параметров для хороших результатов
   - `p(x | y ≥ y*)` — распределение параметров для плохих результатов
3. Выбирает следующую точку, максимизируя `p(x | y < y*) / p(x | y ≥ y*)`

**Преимущества Bayesian Optimization:**
- Эффективен при малом бюджете итераций (50-200)
- Не требует настройки гиперпараметров (в отличие от RL)
- Хорошо работает с категориальными параметрами (snappy/lz4)

**Недостатки:**
- Не обучается online (не использует опыт внутри одного запуска)
- Менее эффективен при очень большом числе параметров (>50)

**Этапы реализации:**
1. Определение пространства поиска через Optuna API
2. Определение objective function (возвращает предсказанное DNN время)
3. Запуск оптимизации: `study.optimize(objective, n_trials=100)`
4. Извлечение лучших параметров: `study.best_params`

**Источник:** Akiba T. et al. Optuna: A Next-generation Hyperparameter Optimization Framework // Proceedings of the 25th ACM SIGKDD. 2019. P. 2623–2631 [22]

---

## 4.3. Интеграция с MLflow для трекинга экспериментов

Для систематического отслеживания экспериментов с различными RL алгоритмами я интегрировал систему MLflow [23]. Для каждого запуска оптимизатора логгируются:

**Параметры (Parameters):**
- Название алгоритма (Q-Learning, DQN, PPO, Bayesian)
- Гиперпараметры (α, γ, ε, количество эпизодов)
- Топология кластера (workers, cores, memory)

**Метрики (Metrics):**
- Лучшее найденное время выполнения
- Количество итераций до сходимости
- История наград (для RL методов)

**Артефакты (Artifacts):**
- Лучшая конфигурация (JSON)
- История поиска (CSV с all tried configurations)
- Графики обучения (reward vs iteration)

Это позволяет сравнивать методы по эффективности, скорости сходимости и качеству найденных решений.

---

## 4.4. Тестирование и валидация прототипа

### 4.4.1. Сравнительный анализ RL алгоритмов

Результаты прогона `out/rl_topo336_constrained/report.json` (WordCount, profile=large, topology `3×3×6`):

| Алгоритм | Предсказанное время (с) | Speedup vs default | Улучшение |
|---|---|---|---|
| Default config | 18.55 (pred) / 18.87 (real) | 1.00× | — |
| Q-Learning (улучшенный) | 15.84 | 1.17× | **+14.6%** |
| DQN | 15.85 | 1.17× | +14.6% |
| PPO | 15.88 | 1.17× | +14.4% |
| Bayesian (Optuna) | 15.87 | 1.17× | +14.5% |

### 4.4.2. Выводы

1. **Валидация подхода из статьи:** улучшенный Q-Learning [16] на нашем датасете даёт такое же качество, как DQN, PPO и Bayesian — разница в пределах <0.1 s, что укладывается в шум повторов. Подход из Sensors22 подтверждается.
2. **Сравнение методов:** все четыре оптимизатора сходятся к очень близким конфигурациям. В прод мы берём Q-Learning (проще и быстрее), Bayesian — как контроль.
3. **Практическая применимость:** прототип находит субоптимальную конфигурацию за сотни итераций (секунды работы оптимизатора) вместо полного перебора 10⁹ конфигураций.
4. **Ограничение текущего результата:** все 14% ускорения — это ускорение относительно **surrogate model**, обученной на WordCount-large. Реальная валидация на Spark показывает такое же порядково ускорение (см. `data/validation_QLearning.csv`), но окончательный вывод требует повтора на новом multi-workload датасете.

---

## ДОПОЛНИТЕЛЬНЫЕ ИСТОЧНИКИ

[17] Sutton R.S., Barto A.G. Reinforcement Learning: An Introduction. 2nd ed. MIT Press, 2018. 548 p. URL: http://incompleteideas.net/book/RLbook2020.pdf

[18] Mnih V. et al. Human-level control through deep reinforcement learning // Nature. 2015. Vol. 518. P. 529–533. URL: https://doi.org/10.1038/nature14236

[19] Schulman J. et al. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347, 2017. URL: https://arxiv.org/abs/1707.06347

[20] Raffin A., Hill A., Gleave A., Kanervisto A., Ernestus M., Dormann N. Stable-Baselines3: Reliable Reinforcement Learning Implementations // Journal of Machine Learning Research. 2021. Vol. 22. № 268. P. 1–8. URL: https://jmlr.org/papers/v22/20-1364.html

[21] Frazier P.I. A Tutorial on Bayesian Optimization. arXiv preprint arXiv:1807.02811, 2018. URL: https://arxiv.org/abs/1807.02811

[22] Akiba T., Sano S., Yanase T., Ohta T., Koyama M. Optuna: A Next-generation Hyperparameter Optimization Framework // Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019. P. 2623–2631. URL: https://doi.org/10.1145/3292500.3330701

[23] Chen A. et al. Developments in MLflow: A System to Accelerate the Machine Learning Lifecycle // Proceedings of the Fourth International Workshop on Data Management for End-to-End Machine Learning. 2020. Article 5. URL: https://doi.org/10.1145/3399579.3399867

---

## ИСТОЧНИКИ

[1] Géron A. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. 2nd ed. O'Reilly Media, 2019. 856 p. URL: https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

[2] Pedregosa F. et al. Scikit-learn: Machine Learning in Python // Journal of Machine Learning Research. 2011. Vol. 12. P. 2825–2830. URL: https://jmlr.org/papers/v12/pedregosa11a.html (тут можно документацию sicit learn)

[3] Ioffe S., Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift // Proceedings of the 32nd International Conference on Machine Learning. 2015. P. 448–456. URL: http://proceedings.mlr.press/v37/ioffe15.html

[4] Hancock J.T., Khoshgoftaar T.M. Survey on categorical data for neural networks // Journal of Big Data. 2020. Vol. 7. № 28. URL: https://doi.org/10.1186/s40537-020-00305-w

[5] Box G.E.P., Cox D.R. An Analysis of Transformations // Journal of the Royal Statistical Society. Series B. 1964. Vol. 26. № 2. P. 211–252. URL: https://www.jstor.org/stable/2984418 (убрать)

[6] Raschka S., Mirjalili V. Python Machine Learning. 3rd ed. Packt Publishing, 2019. 772 p. URL: https://www.packtpub.com/product/python-machine-learning-third-edition/9781789955750

[7] Breiman L. Random Forests // Machine Learning. 2001. Vol. 45. № 1. P. 5–32. URL: https://doi.org/10.1023/A:1010933404324

[8] Bergstra J., Bengio Y. Random Search for Hyper-Parameter Optimization // Journal of Machine Learning Research. 2012. Vol. 13. P. 281–305. URL: https://jmlr.org/papers/v13/bergstra12a.html

[9] Kirkpatrick S. et al. Optimization by Simulated Annealing // Science. 1983. Vol. 220. № 4598. P. 671–680. URL: https://doi.org/10.1126/science.220.4598.671

[10] Hornik K., Stinchcombe M., White H. Multilayer feedforward networks are universal approximators // Neural Networks. 1989. Vol. 2. № 5. P. 359–366. URL: https://doi.org/10.1016/0893-6080(89)90020-8 (смотрим другое)

[11] Nair V., Hinton G.E. Rectified Linear Units Improve Restricted Boltzmann Machines // Proceedings of the 27th International Conference on Machine Learning. 2010. P. 807–814. URL: https://icml.cc/Conferences/2010/papers/432.pdf

[12] Kingma D.P., Ba J. Adam: A Method for Stochastic Optimization // Proceedings of the 3rd International Conference on Learning Representations (ICLR). 2015. URL: https://arxiv.org/abs/1412.6980

[13] Paszke A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library // Advances in Neural Information Processing Systems 32. 2019. P. 8024–8035. URL: https://papers.nips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

[14] Smith L.N. Cyclical Learning Rates for Training Neural Networks // IEEE Winter Conference on Applications of Computer Vision (WACV). 2017. P. 464–472. URL: https://doi.org/10.1109/WACV.2017.58

[15] Chai T., Draxler R.R. Root mean square error (RMSE) or mean absolute error (MAE)? // Geoscientific Model Development. 2014. Vol. 7. P. 1247–1250. URL: https://doi.org/10.5194/gmd-7-1247-2014

[16] Huang X., Zhang H., Zhai X. A Novel Reinforcement Learning Approach for Spark Configuration Parameter Optimization // Sensors. 2022. Vol. 22. № 15. P. 5930. URL: https://doi.org/10.3390/s22155930
