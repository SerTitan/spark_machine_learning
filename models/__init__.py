"""
Spark Configuration Optimization Models

Модули:
- data: загрузка и препроцессинг данных
- baseline: baseline модели (Dummy, RF, SA, MLP)
- dnn_predictor: DNN для предсказания времени выполнения
- rl_optimizer: RL-агенты для поиска оптимальных параметров
"""

from .data import (
    load_dataset,
    create_dataset,
    SparkDataset,
    TOPOLOGY_FEATURES,
    SPARK_PARAMETERS,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    NUMERIC_FEATURES,
)

from .baseline import (
    DummyBaseline,
    RandomForestBaseline,
    SimulatedAnnealingRF,
    MLPBaseline,
    LinearRegressionBaseline,
    SVRBaseline,
    DecisionTreeBaseline,
    ExtraTreesBaseline,
    ModelResult,
    train_all_baselines,
    results_to_dataframe,
    compute_metrics,
)

from .dnn_predictor import (
    DNNPredictor,
    DNNConfig,
    TorchDNN,
    TrainingHistory,
    create_dnn_predictor,
    TORCH_AVAILABLE,
)

from .rl_optimizer import (
    TabularQLearning,
    DQNOptimizer,
    BayesianOptimizer,
    StableBaselinesOptimizer,
    OptimizationResult,
    run_all_optimizers,
)
