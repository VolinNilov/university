# Результаты работы 

## Содержание
1. [Результаты экспериментов по обучению DQN](#результаты-экспериментов-по-обучению-dqn)
2. [Реализация Proximal Policy Optimization (PPO) для стабилизации маятника](#реализация-proximal-policy-optimization-ppo-для-стабилизации-маятника)


## Результаты экспериментов по обучению DQN

### Объясненнение значение параметров
| Параметр                     | Значение                                                                                                                                                                                                 |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `env_name`                   | Имя среды Gym, в которой обучается агент (например, `"CartPole-v1"`). Определяет задачу, которую нужно решить. Необходимо для загрузки конкретной среды обучения.                                      |
| `num_iterations`             | Общее количество итераций обучения. Каждая итерация включает сбор данных и обучение модели. Определяет длительность процесса обучения.                                                                 |
| `initial_collect_steps`      | Количество шагов, которые выполняются в начале для заполнения буфера воспроизведения. Это необходимо для обеспечения достаточного количества данных перед началом обучения.                          |
| `collect_steps_per_iteration`| Количество шагов, которые агент выполняет в среде на каждой итерации для сбора новых данных. Позволяет регулировать объем данных, добавляемых в буфер на каждом шаге.                                |
| `replay_buffer_capacity`     | Максимальный размер буфера воспроизведения. Хранит опыт (состояния, действия, награды) для обучения. Более крупный буфер улучшает стабильность обучения за счет большего разнообразия данных.        |
| `fc_layer_params`            | Архитектура полносвязных слоев нейронной сети (например, `(100,)`). Определяет сложность модели. Больше слоев или нейронов увеличивает выразительность модели, но требует больше ресурсов.          |
| `batch_size`                 | Размер мини-батча, используемого для обновления весов модели. На каждой итерации из буфера выбирается случайный мини-батч этого размера. Меньший батч быстрее, но менее стабилен.                  |
| `learning_rate`              | Скорость обучения оптимизатора. Определяет, насколько сильно модель корректирует свои веса на каждом шаге. Высокая скорость может привести к нестабильности, низкая — к медленному обучению.         |
| `gamma`                      | Коэффициент дисконтирования будущих наград. Значение `0.99` означает, что модель высоко ценит долгосрочные награды. Важен для задач с отложенным вознаграждением.                                   |
| `log_interval`               | Интервал (в итерациях), через который выводится информация о прогрессе обучения (например, потери, средняя награда). Позволяет отслеживать процесс обучения.                                         |
| `num_atoms`                  | Количество "атомов" (дискретных значений), на которые разбивается распределение Q-значений. Увеличивает точность аппроксимации распределения, но требует больше вычислений.                         |
| `min_q_value`                | Минимальное значение Q-функции, которое может быть представлено. Ограничивает диапазон возможных значений для стабильности обучения.                                                               |
| `max_q_value`                | Максимальное значение Q-функции, которое может быть представлено. Ограничивает диапазон возможных значений для стабильности обучения.                                                               |
| `num_eval_episodes`          | Количество эпизодов, которые используются для оценки производительности модели после обучения. Чем больше эпизодов, тем точнее оценка.                                                              |
| `eval_interval`              | Интервал (в итерациях), через который проводится оценка модели. Позволяет регулярно проверять прогресс обучения и принимать решения о завершении или продолжении.                                  |

### Первая попытка обучения

#### Параметры
``` python
env_name = "CartPole-v1" # @param {type:"string"}
num_iterations = 15000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (100,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
```

#### Результат
![Первый результат](https://github.com/VolinNilov/university/blob/main/MoAIiMR/3_lab_work/results/dqn/dqn_1.gif)

### Вторая попытка обучения

#### Параметры
``` python 
env_name = "CartPole-v1" # @param {type:"string"}
num_iterations = 15000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (100,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 100  # @param {type:"integer"}

num_atoms = 102  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 4  # @param {type:"integer"}

num_eval_episodes = 30  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
```

#### Результат
![Второй результат](https://github.com/VolinNilov/university/blob/main/MoAIiMR/3_lab_work/results/dqn/dqn_2.gif)


### Третья попытка обучения

#### Параметры
``` python
env_name = "CartPole-v1" # @param {type:"string"}
num_iterations = 12000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 50000  # @param {type:"integer"}

fc_layer_params = ((128, 64))

batch_size = 64  # @param {type:"integer"}
learning_rate = 5e-4  # @param {type:"number"}
gamma = 0.99
log_interval = 500  # @param {type:"integer"}

num_atoms = 102  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 3  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
```

#### Результат
![Третий результат](https://github.com/VolinNilov/university/blob/main/MoAIiMR/3_lab_work/results/dqn/dqn_3.gif)


## Реализация Proximal Policy Optimization (PPO) для стабилизации маятника 

Для стабилизации маятника (CartPole) можно использовать другой алгоритм обучения с подкреплением, мой выбор пал на Proximal Policy Optimization (PPO). Этот алгоритм является более современным и гибким по сравнению с DQN, особенно для задач с непрерывными действиями или сложной динамикой.

### PPO (Proximal Policy Optimization):
PPO — это policy-based метод, который работает как с дискретными, так и с непрерывными действиями.
Он стабилен, эффективен и широко используется в задачах обучения с подкреплением.
Для CartPole он подходит, потому что задача имеет дискретные действия (двигаться влево или вправо).


### Обучение PPO на 1000 итераций

#### Результат
![Результат 1000 итераций обучения](https://github.com/VolinNilov/university/blob/main/MoAIiMR/3_lab_work/results/ppo/ppo_1000.gif)

