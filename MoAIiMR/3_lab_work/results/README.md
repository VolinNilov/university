# Результаты экспериментов по обучению DQN

## Первая попытка обучения

### Параметры
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

### Результат
![Первый результат](https://github.com/VolinNilov/university/blob/main/MoAIiMR/3_lab_work/results/dqn/dqn_1.gif)

## Вторая попытка обучения

### Параметры
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

### Результат
![Второй результат](https://github.com/VolinNilov/university/blob/main/MoAIiMR/3_lab_work/results/dqn/dqn_2.gif)


## Третья попытка обучения

### Параметры
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

### Результат
![Третий результат](https://github.com/VolinNilov/university/blob/main/MoAIiMR/3_lab_work/results/dqn/dqn_3.gif)
