import numpy as np

from utils.solvers import EulerMethod, HeunMethod, RungeKutta4Method
from utils.model import PendulumMotionModel
from utils.simulator import MotionSimulator


def main():    
    # 1. Определяем модель движения (пример из лекции)
    model = PendulumMotionModel(radius=1.0, gravity=9.81, friction_coeff=0.2, mass=1.0)

    # 2. Задаем начальные условия (пример из лекции стр. 9)
    R = model.R
    angle_rad = 0.1 # Начальный угол в радианах
    x0 = R * np.sin(angle_rad)
    y0 = -R * np.cos(angle_rad) # Отрицательный, так как ось y направлена вниз в примере
    vx0 = 0.0
    vy0 = 0.0
    initial_conditions = [x0, y0, vx0, vy0]

    # 3. Задаем параметры интегрирования
    t_span = (0, 5) # Время от 0 до 5 секунд
    dt_coarse = 0.05   # Грубый шаг для видимой разницы
    dt_fine = 0.005    # Мелкий шаг для эталона

    # 4. Создаем решатели
    euler_solver = EulerMethod()
    heun_solver = HeunMethod()
    rk4_solver = RungeKutta4Method()

    # 5. Создаем симулятор
    simulator = MotionSimulator(model, output_dir_name="data")

    # 6. Запускаем симуляции с разными методами и шагами
    print("Запуск симуляций...")
    
    # Методы Эйлера и Хьюна с грубым шагом (основное сравнение)
    simulator.run_simulation(euler_solver, initial_conditions, t_span, dt_coarse, f'Euler (dt={dt_coarse})')
    simulator.run_simulation(heun_solver, initial_conditions, t_span, dt_coarse, f'Heun (dt={dt_coarse})')
    
    # Метод Рунге-Кутты 4-го порядка с грубым шагом (дополнительно)
    simulator.run_simulation(rk4_solver, initial_conditions, t_span, dt_coarse, f'RK4 (dt={dt_coarse})')

    # Метод Рунге-Кутты 4-го порядка с мелким шагом как эталон
    simulator.run_simulation(rk4_solver, initial_conditions, t_span, dt_fine, f'RK4_ref (dt={dt_fine})')

    # 7. Визуализируем результаты (все требуемые графики ЛР1)
    print("\nПостроение графиков ЛР1...")
    simulator.plot_lab1_graphs(filename_prefix="lab1_results")

    # 8. Сохраняем все результаты в CSV
    print("\nСохранение данных в CSV...")
    simulator.save_results_to_csv()

    # 9. Сравниваем результаты Эйлера и Хьюна с эталоном
    print("\nСравнение решений Эйлера и Хьюна с эталоном (RK4 с мелким шагом):")
    simulator.print_comparison(
        'RK4_ref (dt=0.005)',
        'Euler (dt=0.05)',
        'Heun (dt=0.05)'
    )
    
    # 10. Сравниваем RK4 с грубым и мелким шагом
    print("\nСравнение RK4 с разными шагами:")
    simulator.print_comparison(
        'RK4_ref (dt=0.005)',
        'RK4 (dt=0.05)'
    )


if __name__ == "__main__":
    main()
