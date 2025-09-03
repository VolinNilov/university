# Лабораторная работа 1

## Содержание
1. [Структура проекта]()

## Структура проекта
```
1_Lab
├── main.py              # Точка входа, запуск симуляций, сравнение, вывод графиков
├── utils/
│   ├── __init__.py      # Делает utils пакетом
│   ├── solvers.py       # Реализации ODESolver, EulerMethod, HeunMethod, RungeKutta4Method
│   ├── model.py         # Класс PendulumMotionModel с уравнениями движения
│   └── simulator.py     # Класс MotionSimulator для управления симуляциями и выводом
└── app/
    ├── __init__.py
    ├── model.py          # Модель MVC для UI
    ├── view.py           # View (Qt UI)
    ├── controller.py     # Controller
    └── resources.py      # Для хранения констант/стилей
```