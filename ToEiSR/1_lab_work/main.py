import flet as ft
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import io
import base64

def calculate_response(R1, R2, R3, C1, L1):
    """
    Формирование модели в пространстве состояний и расчет переходной характеристики
    Параметры:
        R1, R2, R3 - сопротивления (Ом)
        C1 - емкость (Ф)
        L1 - индуктивность (Гн)
    Возвращает:
        T - массив временных отсчетов
        Y - массив значений выходного напряжения
    """
    
    # Формирование матриц пространства состояний

    # Уравнения состояния:
    # [dx1/dt] = [ -1/(R1*C1)  -(R2/R1*C1 + 1/C1) ]
    # [dx2/dt]   [   -1/L1        -(R2+R3)/L1     ]
    A = np.array([
        [-1/(R1*C1), - (R2/(R1*C1) + 1/C1)],
        [-1/L1, - (R2 + R3)/L1]              
    ])

    # Матрица входных воздействий B
    # [  1/(R1*C1) ]
    # [    1/L1    ]
    B = np.array([
        [1/(R1*C1)],
        [1/L1]
    ])
    
    C = np.array([[0, R3]])  # Выход: напряжение на R3 (U2 = x2*R3)
    D = np.array([[0]])      # Прямой канал
    
    sys = signal.StateSpace(A, B, C, D)
    
    T, Y = signal.step(sys)  
    
    return T, Y.flatten()

def on_calculate(e, R1, R2, R3, C1, L1, tau_text, sigma_text):
    try:
        R1 = float(R1.value)
        R2 = float(R2.value)
        R3 = float(R3.value)
        C1 = float(C1.value)
        L1 = float(L1.value)
        
        t, y = calculate_response(R1, R2, R3, C1, L1)
        y_inf = y[-1]
        
        eps = 0.05 * abs(y_inf)
        tau = next((t[i] for i in range(len(y)-1, -1, -1) if abs(y[i]-y_inf) >= eps), 0)
        
        y_max = max(y)
        sigma = (y_max - y_inf)/y_inf * 100
        
        tau_text.value = f"Время переходного процесса τ = {tau:.4f} сек"
        sigma_text.value = f"Перерегулирование σ = {sigma:.4f} %"
        
    except Exception as ex:
        tau_text.value = "Ошибка"
        sigma_text.value = "Ошибка"
    e.page.update()

def on_plot(e, R1, R2, R3, C1, L1, img_control):
    try:
        R1 = float(R1.value)
        R2 = float(R2.value)
        R3 = float(R3.value)
        C1 = float(C1.value)
        L1 = float(L1.value)
        
        t, y = calculate_response(R1, R2, R3, C1, L1)
        
        fig, ax = plt.subplots()
        ax.plot(t, y, label="Переходная характеристика")
        ax.set_xlabel("Время, сек")
        ax.set_ylabel("Напряжение U2, В")
        ax.set_title("График переходного процесса")
        ax.grid(True)
        ax.legend()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_control.src_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        e.page.update()
        
    except Exception as ex:
        print("Ошибка построения графика:", ex)
        img_control.src_base64 = ""
        e.page.update()

def main(page: ft.Page):
    page.title = "Расчёт цифровой модели динамической системы"
    page.window.center()
    page.window.resizable = False
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.height = 840
    page.window.width = 510
    
    system_image = ft.Image(src="electric_shem_img.png")

    inputs = [
        ft.TextField(label="R1 (Ом)", value="10", width=150),
        ft.TextField(label="R2 (Ом)", value="20", width=150),
        ft.TextField(label="R3 (Ом)", value="30", width=150),
        ft.TextField(label="C1 (Фараты)", value="0.001", width=150),
        ft.TextField(label="L1 (Генри)", value="0.1", width=150)
    ]
    
    results = [
        ft.Text("Время переходного процесса τ = ..."),
        ft.Text("Перерегулирование σ = ...")
    ]
    
    img_control = ft.Image(width=500, height=300)
    
    page.add(
        ft.Column([
            system_image,
            ft.Row(inputs[:3]), 
            ft.Row(inputs[3:]),
            ft.Row([
                ft.ElevatedButton(
                    "Рассчитать",
                    icon=ft.icons.CALCULATE,
                    on_click=lambda e: on_calculate(e, *inputs, *results)
                ),
                ft.ElevatedButton(
                    "Построить график",
                    icon=ft.icons.SHOW_CHART,
                    on_click=lambda e: on_plot(e, *inputs, img_control)
                )
            ]),
            *results,
            img_control
        ])
    )

ft.app(target=main)