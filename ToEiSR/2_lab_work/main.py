import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import product
from scipy import stats
from sklearn.linear_model import LinearRegression

class FullFactorialExperiment:
    """
    Класс для проведения лабораторной работы №2:
    "Построение и исследование модели с использованием полного факторного эксперимента".
    Выполняет построение матрицы ПФЭ, моделирование отклика и построение регрессионной модели,
    а также рассчитывает статистические критерии и сохраняет результаты в файлы.
    """

    def __init__(self, R1_levels: list[float], C1_levels: list[float], L1_levels: list[float], R3: float = 1000, U1: float = 1.0, freq: float = 1000):
        """
        Инициализация параметров эксперимента.
        """
        self.R1_levels = R1_levels
        self.C1_levels = C1_levels
        self.L1_levels = L1_levels
        self.R3 = R3
        self.U1 = U1
        self.freq = freq
        self.df = None
        self.model = None
        os.makedirs("results", exist_ok=True)

    # Этап 1: Генерация матрицы ПФЭ
    def generate_design_matrix(self) -> None:
        """Генерирует матрицу полного факторного эксперимента 2^3."""
        combinations = list(product(*[[-1, 1]] * 3))
        df = pd.DataFrame(combinations, columns=["X1", "X2", "X3"])
        df["R1"] = df["X1"].map({-1: self.R1_levels[0], 1: self.R1_levels[1]})
        df["C1"] = df["X2"].map({-1: self.C1_levels[0], 1: self.C1_levels[1]})
        df["L1"] = df["X3"].map({-1: self.L1_levels[0], 1: self.L1_levels[1]})
        self.df = df

    # Этап 2: Моделирование отклика
    def compute_response(self, R1: float, C1: float, L1: float) -> float:
        """Вычисляет амплитуду отклика на R3 при заданных элементах схемы."""
        w = 2 * np.pi * self.freq
        Zc = 1 / (1j * w * C1)
        Zr = R1
        Zrc_parallel = 1 / (1/Zc + 1/Zr)
        ZL = 1j * w * L1
        Z_total = Zrc_parallel + ZL + self.R3
        U2 = (self.R3 / Z_total) * self.U1
        return abs(U2)

    def simulate(self) -> None:
        """Моделирует отклик Y для всех комбинаций факторов."""
        if self.df is None:
            raise ValueError("Матрица ПФЭ не сгенерирована. Вызовите generate_design_matrix().")
        self.df["Y"] = self.df.apply(lambda row: self.compute_response(row["R1"], row["C1"], row["L1"]), axis=1)

    # Этап 3: Построение регрессионной модели
    def build_regression_model(self) -> tuple[np.ndarray, float]:
        """Строит регрессионную модель. Возвращает коэффициенты и свободный член."""
        X = self.df[["X1", "X2", "X3"]]
        y = self.df["Y"]
        self.model = LinearRegression()
        self.model.fit(X, y)
        return self.model.coef_, self.model.intercept_

    # Этап 4: Анализ значимости коэффициентов
    def analyze_significance(self) -> tuple[np.ndarray, list[float]]:
        """Вычисляет t-статистики и p-значения для коэффициентов регрессии."""
        X = self.df[["X1", "X2", "X3"]].values
        y = self.df["Y"].values
        n, k = X.shape
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - k - 1)
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * mse)
        t_stats = model.coef_ / se
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - k - 1)) for t in t_stats]
        return t_stats, p_values

    # Этап 5: Проверка адекватности модели
    def f_test(self) -> float:
        """Проверка адекватности модели по критерию Фишера. Возвращает F-статистику."""
        X = self.df[["X1", "X2", "X3"]].values
        y = self.df["Y"].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        ssr = np.sum((y_pred - np.mean(y))**2)
        sse = np.sum((y - y_pred)**2)
        dfr = X.shape[1]
        dfe = len(y) - dfr - 1
        msr = ssr / dfr
        mse = sse / dfe
        F = msr / mse
        return F

    # Этап 6: Сохранение результатов
    def save_results_to_csv(self, filename: str = "results/experiment_results_with_explanations.csv") -> None:
        """Сохраняет результаты с пояснениями в CSV."""
        coef, intercept = self.build_regression_model()
        t_stats, p_vals = self.analyze_significance()
        f_val = self.f_test()

        results = {
            "Описание": [
                "Коэффициент X1", "Коэффициент X2", "Коэффициент X3", "Свободный член",
                "t-статистика X1", "t-статистика X2", "t-статистика X3", 
                "p-значение X1", "p-значение X2", "p-значение X3",
                "F-статистика"
            ],
            "Значение": [
                coef[0], coef[1], coef[2], intercept,
                t_stats[0], t_stats[1], t_stats[2],
                p_vals[0], p_vals[1], p_vals[2],
                f_val
            ]
        }

        df_results = pd.DataFrame(results)
        df_results.to_csv(filename, index=False)

    def save_design_matrix_to_csv(self, filename: str = "results/design_matrix_and_responses.csv") -> None:
        """Сохраняет матрицу ПФЭ и отклики в CSV."""
        self.df.to_csv(filename, index=False)

    # Этап 7: Визуализация результатов
    def plot_results(self) -> None:
        """Строит и сохраняет графики влияния факторов."""
        for x in ["X1", "X2", "X3"]:
            plt.figure()
            self.df.groupby(x)["Y"].mean().plot(kind="bar", title=f"Влияние {x} на Y")
            plt.ylabel("Отклик Y")
            plt.xlabel(x)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/{x}_influence.png")
            plt.close()

    # Дополнительный этап: Сравнение истинных и предсказанных значений
    def compare_true_and_predicted(self, random_combinations: list[tuple[float, float, float]]) -> pd.DataFrame:
        """Сравнивает истинные и предсказанные значения для случайных комбинаций факторов."""
        results = []
        for R1, C1, L1 in random_combinations:
            true_value = self.compute_response(R1, C1, L1)
            predicted_value = self.model.predict([[R1, C1, L1]])[0]
            results.append({"R1": R1, "C1": C1, "L1": L1, "True_Y": true_value, "Predicted_Y": predicted_value})
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Задание уровней факторов
    R1_levels = [100, 1000]
    C1_levels = [1e-6, 10e-6]
    L1_levels = [10e-3, 100e-3]

    # Создание экземпляра класса
    exp = FullFactorialExperiment(R1_levels, C1_levels, L1_levels)

    # Этап 1: Генерация матрицы ПФЭ
    exp.generate_design_matrix()

    # Этап 2: Моделирование отклика
    exp.simulate()

    # Этап 3: Построение регрессионной модели
    coefficients, intercept = exp.build_regression_model()
    print("Коэффициенты модели:", coefficients)
    print("Свободный член:", intercept)

    # Этап 4: Анализ значимости коэффициентов
    t_stats, p_values = exp.analyze_significance()
    print("t-статистики:", t_stats)
    print("p-значения:", p_values)

    # Этап 5: Проверка адекватности модели
    f_stat = exp.f_test()
    print("F-статистика:", f_stat)

    # Этап 6: Сохранение результатов
    exp.save_results_to_csv()
    exp.save_design_matrix_to_csv()

    # Этап 7: Визуализация результатов
    exp.plot_results()

    # Дополнительный этап: Сравнение истинных и предсказанных значений
    random_combinations = [
        (np.random.uniform(*R1_levels), np.random.uniform(*C1_levels), np.random.uniform(*L1_levels))
        for _ in range(5)
    ]
    comparison_df = exp.compare_true_and_predicted(random_combinations)
    print("\nСравнение истинных и предсказанных значений:")
    print(comparison_df)