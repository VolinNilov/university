import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

class ExperimentProcessor:
    def __init__(self, output_dir: str = "results"):
        """
        Инициализация процессора эксперимента.
        Создает директорию для сохранения результатов.
        """
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_experiment_data(self, factors: list, levels: dict, repetitions: int = 3) -> pd.DataFrame:
        """
        Генерирует данные эксперимента на основе планировочной матрицы.
        Декодирует факторы в натуральные значения и добавляет случайную погрешность.
        Возвращает DataFrame с данными эксперимента.
        """

        matrix = self._create_planning_matrix(factors, levels)
        data = []
        for _, row in matrix.iterrows():
            natural_factors = {}
            for factor in factors:
                X0 = levels[factor][1]  # Основной уровень
                n = (levels[factor][0] - levels[factor][2]) / 2  # Интервал варьирования
                coded_value = row[factor]
                natural_value = X0 + coded_value * n
                natural_factors[factor] = natural_value
            true_y = self.calculate_true_response(**natural_factors)
            scale = 0.01 * true_y
            noisy_y = np.random.normal(loc=true_y, scale=scale, size=repetitions)
            for y in noisy_y:
                data.append({**row.to_dict(), "y": y})
        return pd.DataFrame(data)
    
    def calculate_true_response(self, R1: float, R2: float, R3: float, C1: float, L1: float) -> float:
        """
        Рассчитывает истинное значение отклика на основе модели электрической цепи.
        Возвращает время переходного процесса в миллисекундах.
        """

        R_parallel = (R1 * R2) / (R1 + R2)
        R_total = R_parallel + R3
        return (L1 / R_total) * 1e3
    
    def _create_planning_matrix(self, factors: list, levels: dict) -> pd.DataFrame:
        """
        Создает планировочную матрицу полного факторного эксперимента.
        Добавляет взаимодействия между факторами.
        Возвращает DataFrame с кодированными значениями факторов.
        """

        combinations = list(product([-1, 1], repeat=len(factors)))
        matrix = pd.DataFrame(combinations, columns=factors)
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                interaction = f"{factors[i]}*{factors[j]}"
                matrix[interaction] = matrix[factors[i]] * matrix[factors[j]]
        return matrix
    
    def process_data(self, data: pd.DataFrame, alpha: float = 0.05) -> dict:
        """
        Обрабатывает данные эксперимента:
        - Проверяет однородность дисперсий (критерий Кохрена)
        - Рассчитывает коэффициенты регрессии
        - Проверяет адекватность модели
        Возвращает словарь с результатами обработки.
        """

        results = {}
        grouped = data.groupby(list(data.columns[:-1]))
        means = grouped.mean().reset_index()
        variances = grouped.var().reset_index()
        results["means"] = means
        results["variances"] = variances
        
        Gp = variances['y'].max() / variances['y'].sum()
        f1 = len(variances) - 1
        f2 = len(data['y'].unique()) - 1
        Gt = self._get_cochran_critical_value(alpha, f1, f2)
        results["homogeneity"] = Gp < Gt
        
        if results["homogeneity"]:
            coefficients = self._calculate_coefficients(means)
            results["coefficients_coded"] = coefficients
            results["coefficients_natural"] = self.decode_coefficients(coefficients, levels)
        
        y_pred = self._predict(means, coefficients)
        s_adeq = np.sum((means['y'] - y_pred)**2) / (len(means) - len(coefficients))
        Ft = self._get_f_critical_value(alpha, len(means)-len(coefficients), len(data)-1)
        Fp = s_adeq / variances['y'].mean()
        results["adequacy"] = {
            "F_adeq": Fp,
            "is_adequate": Fp < Ft
        }
        return results
    
    def decode_coefficients(self, coefficients: dict, levels: dict) -> dict:
        """
        Декодирует коэффициенты регрессии из кодированного вида в натуральный.
        Возвращает словарь с декодированными коэффициентами.
        """

        decoded = {}
        for key, value in coefficients.items():
            if '*' in key:
                factors = key.split('*')
                term = value
                for f in factors:
                    X0 = levels[f][1]
                    n = (levels[f][0] - levels[f][2]) / 2
                    term *= (n / 2) ** len(factors)
                decoded[key] = term
            else:
                if key in levels:
                    X0 = levels[key][1]
                    n = (levels[key][0] - levels[key][2]) / 2
                    decoded[key] = value * (n / 2)
                else:
                    decoded[key] = value
        return decoded
    
    def compare_predictions(self, coefficients: dict, levels: dict, num_samples: int = 5) -> pd.DataFrame:
        """
        Сравнивает предсказания модели с истинными значениями на случайных выборках.
        Возвращает DataFrame со сравнительным анализом.
        """

        samples = []
        for _ in range(num_samples):
            factors = {}
            for param in ['R1', 'R2', 'R3', 'C1', 'L1']:
                low, mid, high = levels[param]
                factors[param] = np.random.uniform(low, high)
            
            coded = {}
            for param in ['R1', 'R2', 'R3', 'C1', 'L1']:
                X0 = levels[param][1]
                n = (levels[param][0] - levels[param][2]) / 2
                coded[param] = (factors[param] - X0) / n
            
            y_pred = coefficients['b0']
            for key, value in coefficients.items():
                if key == 'b0':
                    continue
                factors_in_term = key.split('*')
                term = 1
                for f in factors_in_term:
                    term *= coded.get(f, 0)
                y_pred += value * term
            
            y_true = self.calculate_true_response(**factors)
            samples.append({
                "Factors": factors,
                "Predicted": y_pred,
                "True": y_true,
                "Error": abs(y_pred - y_true)
            })
        return pd.DataFrame(samples)
    
    def save_results_to_csv(self, results: dict, filename: str):
        """
        Сохраняет результаты эксперимента в CSV файлы:
        - Сводные данные
        - Средние значения
        - Дисперсии
        - Кодированные коэффициенты
        - Натуральные коэффициенты
        """

        base_path = os.path.join(self.output_dir, filename)
        with open(f"{base_path}_summary.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Параметр", "Значение"])
            writer.writerow(["Однородность дисперсий", results.get("homogeneity", "N/A")])
            writer.writerow(["Адекватность модели", results.get("adequacy", {}).get("is_adequate", "N/A")])
        
        for key in ["means", "variances"]:
            if key in results:
                results[key].to_csv(f"{base_path}_{key}.csv", index=False)
        
        if "coefficients_coded" in results:
            pd.DataFrame(results["coefficients_coded"].items(), columns=["Фактор", "Кодированный коэффициент"])\
                .to_csv(f"{base_path}_coeff_coded.csv", index=False)
        
        if "coefficients_natural" in results:
            pd.DataFrame(results["coefficients_natural"].items(), columns=["Фактор", "Натуральный коэффициент"])\
                .to_csv(f"{base_path}_coeff_natural.csv", index=False)
    
    def plot_results(self, data: pd.DataFrame, filename: str):
        """
        Создает графическое представление результатов эксперимента.
        Сохраняет график в формате PNG.
        """

        plt.figure(figsize=(10, 6))
        for factor in data.columns[:-1]:
            plt.scatter(data[factor], data["y"], label=factor)
        plt.title("Зависимость функции отклика от факторов")
        plt.xlabel("Факторы")
        plt.ylabel("Функция отклика")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close()
    
    def _calculate_coefficients(self, means: pd.DataFrame) -> dict:
        """
        Рассчитывает коэффициенты регрессии по методу наименьших квадратов.
        Возвращает словарь с коэффициентами.
        """

        N = len(means)
        coefficients = {}
        for col in means.columns:
            if col == 'y':
                coefficients['b0'] = (means[col] * means[col]).sum() / N
            else:
                coefficients[col] = (means[col] * means['y']).sum() / N
        return coefficients
    
    def _predict(self, data: pd.DataFrame, coefficients: dict) -> np.ndarray:
        """
        Выполняет предсказание отклика на основе рассчитанных коэффициентов.
        Возвращает массив предсказанных значений.
        """

        predictions = np.zeros(len(data))
        for i, row in data.iterrows():
            pred = coefficients['b0']
            for key, value in coefficients.items():
                if key == 'b0':
                    continue
                factors_in_term = key.split('*')
                term = 1
                for f in factors_in_term:
                    term *= row.get(f, 0)
                pred += value * term
            predictions[i] = pred
        return predictions
    
    def _get_cochran_critical_value(self, alpha: float, f1: int, f2: int) -> float:
        """
        Возвращает критическое значение для критерия Кохрена.
        Если точное значение не найдено, возвращает значение по умолчанию.
        """

        cochran_table = {
            (7, 2): 0.8376
        }
        return cochran_table.get((f1, f2), 0.7)
    
    def _get_f_critical_value(self, alpha: float, f1: int, f2: int) -> float:
        """
        Возвращает критическое значение для F-критерия.
        Если точное значение не найдено, возвращает значение по умолчанию.
        """

        f_table = {
            (4, 22): 3.01,
            (7, 22): 2.51
        }
        return f_table.get((f1, f2), 4.0)

# Параметры эксперимента
factors = ["R1", "R2", "R3", "C1", "L1"]
levels = {
    "R1": [60, 50, 40],   # Верхний, основной, нижний
    "R2": [30, 25, 20],
    "R3": [100, 80, 60],
    "C1": [0.0015, 0.001, 0.0005],
    "L1": [0.15, 0.1, 0.05]
}

# Создание и выполнение эксперимента
processor = ExperimentProcessor()
experiment_data = processor.generate_experiment_data(factors, levels, repetitions=3)
results = processor.process_data(experiment_data)

# Сравнение оценок и истинных значений
comparison = processor.compare_predictions(results["coefficients_coded"], levels)
comparison.to_csv(os.path.join(processor.output_dir, "comparison.csv"), index=False)

# Сохранение результатов
processor.save_results_to_csv(results, "experiment_results")
processor.plot_results(experiment_data, "experiment_plot")