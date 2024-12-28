import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Параметры
g = 9.81  # Ускорение свободного падения, м/с^2
m = 0.85  # Масса снаряда, кг
v0 = 800  # Начальная скорость, м/с
theta_c0 = 45  # Угол вылета, градусы

# Функция для расчета траектории
def equations(t, y):
    x, y_pos, vx, vy = y
    v = np.sqrt(vx**2 + vy**2)
    # Сопротивление воздуха учтено
    dvxdt = -0.5 * Cx * A * rho * v * vx / m
    dvydt = -g - 0.5 * Cx * A * rho * v * vy / m
    return [vx, vy, dvxdt, dvydt]

# Угол в радианах
theta_rad = np.radians(theta_c0)
vx0 = v0 * np.cos(theta_rad)
vy0 = v0 * np.sin(theta_rad)
initial_conditions = [0, 0, vx0, vy0]

# Сопротивление воздуха
Cx = 10**-3  # Коэффициент сопротивления
A = 0.01  # Площадь поперечного сечения, м^2
rho = 1.225  # Плотность воздуха, кг/м^3

# Решение уравнений с учетом сопротивления
t_span = (0, 100)  # Время в секундах
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

# Максимальное расстояние
numerical_distance = max(sol.y[0])

# Аналитическое решение
analytical_distance = (v0**2 * np.sin(2 * theta_rad)) / g

# Сравнение результатов
results_df = pd.DataFrame({
    'Метод': ['Численный', 'Аналитический'],
    'Дальность (м)': [numerical_distance, analytical_distance]
})

print("\nСравнение численных и аналитических решений:")
print(results_df)
