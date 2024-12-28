import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры
m = 0.85  # масса снаряда, кг
Cd = 0.58  # коэффициент сопротивления
A = 0.01  # площадь поперечного сечения, м^2
rho = 1.225  # плотность воздуха, кг/м^3
g = 9.81  # ускорение свободного падения, м/с^2

# Начальные условия
v0 = 800  # начальная скорость, м/с
alpha = 45  # угол вылета, градусы
vx0 = v0 * np.cos(np.radians(alpha))
vy0 = v0 * np.sin(np.radians(alpha))
initial_conditions = [0, 0, vx0, vy0]

# Уравнения движения
def equations(t, y):
    x, y_pos, vx, vy = y
    v = np.sqrt(vx**2 + vy**2)
    dvxdt = -0.5 * Cd * A * rho * v * vx / m
    dvydt = -g - 0.5 * Cd * A * rho * v * vy / m
    return [vx, vy, dvxdt, dvydt]

# Решение
t_span = (0, 100)  # время в секундах
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

# График
plt.plot(sol.y[0], sol.y[1])
plt.title("Траектория полета снаряда")
plt.xlabel("Расстояние (м)")
plt.ylabel("Высота (м)")
plt.grid()
plt.show()
