# Python_Ballistic_Lagrange_models.
Расчет внутрибаллистического процесса выстрела с использованием модели Лагранжа. Расчет внешебаллистической траектории полета снаряда


ЛАБОРАТОРНАЯ РАБОТА № 3.
Расчет внутрибаллистического процесса выстрела с использованием модели Лагранжа. Расчет внешебаллистической траектории полета снаряда

1. Цель работы

Научиться рассчитывать параметры выстрела из пушки в приближении Лагранжа и рассчитывать оптимальную траекторию полета снаряда.

3. Задание на лабораторную работу

4.	Решить задачу Лагранжа численно, приняв  , и сравнить с аналитическим решением.

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

 
5.	Построить зависимости давления на дно канала и дно снаряда, а также скорости снаряда от времени выстрела.

import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
P0 = 1e5  # Начальное давление, Па
T0 = 2000  # Начальная температура, К
R = 287  # Газовая постоянная, J/(kg·K)
V = 0.0005  # Объем, м^3
m = 0.85  # Масса снаряда, кг
A = 0.005  # Площадь поперечного сечения, м^2 (примерно 5 см^2)
k = 1000  # Коэффициент сгорания

# Время
t = np.linspace(0, 0.01, 100)  # Время от 0 до 0.01 с

# Давление в канале (предположим, давление уменьшается с течением времени)
P = P0 * np.exp(-k * t)  # Давление в канале

# Масса газов (предположим, что она пропорциональна времени)
mass_gas = (P0 * V) / (R * T0)  # Масса газов при начальных условиях

# Скорость снаряда (изменяется по времени)
# F = P * A - mg
# a = F/m = (P * A - mg) / m
# v(t) = v(0) + ∫a dt
velocity = np.zeros_like(t)
for i in range(1, len(t)):
    a = (P[i] * A - m * 9.81) / m  # Ускорение
    velocity[i] = velocity[i-1] + a * (t[i] - t[i-1])  # Интегрирование для получения скорости

# Построение графиков
plt.figure(figsize=(12, 6))

# График давления
plt.subplot(1, 2, 1)
plt.plot(t, P, color='blue')
plt.title('Давление в канале от времени')
plt.xlabel('Время (с)')
plt.ylabel('Давление (Па)')
plt.grid()

# График скорости
plt.subplot(1, 2, 2)
plt.plot(t, velocity, color='red')
plt.title('Скорость снаряда от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.grid()

plt.tight_layout()
plt.show()

 
6.	Построить зависимости плотности и температуры пороховых газов от времени выстрела.
import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
T0 = 2000  # Начальная температура, К
rho0 = 1  # Начальная плотность, кг/м^3
k = 100  # Коэффициент охлаждения, s^-1
R = 287  # Газовая постоянная для воздуха, J/(kg·K)

# Время
t = np.linspace(0, 0.01, 100)  # Время от 0 до 0.01 с

# Температура
T = T0 * np.exp(-k * t)

# Давление (предположим, что оно остается постоянным на короткий период)
P = 1e5  # Давление, Па (например, атмосферное давление)

# Плотность газа по уравнению состояния
rho = P / (R * T)

# Построение графиков
plt.figure(figsize=(12, 6))

# График температуры
plt.subplot(1, 2, 1)
plt.plot(t, T, color='red')
plt.title('Температура пороховых газов от времени')
plt.xlabel('Время (с)')
plt.ylabel('Температура (К)')
plt.grid()

# График плотности
plt.subplot(1, 2, 2)
plt.plot(t, rho, color='blue')
plt.title('Плотность пороховых газов от времени')
plt.xlabel('Время (с)')
plt.ylabel('Плотность (кг/м^3)')
plt.grid()

plt.tight_layout()
plt.show()

 
7.	Решить задачу внешне баллистики, приняв  , и сравнить с аналитическим решением при   и  .

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

 
Сравнение численных и аналитических решений:
 Метод  Дальность (м)
Численный   46274.996214
Аналитический   65239.551478

8.	Построить траекторию полета снаряда при различных заданных значениях  . Исследовать влияние угла вылета снаряда на дальность стрельбы. Принять  , с шагом  . Построить таблицу и график.

Табл. 2. Параметры артиллерийской системы 
№	Тип
орудия	 ,
дм	 ,
дм2	 ,
кг	 ,
кг	 ,
дм3	 ,
дм	 ,
дм	 ,
кг	 

4	85-мм
ЗП - 1944	0,85	0,58	3,08	9,20	3,87	6,64	46,5	9,20	1,0


	Теплофизические характеристики выстрела приведены в табл. 3
Табл. 3. Теплофизические характеристики выстрела
№	Марка
пороха	 ,
МДж/кг	 ,
Дж/кг/К	 ,
Дж/кг/К	 ,
Дж/кг/К	 
 ,
дм3/кг	 ,
кг/дм3
4	14/7 ВА	1,013	345,1	1885,7	1540,6	0,224	1,010	1,60


import numpy as np
import matplotlib.pyplot as plt

# Параметры
g = 9.81  # Ускорение свободного падения, м/с²
C_d = 0.5  # Коэффициент сопротивления
rho = 1.225  # Плотность воздуха, кг/м³
A = 0.01  # Площадь поперечного сечения, м² (примерно)
m = 1.0  # Масса снаряда, кг

# Функция для вычисления силы сопротивления
def drag_force(v):
    return 0.5 * C_d * rho * A * v**2

# Метод Рунге-Кутты 4-го порядка
def runge_kutta(v0, theta, dt, total_time):
    theta_rad = np.radians(theta)  # Преобразование угла в радианы
    v0_x = v0 * np.cos(theta_rad)  # Начальная скорость по x
    v0_y = v0 * np.sin(theta_rad)  # Начальная скорость по y

    # Инициализация переменных
    t = 0
    x, y = 0, 0
    vx, vy = v0_x, v0_y

    trajectory = [(x, y)]  # Список для хранения координат

    while y >= 0:  # Пока снаряд не приземлится
        # Расчет величин для Рунге-Кутты
        v = np.sqrt(vx**2 + vy**2)
        F_d = drag_force(v)

        # Определение значений k1, k2, k3, k4
        k1vx = -F_d/m * (vx/v) * dt
        k1vy = -g - (F_d/m * (vy/v)) * dt
        k1x = vx * dt
        k1y = vy * dt

        k2vx = -F_d/m * ((vx + k1vx/2)/v) * dt
        k2vy = -g - (F_d/m * ((vy + k1vy/2)/v)) * dt
        k2x = (vx + k1vx/2) * dt
        k2y = (vy + k1vy/2) * dt

        k3vx = -F_d/m * ((vx + k2vx/2)/v) * dt
        k3vy = -g - (F_d/m * ((vy + k2vy/2)/v)) * dt
        k3x = (vx + k2vx/2) * dt
        k3y = (vy + k2vy/2) * dt

        k4vx = -F_d/m * ((vx + k3vx)/v) * dt
        k4vy = -g - (F_d/m * ((vy + k3vy)/v)) * dt
        k4x = (vx + k3vx) * dt
        k4y = (vy + k3vy) * dt

        # Обновление значений
        vx += (k1vx + 2*k2vx + 2*k3vx + k4vx) / 6
        vy += (k1vy + 2*k2vy + 2*k3vy + k4vy) / 6
        x += (k1x + 2*k2x + 2*k3x + k4x) / 6
        y += (k1y + 2*k2y + 2*k3y + k4y) / 6

        trajectory.append((x, y))
        t += dt

    return trajectory

# Параметры запуска
v0 = 800  # Начальная скорость, м/с
theta_values = np.arange(10, 71, 5)  # Углы от 10 до 70 с шагом 5
results = []

# Главный цикл для разных углов
for theta in theta_values:
    trajectory = runge_kutta(v0, theta, 0.01, 10)  # dt = 0.01, total_time = 10
    max_distance = max([point[0] for point in trajectory])  # Максимальная дальность
    results.append((theta, max_distance))

# Вывод результатов
print("Угол (градусы) | Дальность (м)")
for result in results:
    print(f"{result[0]:<16} | {result[1]:.2f}")

# График
angles, distances = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(angles, distances, marker='o')
plt.title('Влияние угла вылета на дальность стрельбы')
plt.xlabel('Угол вылета (градусы)')
plt.ylabel('Дальность полета (м)')
plt.grid()
plt.show()

 

