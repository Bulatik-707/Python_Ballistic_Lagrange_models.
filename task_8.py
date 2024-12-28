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
