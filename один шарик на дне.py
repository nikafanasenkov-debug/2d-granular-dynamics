import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ТЕСТ НА ПРАВОМЕРНОСТЬ: Шарик лежит на дне
# ============================================================
# Параметры
Lx, Ly = 40.0, 80.0
R = 1.0               # Радиус шарика
rho = 1.0
m = rho * np.pi * R**2
g = 9.8
k_n = 10000.0         # Жесткость
gamma_n = 20.0        # Вязкость (нужна, чтобы мгновенно погасить микро-колебания)
dt = 0.001
T_total = 5.0         # Моделируем 5 секунд

# ------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ: Кладем шарик ровно на дно
# ------------------------------------------------------------
# Координата y = R (шарик едва касается пола, перекрытия нет)
pos = np.array([[Lx/2.0, R]])
# Начальная скорость СТРОГО равна нулю
vel = np.array([[0.0, 0.0]])

n_steps = int(T_total / dt)
times = []
KE_arr = []
PE_arr = []
Y_arr = []

# ------------------------------------------------------------
# ФИЗИКА (упрощенная для 1 частицы)
# ------------------------------------------------------------
for step in range(n_steps):
    # 1. Считаем силы
    # Гравитация тянет вниз
    fy = -m * g
    fx = 0.0
    
    # Контакт с дном (стенка на y=0)
    if pos[0, 1] - R < 0:
        delta = R - pos[0, 1]              # Величина проникновения
        fn = np.maximum(k_n * delta - gamma_n * vel[0, 1], 0.0)
        fy += fn

    # 2. Интегратор Верле (Velocity-Verlet)
    vel[0, 0] += 0.5 * dt * fx / m
    vel[0, 1] += 0.5 * dt * fy / m
    
    pos += dt * vel
    
    # Пересчитываем силы в новых координатах
    fy = -m * g
    if pos[0, 1] - R < 0:
        delta = R - pos[0, 1]
        fn = np.maximum(k_n * delta - gamma_n * vel[0, 1], 0.0)
        fy += fn
        
    vel[0, 0] += 0.5 * dt * fx / m
    vel[0, 1] += 0.5 * dt * fy / m

    # 3. Сохраняем энергии
    if step % 10 == 0:  # Сохраняем каждые 10 шагов
        times.append(step * dt)
        Y_arr.append(pos[0, 1])
        KE = 0.5 * m * (vel[0, 0]**2 + vel[0, 1]**2)
        PE = m * g * pos[0, 1]
        KE_arr.append(KE)
        PE_arr.append(PE)

# ------------------------------------------------------------
# РЕЗУЛЬТАТЫ
# ------------------------------------------------------------
times = np.array(times)
KE_arr = np.array(KE_arr)
PE_arr = np.array(PE_arr)
Total_E = KE_arr + PE_arr

print(f"--- РЕЗУЛЬТАТЫ ТЕСТА ---")
print(f"Начальная высота центра: {Y_arr[0]:.6f}")
print(f"Конечная высота центра:  {Y_arr[-1]:.6f}")
print(f"Изменение высоты:        {abs(Y_arr[-1] - Y_arr[0]):.2e} (практически ноль)")
print(f"Кинетическая энергия (макс): {np.max(KE_arr):.6e}")
print(f"Разброс полной энергии:  {np.max(Total_E) - np.min(Total_E):.2e}")

# ------------------------------------------------------------
# ------------------------------------------------------------
# ГРАФИК
# ------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))

# --- Левая панель: сам шарик ---
ax1 = fig.add_axes([0.05, 0.1, 0.35, 0.8])
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, Ly * 0.2) 
ax1.set_aspect('equal')
ax1.plot([0, Lx], [0, 0], 'k-', lw=3)
ax1.plot([0, 0], [0, Ly*0.2], 'k-', lw=3)
ax1.plot([Lx, Lx], [0, Ly*0.2], 'k-', lw=3)
ax1.scatter(pos[0, 0], pos[0, 1], s=500, c='tan', edgecolors='k', zorder=3)
ax1.set_title("Шарик лежит на дне")

# --- Правая панель: главный график энергии ---
ax2 = fig.add_axes([0.5, 0.1, 0.45, 0.8])

# 1. Потенциальная энергия (широкая, полупрозрачная)
ax2.plot(times, PE_arr, color='#1f77b4', linewidth=6, alpha=0.4, label='Потенциальная энергия')

# 2. Полная энергия (тонкая, пунктирная, черная)
ax2.plot(times, Total_E, color='black', linewidth=2, linestyle='--', label='Полная энергия')

# 3. Кинетическая энергия (красная, на нуле)
ax2.plot(times, KE_arr, color='red', linewidth=2, label='Кинетическая энергия (≈ 0)')

ax2.set_xlabel('Время (t)')
ax2.set_ylabel('Энергия')
ax2.set_ylim(-2, max(PE_arr) * 1.05)
ax2.legend(loc='center right')
ax2.grid(alpha=0.3)
ax2.set_title("Сохранение энергии в покое")



# --- Добавляем Inset (увеличенное окошко с первыми 0.5 секундами) ---
ax_inset = ax2.inset_axes([0.15, 0.5, 0.35, 0.4]) # [x, y, width, height] в долях от ax2
ax_inset.plot(times, Total_E, color='black', linewidth=1.5)
ax_inset.set_xlim(0, 0.5)
ax_inset.set_ylim(30.5, 30.85) # Приближаем масштаб
ax_inset.set_title("Увеличение: первые 0.5 сек", fontsize=9)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(alpha=0.3)

plt.show()