import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. Параметры модели
# ============================================================
N            = 700
Lx, Ly       = 40.0, 80.0        # размеры сосуда
r_min, r_max = 0.50, 0.65        # радиусы дисков
rho          = 1.0
g            = 9.8
k_n          = 10000.0           # нормальная жёсткость
gamma_n      = 20.0              # нормальная вязкость (диссипация)
mu           = 0.3              # коэф. трения
gamma_t      = 10.0              # тангенциальная вязкость
dt           = 0.001             # шаг Верле

# --- Параметры остановки ---
T_max        = 100.0             # Максимальное время (защита от вечного цикла)
v_threshold  = 0.01              # Порог скорости (теперь 0.01)
settle_time  = 0.5               # Сколько секунд скорость должна держаться ниже порога
settle_steps = int(settle_time / dt)
extra_time   = 5.0               # Дополнительный запас времени (5 секунд)

save_every   = 100

rng    = np.random.default_rng()
radii  = rng.uniform(r_min, r_max, N)
masses = rho * np.pi * radii**2

_pi, _pj = np.triu_indices(N, k=1)
_rs      = radii[_pi] + radii[_pj]
_rs2     = _rs * _rs

# ============================================================
# 2. Начальное состояние
# ============================================================
pos = np.zeros((N, 2))
placed = 0
y_min_init = Ly * 0.4
y_max_init = Ly - r_max - 0.1
x_min_init = r_max + 0.1
x_max_init = Lx - r_max - 0.1

while placed < N:
    cand = np.array([rng.uniform(x_min_init, x_max_init),
                     rng.uniform(y_min_init, y_max_init)])
    if placed > 0:
        dx = pos[:placed, 0] - cand[0]
        dy = pos[:placed, 1] - cand[1]
        d2 = dx*dx + dy*dy
        min_dist = radii[:placed] + radii[placed] + 1e-4
        if np.any(d2 < min_dist**2):
            continue
    pos[placed] = cand
    placed += 1
vel = np.zeros_like(pos)

# ============================================================
# 3. Силы
# ============================================================
def compute_forces(pos, vel):
    dx = pos[_pj,0] - pos[_pi,0]
    dy = pos[_pj,1] - pos[_pi,1]
    d2 = dx*dx + dy*dy
    mask = d2 < _rs2
    ii, jj = _pi[mask], _pj[mask]
    dx, dy, d2, rs = dx[mask], dy[mask], d2[mask], _rs[mask]

    fx = np.zeros(N)
    fy = -masses * g

    if ii.size > 0:
        d     = np.sqrt(d2)
        nx,ny = dx/d, dy/d
        delta = rs - d
        dvx   = vel[jj,0] - vel[ii,0]
        dvy   = vel[jj,1] - vel[ii,1]
        vn    = dvx*nx + dvy*ny
        fn    = np.maximum(k_n*delta - gamma_n*vn, 0.)
        tx, ty = -ny, nx
        vt     = dvx*tx + dvy*ty
        ft     = np.clip(-gamma_t*vt, -mu*fn, mu*fn)
        Fx, Fy = fn*nx + ft*tx, fn*ny + ft*ty
        np.add.at(fx, ii, -Fx);  np.add.at(fy, ii, -Fy)
        np.add.at(fx, jj,  Fx);  np.add.at(fy, jj,  Fy)

    m = (pos[:,0] - radii) < 0
    if m.any():
        idx   = np.where(m)[0]
        delta = radii[idx] - pos[idx,0]
        fn    = np.maximum(k_n*delta - gamma_n*vel[idx,0], 0)
        fx[idx] += fn
    m = (pos[:,0] + radii) > Lx
    if m.any():
        idx   = np.where(m)[0]
        delta = pos[idx,0] + radii[idx] - Lx
        fn    = np.maximum(k_n*delta + gamma_n*vel[idx,0], 0)
        fx[idx] -= fn
    m = (pos[:,1] - radii) < 0
    if m.any():
        idx   = np.where(m)[0]
        delta = radii[idx] - pos[idx,1]
        fn    = np.maximum(k_n*delta - gamma_n*vel[idx,1], 0)
        fy[idx] += fn

    return fx, fy

# ============================================================
# 4. Velocity-Verlet с динамической остановкой (+5 сек)
# ============================================================
n_steps = int(T_max/dt)
frames, times, KE_arr, PE_arr = [], [], [], []

print("Идет моделирование, пожалуйста подождите...")
quiet_counter = 0
is_settled = False
settle_timestamp = 0.0

for step in range(n_steps):
    fx, fy = compute_forces(pos, vel)
    vel[:,0] += 0.5*dt*fx/masses
    vel[:,1] += 0.5*dt*fy/masses
    pos     += dt*vel
    fx, fy  = compute_forces(pos, vel)
    vel[:,0] += 0.5*dt*fx/masses
    vel[:,1] += 0.5*dt*fy/masses

    current_time = step * dt

    # --- Проверка на успокоение ---
    v_max = np.max(np.linalg.norm(vel, axis=1))
    if v_max < v_threshold:
        quiet_counter += 1
        if quiet_counter >= settle_steps and not is_settled:
            print(f"✅ Система успокоилась на t = {current_time:.2f} сек. Ждем еще {extra_time} сек для стабилизации...")
            is_settled = True
            settle_timestamp = current_time
    else:
        quiet_counter = 0

    # --- Сохранение кадров ---
    if step % save_every == 0:
        frames.append(pos.copy())
        times.append(current_time)
        KE_arr.append(0.5*np.sum(masses*(vel[:,0]**2 + vel[:,1]**2)))
        PE_arr.append(np.sum(masses*g*pos[:,1]))

    # --- Остановка, если истекло добавочное время ---
    if is_settled and (current_time >= settle_timestamp + extra_time):
        print(f"🛑 Моделирование завершено на t = {current_time:.2f} сек. (включая +5 сек запаса)")
        # Записываем финальный кадр
        frames.append(pos.copy())
        times.append(current_time)
        KE_arr.append(0.5*np.sum(masses*(vel[:,0]**2 + vel[:,1]**2)))
        PE_arr.append(np.sum(masses*g*pos[:,1]))
        break
else:
    print(f"⚠️ Достигнут лимит времени T_max = {T_max} сек. Система всё еще колеблется.")

# ============================================================
# 5. Визуализация
# ============================================================
fig = plt.figure(figsize=(12, 7), dpi=100)

ax = fig.add_axes([0.03, 0.05, 0.40, 0.90])
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.plot([0,Lx],[0,0],'k-',lw=2.5); ax.plot([0,0],[0,Ly],'k-',lw=2.5); ax.plot([Lx,Lx],[0,Ly],'k-',lw=2.5)

ax2 = fig.add_axes([0.52, 0.55, 0.44, 0.40])
ax2.plot(times, KE_arr, 'r-',  label='Кинетическая энергия')
ax2.plot(times, PE_arr, 'b-',  label='Потенциальная энергия')
ax2.plot(times, np.array(KE_arr)+np.array(PE_arr), 'k-', label='Полная энергия')
ax2.set_xlabel('Время (t)'); ax2.set_ylabel('Энергия')
ax2.legend(); ax2.grid(alpha=0.3)

ax3 = fig.add_axes([0.52, 0.05, 0.44, 0.40])
ax3.set_xlim(0, Lx); ax3.set_ylim(0, Ly)
ax3.set_aspect('equal'); ax3.set_xticks([]); ax3.set_yticks([])
ax3.plot([0,Lx],[0,0],'k-',lw=2); ax3.plot([0,0],[0,Ly],'k-',lw=2); ax3.plot([Lx,Lx],[0,Ly],'k-',lw=2)

# Отрисовка холста для расчета масштаба
fig.canvas.draw()

bbox         = ax.get_window_extent()
pts_per_data = bbox.width * 72 / fig.dpi / Lx
sc = ax.scatter(pos[:,0], pos[:,1], s=(2*radii*pts_per_data)**2,
                c='tan', edgecolors='k', linewidths=0.4, zorder=3)

bbox3         = ax3.get_window_extent()
pts_per_data3 = bbox3.width * 72 / fig.dpi / Lx
ax3.scatter(frames[-1][:,0], frames[-1][:,1], s=(2*radii*pts_per_data3)**2,
            c='tan', edgecolors='k', linewidths=0.4, zorder=3)
ax3.set_title(f"Конечная упаковка (t = {times[-1]:.2f})")

def animate(i):
    sc.set_offsets(frames[i])
    ax.set_title(f"Модель мягких дисков  —  t = {times[i]:.2f}")
    return sc,

ani = FuncAnimation(fig, animate, frames=len(frames),
                    interval=40, blit=False, repeat=True)
plt.show()