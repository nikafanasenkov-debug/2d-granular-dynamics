import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ============================================================
# 1. Параметры (настроены на 300 КРУПНЫХ частиц)
# ============================================================
N1 = 0          
R1 = 0.50        

N2 = 300    
R2 = 1.00        

N = N1 + N2      

Lx, Ly       = 40.0, 80.0      
rho          = 1.0             
g            = 9.8               
k_n          = 10000.0          
gamma_n      = 20.0              
mu           = 0.30             
gamma_t      = 10.0              
dt           = 0.001            

# ВАЖНО: Ограничиваем время для GIF, чтобы файл не весил 500 МБ
T_max        = 15.0  
# Сохраняем реже, чтобы уменьшить количество кадров (каждый 50-й шаг)
save_every   = 50                

rng = np.random.default_rng(69) 

radii = np.concatenate([np.full(N1, R1), np.full(N2, R2)])
rng.shuffle(radii) 

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
y_max_init = Ly - np.max(radii) - 0.1
x_min_init = np.max(radii) + 0.1
x_max_init = Lx - np.max(radii) - 0.1

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
# 3. Функция вычисления сил
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

    # Стенки
    m = (pos[:,0] - radii) < 0
    if m.any():
        idx = np.where(m)[0]
        delta = radii[idx] - pos[idx,0]
        fn = np.maximum(k_n*delta - gamma_n*vel[idx,0], 0)
        fx[idx] += fn
    m = (pos[:,0] + radii) > Lx
    if m.any():
        idx = np.where(m)[0]
        delta = pos[idx,0] + radii[idx] - Lx
        fn = np.maximum(k_n*delta + gamma_n*vel[idx,0], 0)
        fx[idx] -= fn
    m = (pos[:,1] - radii) < 0
    if m.any():
        idx = np.where(m)[0]
        delta = radii[idx] - pos[idx,1]
        fn = np.maximum(k_n*delta - gamma_n*vel[idx,1], 0)
        fy[idx] += fn

    return fx, fy

# ============================================================
# 4. Моделирование (только запись кадров)
# ============================================================
n_steps = int(T_max/dt)
frames = []

print("Идет моделирование для GIF (15 секунд), пожалуйста подождите...")
for step in range(n_steps):
    fx, fy = compute_forces(pos, vel)
    
    vel[:,0] += 0.5*dt*fx/masses
    vel[:,1] += 0.5*dt*fy/masses
    pos     += dt*vel
    
    fx, fy  = compute_forces(pos, vel)
    vel[:,0] += 0.5*dt*fx/masses
    vel[:,1] += 0.5*dt*fy/masses

    if step % save_every == 0:
        frames.append(pos.copy())

# ============================================================
# 5. Сохранение GIF
# ============================================================
fig, ax = plt.subplots(figsize=(6, 10), dpi=80) # Уменьшили dpi для веса
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.plot([0,Lx],[0,0],'k-',lw=3)
ax.plot([0,0],[0,Ly],'k-',lw=3)
ax.plot([Lx,Lx],[0,Ly],'k-',lw=3)

fig.canvas.draw()
bbox = ax.get_window_extent()
pts_per_data = bbox.width * 72 / fig.dpi / Lx
sizes = (2*radii*pts_per_data)**2

sc = ax.scatter(frames[0][:,0], frames[0][:,1], s=sizes,
                c='tan', edgecolors='k', linewidths=0.5, zorder=3)

def animate(i):
    sc.set_offsets(frames[i])
    return sc,

ani = FuncAnimation(fig, animate, frames=len(frames), interval=40, blit=True)

if not os.path.exists('images'):
    os.makedirs('images')

save_path = 'images/exp_0_300.gif'
print(f"\nСохраняю анимацию в файл: {save_path}")
print("Это займет около 10-20 секунд...")

try:
    # Используем Pillow для сохранения
    ani.save(save_path, writer='pillow', fps=25)
    print("✅ Готово! GIF успешно создан и сохранен.")
except Exception as e:
    print(f"❌ Ошибка при сохранении: {e}")
    print(">>> ВАЖНО: Установите библиотеку Pillow, введя в терминал: pip install pillow")

plt.close(fig)