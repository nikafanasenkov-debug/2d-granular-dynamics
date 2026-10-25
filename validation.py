import numpy as np
import matplotlib.pyplot as plt

Lx, Ly = 40.0, 80.0
rho = 1.0
k_n = 10000.0
dt = 0.001

def compute_forces(pos, vel, radii, masses, gamma_n, gamma_t, mu, g):
    N = len(pos)
    fx = np.zeros(N)
    fy = -masses * g
    if N >= 2:
        _pi, _pj = np.triu_indices(N, k=1)
        dx = pos[_pj,0] - pos[_pi,0]
        dy = pos[_pj,1] - pos[_pi,1]
        d2 = dx*dx + dy*dy
        _rs = radii[_pi] + radii[_pj]
        _rs2 = _rs * _rs
        mask = d2 < _rs2
        ii, jj = _pi[mask], _pj[mask]
        dx, dy, d2, rs = dx[mask], dy[mask], d2[mask], _rs[mask]

        if ii.size > 0:
            d = np.sqrt(d2)
            nx,ny = dx/d, dy/d
            delta = rs - d
            dvx, dvy = vel[jj,0] - vel[ii,0], vel[jj,1] - vel[ii,1]
            vn = dvx*nx + dvy*ny
            fn = np.maximum(k_n*delta - gamma_n*vn, 0.)
            tx, ty = -ny, nx
            vt = dvx*tx + dvy*ty
            ft = np.clip(-gamma_t*vt, -mu*fn, mu*fn)
            Fx, Fy = fn*nx + ft*tx, fn*ny + ft*ty
            np.add.at(fx, ii, -Fx);  np.add.at(fy, ii, -Fy)
            np.add.at(fx, jj,  Fx);  np.add.at(fy, jj,  Fy)

    m = (pos[:,0] - radii) < 0
    if m.any():
        idx = np.where(m)[0]
        fx[idx] += np.maximum(k_n*(radii[idx] - pos[idx,0]) - gamma_n*vel[idx,0], 0)
    m = (pos[:,0] + radii) > Lx
    if m.any():
        idx = np.where(m)[0]
        fx[idx] -= np.maximum(k_n*(pos[idx,0] + radii[idx] - Lx) + gamma_n*vel[idx,0], 0)
    m = (pos[:,1] - radii) < 0
    if m.any():
        idx = np.where(m)[0]
        fy[idx] += np.maximum(k_n*(radii[idx] - pos[idx,1]) - gamma_n*vel[idx,1], 0)
    return fx, fy

def calculate_packing_fraction_contour(pos, radii, Lx, num_bins=10000):
    area_particles = np.sum(np.pi * radii**2)
    dx = pos[:,0][:,None] - pos[:,0][None,:]
    dy = pos[:,1][:,None] - pos[:,1][None,:]
    dist = np.sqrt(dx**2 + dy**2)
    r1 = radii[:, None]
    r2 = radii[None, :]
    
    mask = (dist < r1 + r2) & (dist > np.abs(r1 - r2)) & (dist > 1e-6)
    safe_dist = np.where(mask, dist, 1.0) 
    
    part1 = r1**2 * np.arccos(np.clip((safe_dist**2 + r1**2 - r2**2) / (2 * safe_dist * r1), -1.0, 1.0))
    part2 = r2**2 * np.arccos(np.clip((safe_dist**2 + r2**2 - r1**2) / (2 * safe_dist * r2), -1.0, 1.0))
    part3 = 0.5 * np.sqrt(np.maximum(0, (-safe_dist + r1 + r2) * (safe_dist + r1 - r2) * (safe_dist - r1 + r2) * (safe_dist + r1 + r2)))
    
    area_ij = part1 + part2 - part3
    area_overlap = 0.5 * np.sum(np.where(mask, area_ij, 0.0))
    area_effective = area_particles - area_overlap

    x_grid = np.linspace(0, Lx, num_bins)
    y_top = np.zeros_like(x_grid)
    
    for i, x in enumerate(x_grid):
        dx_centers = np.abs(pos[:, 0] - x)
        mask_contour = dx_centers < radii
        if np.any(mask_contour):
            y_vals = pos[mask_contour, 1] + np.sqrt(radii[mask_contour]**2 - dx_centers[mask_contour]**2)
            y_top[i] = np.max(y_vals)
        else:
            y_top[i] = 0.0
            
    area_layer = np.sum(y_top) * (Lx / num_bins)
    return area_effective / area_layer if area_layer > 0 else 0.0

def run_simulation_with_history(pos_init, radii, g_test, T_max, gamma_n=20.0, gamma_t=10.0, mu=0.3, vel_init=None):
    N = len(pos_init)
    pos = np.copy(pos_init)
    vel = np.zeros_like(pos) if vel_init is None else np.copy(vel_init)
    masses = rho * np.pi * radii**2
    
    n_steps = int(T_max / dt)
    time_hist, pos_hist, vel_hist = [], [], []
    
    for step in range(n_steps):
        fx, fy = compute_forces(pos, vel, radii, masses, gamma_n, gamma_t, mu, g_test)
        vel[:,0] += 0.5*dt*fx/masses
        vel[:,1] += 0.5*dt*fy/masses
        pos     += dt*vel
        fx, fy  = compute_forces(pos, vel, radii, masses, gamma_n, gamma_t, mu, g_test)
        vel[:,0] += 0.5*dt*fx/masses
        vel[:,1] += 0.5*dt*fy/masses
        
        if step % 100 == 0: 
            time_hist.append(step * dt)
            pos_hist.append(pos.copy())
            vel_hist.append(vel.copy())
            
    return np.array(time_hist), np.array(pos_hist), np.array(vel_hist)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Валидация физической модели гранулированной среды', fontsize=16, fontweight='bold')

radii1 = np.array([1.0])
pos1 = np.array([[Lx/2, Ly/2]])
t1, p1, v1 = run_simulation_with_history(pos1, radii1, g_test=9.8, T_max=15.0)

axs[0, 0].plot(t1, p1[:, 0, 1], 'b-', lw=2, label='Высота центра ')
axs[0, 0].axhline(y=1.0, color='r', linestyle='--', label='ожидаемый уровень центра диска')
axs[0, 0].set_title('Тест 1: Затухание отскока частицы', fontweight='bold')
axs[0, 0].set_xlabel('Время, с')
axs[0, 0].set_ylabel('Координата Y')
axs[0, 0].legend()
axs[0, 0].set_ylim(0, Ly/2 + 2)

radii2 = np.array([1.0, 1.0])
pos2 = np.array([[Lx/2, Ly/2], [Lx/2, Ly/2 + 2.5]])
t2, p2, v2 = run_simulation_with_history(pos2, radii2, g_test=9.8, T_max=50.0) # 50 секунд для полной остановки
pos2_final = p2[-1]

phi_test2 = calculate_packing_fraction_contour(pos2_final, radii2, Lx)
print(f"--> ТЕСТ 2: Коэффициент заполнения для двух дисков phi = {phi_test2:.4f}")

circle1 = plt.Circle((pos2_final[0,0], pos2_final[0,1]), radii2[0], color='tan', ec='k')
circle2 = plt.Circle((pos2_final[1,0], pos2_final[1,1]), radii2[1], color='tan', ec='k')
axs[0, 1].add_patch(circle1)
axs[0, 1].add_patch(circle2)

x_grid = np.linspace(0, Lx, 1000)
y_top = np.zeros_like(x_grid)
for i, x in enumerate(x_grid):
    dx_c = np.abs(pos2_final[:, 0] - x)
    mask = dx_c < radii2
    if np.any(mask):
        y_vals = pos2_final[mask, 1] + np.sqrt(radii2[mask]**2 - dx_c[mask]**2)
        y_top[i] = np.max(y_vals)

axs[0, 1].plot(x_grid, y_top, 'r-', lw=2, label='Контур слоя')

axs[0, 1].set_title(f'Тест 2: Контур упаковки (phi = {phi_test2:.3f})', fontweight='bold')
axs[0, 1].set_xlim(15, 25); axs[0, 1].set_ylim(-0.5, 5)
axs[0, 1].set_aspect('equal')
axs[0, 1].set_xlabel('Координата X'); axs[0, 1].set_ylabel('Координата Y')
axs[0, 1].legend(loc='upper right')

radii3 = np.array([0.5])
pos3 = np.array([[Lx/2, Ly/2]])
vel3 = np.array([[0.0, 500.0]])
t3, p3, v3 = run_simulation_with_history(pos3, radii3, g_test=0.0, T_max=0.4, vel_init=vel3)

axs[1, 0].plot(t3, p3[:, 0, 1], 'm-', lw=2)
axs[1, 0].axhline(y=Ly, color='k', linestyle='--', label='Верхняя граница сосуда')
axs[1, 0].axhline(y=Ly+5, color='r', linestyle=':', label='Порог срабатывания валидатора')
axs[1, 0].set_title('Тест 3: Отлов вылета за верхнюю границу', fontweight='bold')
axs[1, 0].set_xlabel('Время, с')
axs[1, 0].set_ylabel('Координата Y')
axs[1, 0].legend()

radii4 = np.full(10, 1.0)
pos4 = np.zeros((10, 2))
idx = 0
for i in range(5):
    for j in range(2):
        pos4[idx] = [10.0 + i*1.9, 10.0 + j*1.9]
        idx += 1
np.random.seed(42)
vel4 = np.random.uniform(-2, 2, (10, 2))

t4, p4, v4 = run_simulation_with_history(pos4, radii4, g_test=0.0, T_max=2.0, gamma_n=0.0, gamma_t=0.0, mu=0.0, vel_init=vel4)

masses4 = rho * np.pi * radii4**2
KE_arr, PE_arr, E_total = [], [], []
_pi, _pj = np.triu_indices(10, k=1)
rs4 = radii4[_pi] + radii4[_pj]

for i in range(len(t4)):
    v_curr = v4[i]
    p_curr = p4[i]
    KE = 0.5 * np.sum(masses4 * (v_curr[:,0]**2 + v_curr[:,1]**2))
    dist = np.linalg.norm(p_curr[_pi] - p_curr[_pj], axis=1)
    delta = np.maximum(0, rs4 - dist)
    PE = 0.5 * k_n * np.sum(delta**2)
    KE_arr.append(KE)
    PE_arr.append(PE)
    E_total.append(KE + PE)

axs[1, 1].plot(t4, KE_arr, 'r-', label='Кинетическая энергия')
axs[1, 1].plot(t4, PE_arr, 'b-', label='Потенциальная энергия')
axs[1, 1].plot(t4, E_total, 'k--', lw=2, label='Полная энергия (константа)')
axs[1, 1].set_title('Тест 4: Сохранение энергии (без диссипации)', fontweight='bold')
axs[1, 1].set_xlabel('Время, с')
axs[1, 1].set_ylabel('Энергия')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('validation_results.png', dpi=300)
plt.show()
