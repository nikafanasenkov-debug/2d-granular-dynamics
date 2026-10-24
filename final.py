import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N1 =  0       
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

T_max             = 100.0        
rel_change_thresh = 0.001        
quiet_steps_needed = 200        

save_every   = 20                

rng = np.random.default_rng(69) 

radii = np.concatenate([np.full(N1, R1), np.full(N2, R2)])
rng.shuffle(radii) 

masses = rho * np.pi * radii**2

_pi, _pj = np.triu_indices(N, k=1)
_rs      = radii[_pi] + radii[_pj]
_rs2     = _rs * _rs

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
    # Правая стенка
    m = (pos[:,0] + radii) > Lx
    if m.any():
        idx   = np.where(m)[0]
        delta = pos[idx,0] + radii[idx] - Lx
        fn    = np.maximum(k_n*delta + gamma_n*vel[idx,0], 0)
        fx[idx] -= fn
    # Дно
    m = (pos[:,1] - radii) < 0
    if m.any():
        idx   = np.where(m)[0]
        delta = radii[idx] - pos[idx,1]
        fn    = np.maximum(k_n*delta - gamma_n*vel[idx,1], 0)
        fy[idx] += fn

    return fx, fy

n_steps = int(T_max/dt)
frames, times, KE_arr, PE_arr = [], [], [], []

print("Идет моделирование, пожалуйста подождите...")
quiet_counter = 0
prev_v_max = 0.0
relaxation_time = T_max

for step in range(n_steps):
    fx, fy = compute_forces(pos, vel)
    vel[:,0] += 0.5*dt*fx/masses
    vel[:,1] += 0.5*dt*fy/masses
    pos     += dt*vel
    fx, fy  = compute_forces(pos, vel)
    vel[:,0] += 0.5*dt*fx/masses
    vel[:,1] += 0.5*dt*fy/masses

    current_time = step * dt
    v_max = np.max(np.linalg.norm(vel, axis=1))

    if v_max < 0.1: 
        if step > 0 and prev_v_max > 1e-6:
            rel_change = abs(v_max - prev_v_max) / prev_v_max
        else:
            rel_change = 0.0
        if rel_change < rel_change_thresh:
            quiet_counter += 1
        else:
            quiet_counter = 0
    else:
        quiet_counter = 0
    prev_v_max = v_max

    if quiet_counter >= quiet_steps_needed:
        relaxation_time = current_time
        print(f"✅ Система успокоилась! Время релаксации: t = {relaxation_time:.2f} сек.")
        frames.append(pos.copy())
        times.append(current_time)
        KE_arr.append(0.5*np.sum(masses*(vel[:,0]**2 + vel[:,1]**2)))
        PE_arr.append(np.sum(masses*g*pos[:,1]))
        break

    if step % save_every == 0:
        frames.append(pos.copy())
        times.append(current_time)
        KE_arr.append(0.5*np.sum(masses*(vel[:,0]**2 + vel[:,1]**2)))
        PE_arr.append(np.sum(masses*g*pos[:,1]))
else:
    print(f"⚠️ Достигнут лимит T_max = {T_max} сек.")

def calculate_packing_fraction_contour(pos, radii, Lx, num_bins=1000):
    area_particles = np.sum(np.pi * radii**2)
    dx = pos[:,0][:,None] - pos[:,0][None,:]
    dy = pos[:,1][:,None] - pos[:,1][None,:]
    dist = np.sqrt(dx**2 + dy**2)
    r1 = radii[:, None]
    r2 = radii[None, :]
    
    mask = (dist < r1 + r2) & (dist > np.abs(r1 - r2)) & (dist > 1e-6)
    
    safe_dist = np.where(mask, dist, 1.0) 
    
    # Аналитическая формула площади пересечения двух кругов
    part1 = r1**2 * np.arccos(np.clip((safe_dist**2 + r1**2 - r2**2) / (2 * safe_dist * r1), -1.0, 1.0))
    part2 = r2**2 * np.arccos(np.clip((safe_dist**2 + r2**2 - r1**2) / (2 * safe_dist * r2), -1.0, 1.0))
    part3 = 0.5 * np.sqrt(np.maximum(0, (-safe_dist + r1 + r2) * 
                                     (safe_dist + r1 - r2) * 
                                     (safe_dist - r1 + r2) * 
                                     (safe_dist + r1 + r2)))
    
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

def validate_result(pos, vel, radii, Lx, Ly):
    errors = []
    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)): errors.append("NaN/Inf в координатах")
    if np.min(pos[:,0] - radii) < -0.1: errors.append("Выход за левую стенку")
    if np.max(pos[:,0] + radii) > Lx + 0.1: errors.append("Выход за правую стенку")
    if np.min(pos[:,1] - radii) < -0.1: errors.append("Выход за дно")
    if np.max(pos[:,1] - Ly) > 5.0: errors.append("Вылет за верхнюю границу сосуда")
    return errors

packing_fraction = calculate_packing_fraction_contour(pos, radii, Lx)
validation_errors = validate_result(pos, vel, radii, Lx, Ly)

print(f"\n--- РЕЗУЛЬТАТЫ ---")
print(f"Время релаксации: {relaxation_time:.2f} сек")
print(f"Коэффициент заполнения: {packing_fraction:.4f} ({packing_fraction*100:.1f}%)")

fig = plt.figure(figsize=(16, 7), dpi=100)
mixture_info = f"Состав:\n{N1} шт. × R={R1}\n{N2} шт. × R={R2}"

ax = fig.add_axes([0.02, 0.05, 0.28, 0.85])
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.plot([0,Lx],[0,0],'k-',lw=2.5); ax.plot([0,0],[0,Ly],'k-',lw=2.5); ax.plot([Lx,Lx],[0,Ly],'k-',lw=2.5)

info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                    va='top', ha='left', fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

ax3 = fig.add_axes([0.33, 0.05, 0.28, 0.85])
ax3.set_xlim(0, Lx); ax3.set_ylim(0, Ly)
ax3.set_aspect('equal'); ax3.set_xticks([]); ax3.set_yticks([])
ax3.plot([0,Lx],[0,0],'k-',lw=2); ax3.plot([0,0],[0,Ly],'k-',lw=2); ax3.plot([Lx,Lx],[0,Ly],'k-',lw=2)

final_text_str = (f"Время релаксации: {relaxation_time:.2f} c\n"
                  f"Коэф. заполнения: {packing_fraction:.3f}\n"
                  f"{mixture_info}")
ax3.text(0.02, 0.98, final_text_str, transform=ax3.transAxes, 
         va='top', ha='left', fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

ax2 = fig.add_axes([0.65, 0.20, 0.33, 0.65]) 
ax2.plot(times, KE_arr, 'r-',  label='Кинетическая энергия')
ax2.plot(times, PE_arr, 'b-',  label='Потенциальная энергия')
ax2.plot(times, np.array(KE_arr)+np.array(PE_arr), 'k-', label='Полная энергия')
ax2.set_xlabel('Время (t)'); ax2.set_ylabel('Энергия')
ax2.legend(); ax2.grid(alpha=0.3)

fig.canvas.draw()

bbox         = ax.get_window_extent()
pts_per_data = bbox.width * 72 / fig.dpi / Lx
sc = ax.scatter(pos[:,0], pos[:,1], s=(2*radii*pts_per_data)**2,
                c='tan', edgecolors='k', linewidths=0.4, zorder=3)

bbox3         = ax3.get_window_extent()
pts_per_data3 = bbox3.width * 72 / fig.dpi / Lx
ax3.scatter(frames[-1][:,0], frames[-1][:,1], s=(2*radii*pts_per_data3)**2,
            c='tan', edgecolors='k', linewidths=0.4, zorder=3)

x_grid = np.linspace(0, Lx, 500)
y_contour = np.zeros_like(x_grid)
for i, x in enumerate(x_grid):
    dx_centers = np.abs(pos[:, 0] - x)
    mask = dx_centers < radii
    if np.any(mask):
        y_vals = pos[mask, 1] + np.sqrt(radii[mask]**2 - dx_centers[mask]**2)
        y_contour[i] = np.max(y_vals)
ax3.plot(x_grid, y_contour, 'r-', lw=2, label='Контур слоя')

def animate(i):
    sc.set_offsets(frames[i])
    t = times[i]
    info_text.set_text(f"t = {t:.2f} c\nВремя релаксации: {relaxation_time:.2f} c\nКоэф. заполнения: {packing_fraction:.3f}\n{mixture_info}")
    return sc, info_text

ani = FuncAnimation(fig, animate, frames=len(frames), interval=40, blit=False, repeat=True)
plt.show()
