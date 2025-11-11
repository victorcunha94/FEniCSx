# buckley_quarter_fivespot_upwind.py
# Euler explícito + upwind para quarter five-spot (dolfinx >= 0.10.0)

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, plot
import ufl
import pyvista

# -------------------------
# Parâmetros do problema
# -------------------------
Lx, Ly = 1.0, 1.0
Nx_cells, Ny_cells = 128, 128   # ajustar conforme memória/precisão
nx_nodes = Nx_cells + 1
ny_nodes = Ny_cells + 1

# Velocidades constantes (quarter five-spot típico: fluxo diagonal)
v_x = 1.0
v_y = 1.0
vel = np.array([v_x, v_y])


# adiciona parâmetro m (razão de mobilidade)
M = 10.0   # experimente 5, 10, 20

def fractional_flow_arr(s, m=M):
    return (m * s**2) / (m * s**2 + (1.0 - s)**2 + 1e-14)


# CFL e tempo
CFL = 0.4
dx_val = Lx / Nx_cells
dy_val = Ly / Ny_cells
dt = CFL * min(dx_val / abs(v_x) if v_x != 0 else 1e6,
               dy_val / abs(v_y) if v_y != 0 else 1e6)

T = 0.5
num_steps = int(np.ceil(T / dt))
dt = float(T / num_steps)  # reajusta dt para dividir T exatamente

# -------------------------
# Malha e espaço de funções
# -------------------------
msh = mesh.create_rectangle(MPI.COMM_WORLD,
                            [[0.0, 0.0], [Lx, Ly]],
                            [Nx_cells, Ny_cells],
                            mesh.CellType.triangle)

V = fem.functionspace(msh, ("Lagrange", 1))

# -------------------------
# Coordenadas e mapeamento para grade regular
# -------------------------
coords = np.asarray(msh.geometry.x)  # (N_vertices, 2)
xu = np.unique(np.round(coords[:, 0], 12))
yu = np.unique(np.round(coords[:, 1], 12))
if xu.size * yu.size != coords.shape[0]:
    raise RuntimeError("Malha não estruturada regular detectada — este código assume malha regular.")

# dicionário coord -> index do vértice
coord_to_index = {}
for k, c in enumerate(coords):
    key = (round(float(c[0]), 12), round(float(c[1]), 12))
    coord_to_index[key] = k

index_grid = np.empty((yu.size, xu.size), dtype=np.int64)
for iy, yv in enumerate(yu):
    for ix, xv in enumerate(xu):
        key = (round(float(xv), 12), round(float(yv), 12))
        index_grid[iy, ix] = coord_to_index[key]

# -------------------------
# Inicialização da saturação s^0 (s=0 no domínio)
# -------------------------
s = fem.Function(V)
s.name = "Saturation"
s.x.array[:] = 0.0

# Forçar injeção no canto (0,0). Encontra os índices cujo coord distancia < tol
tol = min(dx_val, dy_val) * 0.5


injection_nodes = []
production_nodes = []
for k, c in enumerate(coords):
    if (abs(c[0] - 0.0) < tol) and (abs(c[1] - 0.0) < tol):
        injection_nodes.append(k)
    if (abs(c[0] - Lx) < tol) and (abs(c[1] - Ly) < tol):
        production_nodes.append(k)

# garantir que injeção exista (senão pega nós mais próximos)
if len(injection_nodes) == 0:
    # procura nó com menor distância a (0,0)
    dists = (coords[:,0]**2 + coords[:,1]**2)
    injection_nodes = [int(np.argmin(dists))]
if len(production_nodes) == 0:
    dists = ((coords[:,0]-Lx)**2 + (coords[:,1]-Ly)**2)
    production_nodes = [int(np.argmin(dists))]

inj_center = injection_nodes[0]
iy0, ix0 = np.where(index_grid == inj_center)
iy0, ix0 = int(iy0[0]), int(ix0[0])
inj_radius = 2
inj_set = []
for dyi in range(-inj_radius, inj_radius+1):
    for dxi in range(-inj_radius, inj_radius+1):
        iy = min(max(0, iy0 + dyi), yu.size-1)
        ix = min(max(0, ix0 + dxi), xu.size-1)
        inj_set.append(int(index_grid[iy, ix]))
injection_nodes = sorted(set(inj_set))
print(f"Injection nodes count: {len(injection_nodes)}")


# aplica condição inicial: domínio óleo (0) já definida; injetor = 1
#s.x.array[:] = 0.0
for idx in injection_nodes:
    s.x.array[idx] = 1.0

# -------------------------
# Preparação arrays para evolução explícita
# -------------------------
s_vec = np.asarray(s.x.array.real).copy()   # vetor global atual
s_new = np.empty_like(s_vec)

# Precomputar sinal de v para upwind
sign_vx = np.sign(v_x)
sign_vy = np.sign(v_y)

# -------------------------
# Preparar XDMF para salvar
# -------------------------
xdmf = io.XDMFFile(msh.comm, "buckley_quarter_fivespot_upwind.xdmf", "w")
xdmf.write_mesh(msh)
xdmf.write_function(s, 0.0)


try:
    if msh.comm.rank == 0:
        cells, types, points = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, points)
        plotter = pyvista.Plotter(off_screen=True)
        plotter.open_gif("saturation_evolution.gif", fps=10)
        grid.point_data["s"] = s_vec
        warped = grid.warp_by_scalar("s", factor=1)
        plmesh = plotter.add_mesh(warped, show_edges=False, clim=[0, 1], cmap='viridis')
        plotter.add_scalar_bar("Saturation", vertical=True)
except Exception as e:
    print(f"PyVista setup failed: {e}")
    plotter = None

# -------------------------
# Loop explícito com upwind conservativo
# -------------------------
print(f"nx={xu.size}, ny={yu.size}, dt={dt:.3e}, num_steps={num_steps}")
for step in range(1, num_steps + 1):
    t = step * dt

    # Reconstruir grade 2D de s
    s_grid = np.empty((yu.size, xu.size), dtype=float)
    for iy in range(yu.size):
        for ix in range(xu.size):
            idx = index_grid[iy, ix]
            s_grid[iy, ix] = s_vec[idx]

    # calcular f(s) na grade
    #f_grid = fractional_flow_arr(s_grid)
    f_grid = fractional_flow_arr(s_grid, m=M)

    # fluxos nas interfaces (F_x at i+1/2, F_y at j+1/2) por upwind
    Fx = np.zeros((yu.size, xu.size + 1), dtype=float)
    Fy = np.zeros((yu.size + 1, xu.size), dtype=float)

    # X interfaces: para cada interface i+1/2, usar upwind de acordo com sign_vx
    if sign_vx >= 0:
        # v_x >= 0 -> upwind pega valor à esquerda
        for iy in range(yu.size):
            Fx[iy, 0] = v_x * f_grid[iy, 0]  # boundary
            for ix in range(1, xu.size):
                Fx[iy, ix] = v_x * f_grid[iy, ix-1]
            Fx[iy, xu.size] = v_x * f_grid[iy, xu.size-1]  # boundary
    else:
        # v_x < 0 -> upwind pega valor à direita
        for iy in range(yu.size):
            Fx[iy, 0] = v_x * f_grid[iy, 0]  # boundary
            for ix in range(0, xu.size-1):
                Fx[iy, ix+1] = v_x * f_grid[iy, ix+1]
            Fx[iy, xu.size] = v_x * f_grid[iy, xu.size-1]  # boundary

    # Y interfaces: para cada interface j+1/2, usar upwind de acordo com sign_vy
    if sign_vy >= 0:
        # v_y >= 0 -> upwind pega valor abaixo
        for ix in range(xu.size):
            Fy[0, ix] = v_y * f_grid[0, ix]  # boundary
            for iy in range(1, yu.size):
                Fy[iy, ix] = v_y * f_grid[iy-1, ix]
            Fy[yu.size, ix] = v_y * f_grid[yu.size-1, ix]  # boundary
    else:
        # v_y < 0 -> upwind pega valor acima
        for ix in range(xu.size):
            Fy[0, ix] = v_y * f_grid[0, ix]  # boundary
            for iy in range(0, yu.size-1):
                Fy[iy+1, ix] = v_y * f_grid[iy+1, ix]
            Fy[yu.size, ix] = v_y * f_grid[yu.size-1, ix]  # boundary

    # Atualização explícita: s_new = s_old - dt * (div F)
    for iy in range(yu.size):
        for ix in range(xu.size):
            divF = (Fx[iy, ix+1] - Fx[iy, ix]) / dx_val + (Fy[iy+1, ix] - Fy[iy, ix]) / dy_val
            s_new[index_grid[iy, ix]] = s_grid[iy, ix] - dt * divF

    # Manter condições de contorno: injeção sempre em 1
    for idx in injection_nodes:
        s_new[idx] = 1.0

    # Garantir limites físicos [0, 1]
    s_new = np.clip(s_new, 0.0, 1.0)

    # Atualizar s_vec para próximo passo
    s_vec[:] = s_new[:]

    # Atualizar função dolfinx para output
    s.x.array[:] = s_vec

    # Salvar a cada N passos
    if step % max(1, num_steps // 20) == 0 or step == num_steps:
        xdmf.write_function(s, t)
        print(f"Step {step}/{num_steps}, t={t:.3f}, min(s)={np.min(s_vec):.3f}, max(s)={np.max(s_vec):.3f}")

    # Atualizar visualização pyvista
    if plotter is not None and msh.comm.rank == 0:
        grid.point_data["s"] = s_vec
        warped = grid.warp_by_scalar("s", factor=0.1)  # Reduced warp factor for better visualization
        plotter.remove_actor(plmesh)
        plmesh = plotter.add_mesh(warped, show_edges=False, clim=[0, 1], cmap='viridis')
        plotter.write_frame()

# -------------------------
# Finalização
# -------------------------
xdmf.close()
if plotter is not None and msh.comm.rank == 0:
    plotter.close()

print("Simulação concluída!")
print(f"Saturação final: min={np.min(s_vec):.3f}, max={np.max(s_vec):.3f}, mean={np.mean(s_vec):.3f}")
