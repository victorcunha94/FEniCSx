from dolfinx import fem, io, mesh, plot
import ufl
from petsc4py.PETSc import ScalarType 
from ufl import dx, grad, inner, TestFunction, TrialFunction
from mpi4py import MPI
import numpy as np

# Criar malha 1D
msh = mesh.create_interval(MPI.COMM_WORLD, nx=10000, points=[0.0, 1.0])

# Espaço de funções
V = fem.functionspace(msh, ("Lagrange", 1))

# Função para o fluxo de Buckley-Leverett
def buckley_leverett_flux(s):
    return (10 * s**2) / (10 * s**2 + (1 - s)**2)

# Condição inicial
def initial_condition(x):
    return 0.45 * np.sin(2 * np.pi * x[0]) + 0.5

# Aplicar condição inicial
s_sol = fem.Function(V, name="Saturação")
s_sol.interpolate(initial_condition)

# Parâmetros temporais
dt = 0.001
T = 0.05  # Tempo final
num_steps = int(T / dt)

# Lista para armazenar soluções em diferentes tempos
time_solutions = []

# Salvar condição inicial (t=0)
time_solutions.append((0.0, s_sol.copy()))

print("Simulação Buckley-Leverett...")

# Loop temporal explícito
for i in range(num_steps):
    current_time = (i + 1) * dt
    
    # Calcular o fluxo
    s_array = s_sol.x.array
    flux_array = buckley_leverett_flux(s_array)
    
    # Aplicar esquema upwind manualmente
    dx_val = 1.0 / 100  # tamanho da célula
    ds_dt = np.zeros_like(s_array)
    
    for j in range(1, len(s_array)-1):
        # Esquema upwind
        if flux_array[j] > flux_array[j-1]:
            ds_dt[j] = -(flux_array[j] - flux_array[j-1]) / dx_val
        else:
            ds_dt[j] = -(flux_array[j+1] - flux_array[j]) / dx_val
    
    # Condições de contorno periódicas
    ds_dt[0] = -(flux_array[1] - flux_array[0]) / dx_val
    ds_dt[-1] = -(flux_array[-1] - flux_array[-2]) / dx_val
    
    # Atualizar solução
    s_sol.x.array[:] += dt * ds_dt
    
    # Garantir que a saturação fique entre 0 e 1
    s_sol.x.array[:] = np.clip(s_sol.x.array, 0.0, 1.0)
    
    # Salvar em tempos específicos
    save_times = [0.01, 0.02, 0.03, 0.04, 0.05]
    if any(abs(current_time - t_save) < dt/2 for t_save in save_times):
        time_solutions.append((current_time, s_sol.copy()))
        print(f"Salvando t = {current_time:.3f}")
    
    if i % 100 == 0:
        print(f"Passo {i}, tempo {current_time:.3f}")

# Salvar todas as soluções em arquivos XDMF separados
for time, solution in time_solutions:
    filename = f"buckley_leverett_t{time:.3f}.xdmf"
    with io.XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(solution)
    print(f"Salvo: {filename}")

# Visualização com PyVista - TODOS OS TEMPOS JUNTOS
try:
    import pyvista
    import matplotlib.pyplot as plt
    
    # Criar figura para plot 2D
    plt.figure(figsize=(10, 6))
    
    # Coordenadas dos pontos da malha
    cells, types, x_coords = plot.vtk_mesh(V)
    
    # Plotar cada tempo
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    times = [t for t, _ in time_solutions]
    
    for idx, (time, solution) in enumerate(time_solutions):
        saturation = solution.x.array.real
        plt.plot(x_coords[:, 0], saturation, 
                label=f't = {time:.3f}', 
                color=colors[idx % len(colors)],
                linewidth=2)
    
    plt.xlabel('Posição x')
    plt.ylabel('Saturação da Água')
    plt.title('Evolução Temporal - Equação de Buckley-Leverett')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig('buckley_leverett_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualização 3D alternativa (tempo como terceira dimensão)
    plotter = pyvista.Plotter()
    
    for idx, (time, solution) in enumerate(time_solutions):
        cells, types, x = plot.vtk_mesh(V)
        
        # Criar coordenadas 3D: (x, tempo, saturação)
        points_3d = np.zeros((len(x), 3))
        points_3d[:, 0] = x[:, 0]  # Posição x
        points_3d[:, 1] = time * 5  # Escalar o tempo para melhor visualização
        points_3d[:, 2] = solution.x.array.real  # Saturação como altura
        
        grid = pyvista.UnstructuredGrid(cells, types, points_3d)
        grid.point_data["Saturação"] = solution.x.array.real
        
        plotter.add_mesh(grid, line_width=3, 
                        render_lines_as_tubes=True,
                        scalars=solution.x.array.real,
                        cmap="viridis")
    
    plotter.add_title("Evolução Temporal da Saturação")
    plotter.set_background("white")
    
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("buckley_leverett_3d.png")
    else:
        plotter.show()
        
except ModuleNotFoundError:
    print("'pyvista' e 'matplotlib' são necessários para visualização")

print("Simulação concluída!")
print(f"Tempos salvos: {[f't={t:.3f}' for t, _ in time_solutions]}")
