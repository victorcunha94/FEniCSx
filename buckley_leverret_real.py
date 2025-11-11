# buckley_leverret_semi_implicit.py
import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)

# --- tempo e malha (reaproveitando seu esqueleto) ---
t = 0.0
T = 1.0
num_steps = 200
dt_val = T / num_steps

nx, ny = 64, 64
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-2, -2]), np.array([2, 2])],
    [nx, ny],
    mesh.CellType.triangle,
)
V = fem.functionspace(domain, ("Lagrange", 1))


def initial_condition(x, a=5):
    return np.exp(-a * (x[0] ** 2 + x[1] ** 2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(
    PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V
)

# --- XDMF e função solução ---
xdmf = io.XDMFFile(domain.comm, "buckley_semiimplicit.xdmf", "w")
xdmf.write_mesh(domain)

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

# --- Variáveis UFL ---
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# parâmetros físicos
epsilon = 1e-4  # difusão artificial
velocity = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0)))  # velocidade constante

# dt como Python float (montaremos a matriz uma única vez)
dt = float(dt_val)

# Formas para a parte linear implícita (tempo + difusão)
a_form = u * v * ufl.dx + dt * epsilon * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
A_form = fem.form(a_form)

# Monta matriz (constante no tempo) e fatoriza
A = assemble_matrix(A_form, bcs=[bc])
A.assemble()

# vetor RHS (criado a partir de espaços de L)
# a forma RHS depende de u_n e do termo advectivo explicitamente avaliado com u_n
# definimos o objeto linear_form abaixo e reusaremos o vetor b
u_n_f = u_n  # fusão de nome
f_term = fem.Constant(domain, PETSc.ScalarType(0.0))  # sem fonte externa

# Forma linear: RHS = ∫ u_n v dx  - dt * ∫ inner(f(u_n)*vel, grad(v)) dx
def fractional_flow_arr(u_val):
    # aqui usamos expressão UFL substituível abaixo via Function.interpolate
    # mas neste esquema vamos montar a forma com ufl.Function u_n
    return u_val**2 / (u_val**2 + (1.0 - u_val)**2 + 1e-16)

# UFL form using u_n as fem.Function (works because fem.form will capture it)
f_u_n = fractional_flow_arr(u_n)  # ufl expression using u_n
adv_form = -dt * ufl.inner(f_u_n * velocity, ufl.grad(v)) * ufl.dx
rhs_form = u_n * v * ufl.dx + adv_form
linear_form = fem.form(rhs_form)

# Pre-create vector b compatible with linear_form function spaces
b = create_vector(fem.extract_function_spaces(linear_form))

# --- Solver setup (reaproveitando KSP LU como você fez) ---
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# --- pyvista plot setup (igual ao seu) ---
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
plotter = pyvista.Plotter()
plotter.open_gif("buckley_semiimplicit.gif", fps=10)
grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=1)
viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=14, label_font_size=12, fmt="%.2e",
             color="black", position_x=0.1, position_y=0.8, width=0.8, height=0.1)
renderer = plotter.add_mesh(warped, show_edges=True, lighting=False, cmap=viridis,
                            scalar_bar_args=sargs, clim=[0, max(uh.x.array)])

# --- Loop temporal ---
for i in range(num_steps):
    t += dt

    # --- monta RHS com u_n (advectivo explicitamente) ---
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # aplicar BC no RHS
    apply_lifting(b, [A_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # --- resolver sistema linear para uh ---
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # --- atualizar u_n e salvar ---
    u_n.x.array[:] = uh.x.array[:]
    xdmf.write_function(uh, t)

    # update pyvista plot
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()

plotter.close()
xdmf.close()

A.destroy()
b.destroy()
solver.destroy()

