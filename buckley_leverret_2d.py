# buckley_leverret_2d.py
# Compatível com dolfinx 0.10.0
# Exemplo simplificado para simulação 2D do problema de Buckley-Leverett

from dolfinx import fem, io, mesh, plot
import ufl
from petsc4py.PETSc import ScalarType 
from ufl import dx, ds, grad, inner, FacetNormal
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
import numpy as np

# -------------------------------------------------------------------------
# 1. Domínio e espaço de funções
# -------------------------------------------------------------------------
Nx, Ny = 64, 64
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(Nx, Ny),
    cell_type=mesh.CellType.triangle,
)

V = fem.functionspace(msh, ("Lagrange", 1))

# -------------------------------------------------------------------------
# 2. Condições de contorno
# -------------------------------------------------------------------------
fdim = msh.topology.dim - 1
msh.topology.create_connectivity(fdim, msh.topology.dim)

# Exemplo: fronteira esquerda com valor 1.0, demais 0.0
def left_boundary(x):
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    return np.isclose(x[0], 1.0)

facets_left = mesh.locate_entities_boundary(msh, fdim, left_boundary)
facets_right = mesh.locate_entities_boundary(msh, fdim, right_boundary)

dofs_left = fem.locate_dofs_topological(V, fdim, facets_left)
dofs_right = fem.locate_dofs_topological(V, fdim, facets_right)

uL = fem.Constant(msh, ScalarType(1.0))
uR = fem.Constant(msh, ScalarType(0.0))

bcs = [
    fem.dirichletbc(uL, dofs_left, V),
    fem.dirichletbc(uR, dofs_right, V)
]

# -------------------------------------------------------------------------
# 3. Definições físicas e forma fraca
# -------------------------------------------------------------------------
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Permeabilidade ou fluxo fracionário f(u)
# Aqui apenas um exemplo genérico
def fractional_flow(u):
    return u**2 / (u**2 + (1 - u)**2)

x = ufl.SpatialCoordinate(msh)
K = fem.Constant(msh, ScalarType(1.0))  # coeficiente
f = fem.Constant(msh, ScalarType(0.0))  # fonte

# Fluxo advectivo não-linear (Buckley-Leverett simplificado)
a = ufl.inner(K * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# -------------------------------------------------------------------------
# 4. Montagem e solução (linear para teste)
# -------------------------------------------------------------------------
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options_prefix="poisson_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True
    },
)
uh = problem.solve()


# -------------------------------------------------------------------------
# 5. Pós-processamento
# -------------------------------------------------------------------------
with io.XDMFFile(msh.comm, "out_buckley_leverret_2d.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

# Visualização (opcional)
# Visualização (opcional)
try:
    import pyvista
    from dolfinx import plot

    # Remove ou comenta a linha do xvfb
    # pyvista.start_xvfb(wait=0.1)
    
    cells, cell_types, points = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_scalar_bar("Saturação")
    plotter.show()
except ModuleNotFoundError:
    print("Instale pyvista para visualização: pip install pyvista")
except Exception as e:
    print(f"Visualização não disponível: {e}")

