from dolfinx import fem, io, mesh, plot
import ufl
from petsc4py.PETSc import ScalarType 
from ufl import dx, ds, grad, inner, FacetNormal
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
import numpy as np

#Construindo uma participação do domínio Omega, neste caso uma malha retangular
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 1))


#Identificando as faces que formam as condições de contorno
tdim = msh.topology.dim
fdim = tdim -1
msh.topology.create_connectivity(fdim, tdim)

#facets = mesh.locate_entities_boundary( msh, dim=(msh.topology.dim - 1),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0), )

#facets=np.flatnonzero(mesh.compute_boundary_facets(msh.topology))

facets = mesh.exterior_facet_indices(msh.topology)

dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)

u_boundary = fem.Constant(msh, ScalarType(0.0))
bc = fem.dirichletbc(u_boundary, dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
#f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
f = fem.Constant(msh, ScalarType(1.0))
mu = fem.Constant(msh, ScalarType(1.00))
#g = ufl.sin(5 * x[0])
a = ufl.inner(mu*ufl.grad(u), ufl.grad(v)) * ufl.dx
#L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

L = ufl.inner(f, v) * ufl.dx 

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="demo_poisson_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)
uh = problem.solve()
assert isinstance(uh, fem.Function)
uh.name = "Velocidade"
with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
    
    
try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution.")
    print("To install pyvista with pip: 'python3 -m pip install pyvista'.")


