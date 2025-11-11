import ufl
import numpy

from mpi4py import MPI

from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem


def q(u):
    return 1 + u**2


domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2 * x[1]
f = -ufl.div(q(u_ufl) * ufl.grad(u_ufl))

V = fem.functionspace(domain, ("Lagrange", 1))

def u_exact(x):
    return eval(str(u_ufl))
    
    
    
u_D = fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))


uh = fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_monitor": None,
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}


problem = NonlinearProblem(
    F,
    uh,
    bcs=[bc],
    petsc_options=petsc_options,
    petsc_options_prefix="nonlinpoisson",
)


problem.solve()
converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()
assert converged > 0, "Solver did not converge, got {converged}."
print(
    f"Solver converged after {num_iter} iterations with converged reason {converged}."
)

# Compute L2 error and error at nodes
V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(
    numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX
)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")
    
