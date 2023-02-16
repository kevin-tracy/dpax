import jax 
from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 

# cost terms for QP
def get_cost_terms(a,b,c,d):

	F = jnp.vstack(((a-b), (d - c))).T # annoying
	g = b - d

	Q = 2 * F.T @ F
	q = 2 * F.T @ g


	return (Q, q, jnp.dot(g,g))

# is z in [0, 1]
def in_0_1(z):
	return jnp.where((jnp.min(z) >= 0) * (jnp.max(z) <= 1), True, False)

# cost for QP (0.5z'Qz + q'z)
def cost(z,Q,q):
	return (0.5 * z.dot( Q @ z) + z.dot(q))

# evaluate active set boundaries
def eval_boundaries(Q,q):
	# test all 8 boundaries of feasible set
	Q1 = Q[0,0]
	Q2 = Q[0,1]
	Q3 = Q[1,1]
	q1 = q[0]
	q2 = q[1]

	# fix one value, and evaluate the unconstrained value for the other
	# after clipping it to be within 0,1
	tmp = jnp.clip(-(Q2 + q2)/Q3,0.0,1.0)
	z1 = jnp.array([1.0, tmp])
	J1 = cost(z1,Q,q)

	tmp = jnp.clip(-q2/Q3,0.0,1.0)
	z2 = jnp.array([0.0, tmp])
	J2 = cost(z2,Q,q)

	tmp = jnp.clip(-(Q2 + q1)/Q1,0.0,1.0)
	z3 = jnp.array([tmp, 1.0])
	J3 = cost(z3,Q,q)

	tmp = jnp.clip(-q1/Q1,0.0,1.0)
	z4 = jnp.array([tmp, 0.0])
	J4 = cost(z4,Q,q)

	# vector of costs
	Js = jnp.array([J1, J2, J3, J4])

	# vector of potential solutions
	Zs = jnp.array([z1,z2,z3,z4])

	# return solution with min cost
	minIndex = jnp.argmin(Js)
	return Zs[minIndex]

# active set QP solver for min 0.5z'*Q*z + q'z st. 0 <= z <= 1
def active_set_qp(Q, q):

	# check for (regularized) unconstrained solution
	reg = jnp.where(jnp.abs(jnp.linalg.det(Q)) < 1e-5, 1e-5, 0.0)
	Qreg = Q + reg*jnp.identity(2)
	z = -jax.scipy.linalg.solve(Qreg,q,assume_a = 'pos')

	# return z if uncon solution is feasible, otherwise eval_boundaries
	return jnp.where(in_0_1(z), z, eval_boundaries(Q,q))