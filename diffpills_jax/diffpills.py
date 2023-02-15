import jax 
from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 

# direction cosine matrix from mrp
def dcm_from_mrp(p):
	p1,p2,p3 = p
	den = (p1**2 + p2**2 + p3**2 + 1)**2
	a = (4*p1**2 + 4*p2**2 + 4*p3**2 - 4)
	return jnp.array([[(-((8*p2**2+8*p3**2)/den-1)),   (8*p1*p2 + p3*a)/den,     (8*p1*p3 - p2*a)/den],
	               [(8*p1*p2 - p3*a)/den, (-((8*p1**2 + 8*p3**2)/den - 1)),   (8*p2*p3 + p1*a)/den],
	               [(8*p1*p3 + p2*a)/den,  (8*p2*p3 - p1*a)/den,  (-((8*p1**2 + 8*p2**2)/den - 1))]])


# get ends of the capsule
def get_ends(L,r,p):
	n_Q_b = dcm_from_mrp(p)
	v = n_Q_b @ jnp.array([L/2,0,0.0])
	a = r - v
	b = r + v
	return (a, b)

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


@custom_jvp
def proximityMRP(R1,L1,r1,p1,R2,L2,r2,p2):
	a,b = get_ends(L1,r1,p1)
	c,d = get_ends(L2,r2,p2)

	Q, q, r = get_cost_terms(a,b,c,d)

	z = active_set_qp(Q,q)

	return cost(z,Q,q) + r - (R1 + R2)**2

@jit 
def lagrangianMRP(R1,L1,r1,p1,R2,L2,r2,p2,z):
	a,b = get_ends(L1,r1,p1)
	c,d = get_ends(L2,r2,p2)

	Q, q, r = get_cost_terms(a,b,c,d)

	return cost(z,Q,q) + r - (R1 + R2)**2

@proximityMRP.defjvp
def _proximityMRP_gradient(primals, tangents):
  R1,L1,r1,p1,R2,L2,r2,p2 = primals 
  dR1,dL1,dr1,dp1,dR2,dL2,dr2,dp2 = tangents 

  # proxmity_mrp function 
  a,b = get_ends(L1,r1,p1)
  c,d = get_ends(L2,r2,p2)

  Q, q, r = get_cost_terms(a,b,c,d)

  z = active_set_qp(Q,q)

  primal_out = cost(z,Q,q) + r - (R1 + R2)**2

  (g_R1, g_L1, g_r1, g_p1, g_R2, g_L2, g_r2, g_p2) = grad(lagrangianMRP, argnums=(0,1,2,3,4,5,6,7))(R1,L1,r1,p1,R2,L2,r2,p2,z)

  # form tangent out 
  tangent_out = (g_R1.dot(dR1) + g_L1.dot(dL1) + g_r1.dot(dr1) + 
                 g_p1.dot(dp1) + g_R2.dot(dR2) + g_L2.dot(dL2) +
                 g_r2.dot(dr2) + g_p2.dot(dp2))
  
  return primal_out, tangent_out 

# def proximityMRP_gradient(R1,L1,r1,p1,R2,L2,r2,p2):
# 	a,b = get_ends(L1,r1,p1)
# 	c,d = get_ends(L2,r2,p2)

# 	Q, q, r = get_cost_terms(a,b,c,d)

# 	z = active_set_qp(Q,q)

# 	phi = cost(z,Q,q) + r - (R1 + R2)**2

# 	g_r1, g_p1, g_r2, g_p2 = grad(lagrangianMRP, argnums=(2,3,6,7))(R1,L1,r1,p1,R2,L2,r2,p2,z)

# 	return phi, g_r1, g_p1, g_r2, g_p2

# R1 = 1.4
# L1 = 1.8
# R2 = 0.7
# L2 = 1.1
# r1 = jnp.array([2.1,-3.3,1.4])
# p1 = jnp.array([0.1,0.3,0.4])
# r2 = jnp.array([-2.1,-4.3,-4.4])
# p2 = jnp.array([-0.23,0.11,-0.32])

# phi = proximityMRP(R1,L1,r1,p1,R2,L2,r2,p2)

# phi, g_r1, g_p1, g_r2, g_p2 = proximityMRP_gradient(R1,L1,r1,p1,R2,L2,r2,p2)
# # print("phi",phi)
# print("g_r1",g_r1)
# print("g_p1",g_p1)
# print("g_r2",g_r2)
# print("g_p2",g_p2)

# # g_R1, g_L1, g_r1, g_p1, g_R2, g_L2, g_r2, g_p2 = 

# g_R1, g_L1, g_r1, g_p1, g_R2, g_L2, g_r2, g_p2 = grad(proximityMRP, argnums = (0,1,2,3,4,5,6,7))(R1,L1,r1,p1,R2,L2,r2,p2)

# print("g_r1",g_r1)
# print("g_p1",g_p1)
# print("g_r2",g_r2)
# print("g_p2",g_p2)

# from jax.test_util import check_grads
# print(check_grads(proximityMRP,  (R1,L1,r1,p1,R2,L2,r2,p2), order=1))

# key1 = jax.random.PRNGKey(0)
# key2, key3 = jax.random.split(key1)
# key4, key5 = jax.random.split(key2)
# key6, key7 = jax.random.split(key3)

# N_capsules = 40#_000
# p1s = jax.random.normal(key1, (N_capsules, 3))
# r1s = jax.random.normal(key2, (N_capsules, 3))
# p2s = jax.random.normal(key3, (N_capsules, 3))
# r2s = jax.random.normal(key4, (N_capsules, 3))
# R1s = jax.random.normal(key5, (N_capsules,))
# L1s = jax.random.normal(key5, (N_capsules,))
# R2s = jax.random.normal(key5, (N_capsules,))
# L2s = jax.random.normal(key5, (N_capsules,))

# batch_proximity = jax.vmap(proximityMRP, in_axes = (0,0,0,0,0,0,0,0))

# phis = batch_proximity(R1s, L1s, r1s, p1s, R2s, L2s, r2s, p2s)

# print(len(phis))
# print(phis)

# batch_proximity_grad = jax.vmap(grad(proximityMRP, argnums = (0,1,2,3,4,5,6,7)),in_axes = (0,0,0,0,0,0,0,0))

# g_R1s, g_L1s, g_r1s, g_p1s, g_R2s, g_L2s, g_r2s, g_p2s = batch_proximity_grad(R1s, L1s, r1s, p1s, R2s, L2s, r2s, p2s)

# print(g_R1s.shape)
# print(g_r1s.shape)


