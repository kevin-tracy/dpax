import jax 
from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 

from dpax.qp_utils import *

@custom_jvp
def proximity(R1,a1,b1,R2,a2,b2):

	L1 = jnp.linalg.norm(a1 - b1)
	L2 = jnp.linalg.norm(a2 - b2)

	Q, q, r = get_cost_terms(a1,b1,a2,b2)

	z = active_set_qp(Q,q)

	return cost(z,Q,q) + r - (R1 + R2)**2

@jit 
def lagrangian(R1,a1,b1,R2,a2,b2,z):
	L1 = jnp.linalg.norm(a1 - b1)
	L2 = jnp.linalg.norm(a2 - b2)

	Q, q, r = get_cost_terms(a1,b1,a2,b2)

	return cost(z,Q,q) + r - (R1 + R2)**2

@proximity.defjvp
def _proximity_gradient(primals, tangents):
	R1,  a1,  b1,  R2,  a2,  b2 = primals 
	dR1, da1, db1, dR2, da2, db2 = tangents 

	# proxmity function 
	L1 = jnp.linalg.norm(a1 - b1)
	L2 = jnp.linalg.norm(a2 - b2)

	Q, q, r = get_cost_terms(a1,b1,a2,b2)

	z = active_set_qp(Q,q)

	primal_out = cost(z,Q,q) + r - (R1 + R2)**2

	gR1, ga1, gb1, gR2, ga2, gb2 = grad(lagrangian, argnums=(0,1,2,3,4,5))(R1,a1,b1,R2,a2,b2,z)

	# form tangent out 
	tangent_out = (gR1.dot(dR1) + ga1.dot(da1) + gb1.dot(db1) +
									gR2.dot(dR2) + ga2.dot(da2) + gb2.dot(db2))


	return primal_out, tangent_out 


# R1 = 1.4
# L1 = 1.8
# R2 = 0.7
# L2 = 1.1
# r1 = jnp.array([2.1,-3.3,1.4])
# p1 = jnp.array([0.1,0.3,0.4])
# r2 = jnp.array([-2.1,-4.3,-4.4])
# p2 = jnp.array([-0.23,0.11,-0.32])

# a1, b1 = get_ends(L1,r1,p1)
# a2, b2 = get_ends(L2,r2,p2)

# phi = proximity(R1,a1,b1,R2,a2,b2)
# print(phi)

# # # calculate proximity gradients 
# proximity_grad = grad(proximity, argnums = (0,1,2,3,4,5))
# dphi_dR1, dphi_da1, dphi_db1, dphi_dR2, dphi_da2, dphi_db2 = proximity_grad(R1,a1,b1,R2,a2,b2)
# print("dphi_dR1",dphi_dR1)
# print("dphi_da1",dphi_da1)
# print("dphi_db1",dphi_db1)
# print("dphi_dR2",dphi_dR2)
# print("dphi_da2",dphi_da2)
# print("dphi_db2",dphi_db2)

# J1 = jax.jacobian(get_ends, argnums =(1,2))(L1,r1,p1)
# da1_dr1, da1_dp1 = J1[0]
# db1_dr1, db1_dp1 = J1[1]
# J2 = jax.jacobian(get_ends, argnums =(1,2))(L2,r2,p2)
# da2_dr2, da2_dp2 = J2[0]
# db2_dr2, db2_dp2 = J2[1]

# dphi_dr1 = dphi_da1 @ da1_dr1 + dphi_db1 @ db1_dr1
# dphi_dp1 = dphi_da1 @ da1_dp1 + dphi_db1 @ db1_dp1
# dphi_dr2 = dphi_da2 @ da2_dr2 + dphi_db2 @ db2_dr2
# dphi_dp2 = dphi_da2 @ da2_dp2 + dphi_db2 @ db2_dp2
# # print(dphi_da1)
# print(dphi_dr1)
# print(dphi_dp1)
# print(dphi_dr2)
# print(dphi_dp2)

# from jax.test_util import check_grads
# check_grads(proximity,  (R1,a1,b1,R2,a2,b2), order=1, atol = 5e-2)

# # vmap stuff 
# key1 = jax.random.PRNGKey(0)
# key2, key3 = jax.random.split(key1)
# key4, key5 = jax.random.split(key2)
# key6, key7 = jax.random.split(key3)
# key8, key9 = jax.random.split(key4)

# N_capsules = 40
# a1s = jax.random.normal(key1, (N_capsules, 3))
# b1s = jax.random.normal(key2, (N_capsules, 3))
# a2s = jax.random.normal(key3, (N_capsules, 3))
# b2s = jax.random.normal(key4, (N_capsules, 3))
# R1s = jax.random.normal(key5, (N_capsules,))
# R2s = jax.random.normal(key7, (N_capsules,))

# # vmap over proxmityMRP
# batch_proximity = jax.vmap(proximity, in_axes = (0,0,0,0,0,0))

# phis = batch_proximity(R1s,a1s,b1s,R2s,a2s,b2s)

# # vmap over proximit_grad
# batch_proximity_grad = jax.vmap(proximity_grad, in_axes = (0,0,0,0,0,0))

# g_R1s,g_a1s,g_b1s,g_R2s,g_a2s,g_b2 = batch_proximity_grad(R1s,a1s,b1s,R2s,a2s,b2s)
