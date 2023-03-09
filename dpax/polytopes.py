import jax 
from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 

from dpax.pdip_solver import *

"""
function to create standard form min c'x st Gx<=h from 
the description of two polytopes using the DCOL alg. 

https://arxiv.org/abs/2207.00669

each polytope is described in a body frame (B) by Ax<=b, 
with a position r in some world frame (W), and an attitude 
described by a 3x3 rotation matrix W_Q_B. 

args:
	A1: [n1,3] jnp.array, polytope 1 description (Ax <= b)
	b1: [n1] jnp.array, polytope 1 description (Ax <= b)
	r1: [3] jnp.array, position of polytope 1 in world frame 
	Q1: [3,3] jnp.array, W_Q_B rotation matrix for poly 1 
	A2: [n2,3] jnp.array, polytope 2 description (Ax <= b)
	b2: [n2] jnp.array, polytope 2 description (Ax <= b)
	r2: [3] jnp.array, position of polytope 2 in world frame 
	Q2: [3,3] jnp.array, W_Q_B rotation matrix for poly 2

outputs:
	c: [4] jnp.array, linear cost term 
	G: [n1 + n2, 4] jnp.array, inequality constraint Gx<=h
	h: [n1 + n2] jnp.array, inequality constraint Gx<=h

"""
def problem_matrices(A1,b1,r1,Q1,A2,b2,r2,Q2):

  c = jnp.array([0,0,0,1.])

  G = jnp.vstack((
      jnp.hstack((A1 @ Q1.T, -jnp.reshape(b1,(6,1)) )),
      jnp.hstack((A2 @ Q2.T, -jnp.reshape(b2,(6,1)) ))
  ))
  h = jnp.concatenate((
      A1 @ Q1.T @ r1, 
      A2 @ Q2.T @ r2, 
  ))

  return c, G, h 



@custom_jvp
@jit 
def polytope_proximity(A1,b1,r1,Q1,A2,b2,r2,Q2):
  c, G, h = problem_matrices(A1,b1,r1,Q1,A2,b2,r2,Q2)
  x,s,z = solve_lp(c,G,h)
  return x[3]

@jit 
def polytope_lagrangian(A1,b1,r1,Q1,A2,b2,r2,Q2, x, s, z):
  c, G, h = problem_matrices(A1,b1,r1,Q1,A2,b2,r2,Q2)

  # ommit the cost term since c'x doesn't depend on problem data 
  return z.dot(G @ x - h)

@jit 
def polytope_proximity_grads(A1,b1,r1,Q1,A2,b2,r2,Q2):
  c, G, h = problem_matrices(A1,b1,r1,Q1,A2,b2,r2,Q2)
  x,s,z = solve_lp(c,G,h)

  alpha = x[3]

  lag_grad = grad(polytope_lagrangian, argnums = (0,1,2,3,4,5,6,7))
  grads = lag_grad(A1,b1,r1,Q1,A2,b2,r2,Q2, x, s, z)

  gA1, gb1, gr1, gQ1, gA2, gb2, gr2, gQ2 = grads 

  return alpha, gA1, gb1, gr1, gQ1, gA2, gb2, gr2, gQ2

@polytope_proximity.defjvp
@jit 
def _polytope_proximity_gradient(primals, targets):
  A1,b1,r1,Q1,A2,b2,r2,Q2 = primals 
  dA1,db1,dr1,dQ1,dA2,db2,dr2,dQ2 = targets 

  grads = polytope_proximity_grads(A1,b1,r1,Q1,A2,b2,r2,Q2)

  alpha, gA1, gb1, gr1, gQ1, gA2, gb2, gr2, gQ2 = grads

  primal_out = alpha 

  tangent_out = (jnp.sum(dA1 * gA1) + db1.dot(gb1) + dr1.dot(gr1) + jnp.sum(dQ1 * gQ1) + 
                 jnp.sum(dA2 * gA2) + db2.dot(gb2) + dr2.dot(gr2) + jnp.sum(dQ2 * gQ2)) 
  
  return primal_out, tangent_out 
  