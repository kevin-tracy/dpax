import jax 
from jax import grad 
from jax import jit 
from jax import custom_jvp
from jax import jvp 
import jax.numpy as jnp 

"""
Primal dual interior point method for solving the following problem

    min    q'x
    st     Gx ≦ h

alg: https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf

KKT systems are solved via reduction
reduction: Nocedal and Wright, Numerical Optimization, pg 482 (16.62)
"""

@jit 
def ort_linesearch(x,dx):
  # maximum alpha <= 1 st x + alpha * dx >= 0 
  body = lambda _x, _dx: jnp.where(_dx<0, -_x/_dx, jnp.inf)
  batch = jax.vmap(body, in_axes = (0,0))
  return jnp.min(jnp.array([1.0, jnp.min(batch(x,dx))]))

@jit 
def centering_params(s, z, ds_a, dz_a):
  # duality gap + cc term in predictor-corrector PDIP 
  mu = jnp.dot(s, z)/len(s)

  alpha = jnp.min(jnp.array([
      ort_linesearch(s, ds_a),
      ort_linesearch(z, dz_a)]
      ))

  sigma = (jnp.dot(s + alpha * ds_a, z + alpha * dz_a)/jnp.dot(s, z))**3

  return sigma, mu 


@jit 
def pdip_step(inputs):
  q,G,h,x,s,z,pdip_iter = inputs 
  
  # evaluate residuals 
  r1 = G.T @ z + q 
  r2 = s * z 
  r3 = G @ x + s - h 

  # solve for affine step 
  invSZ = jnp.diag(z / s)

  GSG = G.T @ invSZ @ G
  max_elt = jnp.max(jnp.abs(GSG))
  # Regularize since GSG may be non-positive-semidefinite due to numerical
  # issues
  GSG = GSG + 1e-8 * max_elt * jnp.eye(4)
  F = jax.scipy.linalg.cho_factor(GSG)

  dx_a = jax.scipy.linalg.cho_solve(F, -r1 + G.T @ invSZ @ (-r3 + (r2 / z)))
  ds_a = -(G @ dx_a + r3 )
  dz_a = -(r2 + (z * ds_a)) / s 

  # corrector + centering step 
  sigma, mu = centering_params(s, z, ds_a, dz_a)
  r2 = r2 - (sigma * mu - (ds_a * dz_a))
  dx = jax.scipy.linalg.cho_solve(F, -r1 + G.T @ invSZ @ (-r3 + (r2 / z)))
  ds = -(G @ dx + r3) 
  dz = -(r2 + (z * ds)) / s 

  # linesearch and update primal & dual vars 
  alpha = 0.99*jnp.min(jnp.array([
                ort_linesearch(s, ds),
                ort_linesearch(z, dz) 
                ]))
  x = x + alpha * dx 
  s = s + alpha * ds 
  z = z + alpha * dz 

  return (q,G,h,x,s,z,pdip_iter + 1)


@jit 
def solve_lp(q,G,h):

  ns, nx = G.shape

  # must initialize with s, z > 0 
  x = jnp.zeros(nx) # primal 
  s = jnp.ones(ns)  # slack (primal)
  z = jnp.ones(ns)  # dual 

  # keep iterating only if the following are true:
  # - pdip_iter < 20, so the max iters are 20 
  # - either duality gap is too high, or equality res is 
  def continuation_criteria(inputs):
    q,G,h,x,s,z,pdip_iter = inputs 

    duality_gap = s.dot(z)/len(s)
    eq_res = jnp.linalg.norm(G @ x + s - h)

    return jnp.logical_and(
                           pdip_iter < 20, # continue if it's below max iter
                           jnp.logical_or(duality_gap > 1e-5, eq_res > 1e-5))
                           # and either duality gap or eq res aren't converged

  inputs = (q,G,h,x,s,z, 0)

  outputs = jax.lax.while_loop(continuation_criteria, pdip_step, inputs)

  x = outputs[3]
  s = outputs[4]
  z = outputs[5] 

  return x, s, z 
