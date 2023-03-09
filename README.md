# dpax
Differentiable collision detection for capsules or polygones in JAX that is fully compatible with `jit`, `grad`, and `vmap`. 

The capsule collision detection is based on the following [arXiv paper](https://arxiv.org/abs/2207.00202), and existing Julia package [DiffPills.jl](https://github.com/kevin-tracy/DiffPills.jl).

The polytope collision detection is based on the following [arXiv paper](https://arxiv.org/abs/2207.00669), and existing Julia package [DifferentiableCollisions.jl](https://github.com/kevin-tracy/DifferentiableCollisions.jl).

For general purpose differentiable collision detection between a variety of convex primitives, see the [second paper](https://arxiv.org/abs/2207.00669) and [DifferentiableCollisions.jl](https://github.com/kevin-tracy/DifferentiableCollisions.jl).  The framework in that package can handle interactions between any pair of 5 different convex primitives. In this repo, only functionality for polytopes has been ported to python/JAX.

## Installation

To install directly from github using `pip`:

```bash
$ pip install git+https://github.com/kevin-tracy/dpax
```

Alternatively, to install from source:

```bash
$ python setup.py install
```

## Basic Usage (Capsules)
This package allows for proximity calculations between capsules that can be described in one of two different ways. The first way parametrizes the capsule by its endpoints and radius, and the second way parametrizes the capsule with a position and attitude (modified rodrigues parameter). 

### Endpoints

You can specify a capsule by its endpoints, and call `proximity` on this description as follows:
```python
import jax
import jax.numpy as jnp 
from jax import jit, grad, vmap 

import dpax
from dpax.endpoints import proximity

"""
Capsules can be described by their endpoints a, b,
and a radius R. 

         ---------------       -
       /                 \     |  
      /                   \    |
     (   . a          b .  )   | R
      \                   /    |
       \                 /     |
         ---------------       -
"""         
# capsule 1 
R1 = 1.4
a1 = jnp.array([2.3337867, -4.107256, 1.7219955])
b1 = jnp.array([1.8662131, -2.4927437, 1.0780045])

# capsule 2 
R2 = 0.7
a2 = jnp.array([-2.2803261, -3.7882166, -4.4897776])
b2 = jnp.array([-1.9196738, -4.811784, -4.3102226])

# calculate proximity 
phi = proximity(R1,a1,b1,R2,a2,b2)

# calculate proximity gradients
proximity_grad = grad(proximity, argnums = (0,1,2,3,4,5))
(dphi_dR1, dphi_da1, dphi_db1,
 dphi_dR2, dphi_da2, dphi_db2) = proximity_grad(R1,a1,b1,R2,a2,b2)

# check these gradients with finite diff
from jax.test_util import check_grads
check_grads(proximity,  (R1,a1,b1,R2,a2,b2), order=1, atol = 5e-2)
```
We can also `vmap` over these two functions. 
```python 
# random keys 
key1 = jax.random.PRNGKey(0)
key2, key3 = jax.random.split(key1)
key4, key5 = jax.random.split(key2)
key6, key7 = jax.random.split(key3)

N_capsules = 40
a1s = jax.random.normal(key1, (N_capsules, 3))
b1s = jax.random.normal(key2, (N_capsules, 3))
a2s = jax.random.normal(key3, (N_capsules, 3))
b2s = jax.random.normal(key4, (N_capsules, 3))
R1s = jax.random.normal(key5, (N_capsules,))
R2s = jax.random.normal(key7, (N_capsules,))

# vmap over proxmity
batch_proximity = jax.vmap(proximity, in_axes = (0,0,0,0,0,0))
phis = batch_proximity(R1s,a1s,b1s,R2s,a2s,b2s)

# vmap over proximit_grad
batch_proximity_grad = jax.vmap(proximity_grad, in_axes = (0,0,0,0,0,0))
(g_R1s, g_a1s, g_b1s,
 g_R2s, g_a2s, g_b2) = batch_proximity_grad(R1s,a1s,b1s,R2s,a2s,b2s)
```

### Position and Attitude

```python
import jax
import jax.numpy as jnp 
from jax import jit, grad, vmap 

import dpax
from dpax.mrp import proximityMRP

"""
Each capsule is described with:
 - R: radius 
 - L: length 
 - r: [3] position 
 - p, [3] attitude (modified rodrigues parameter)
"""

R1 = 1.4
L1 = 1.8
R2 = 0.7
L2 = 1.1
r1 = jnp.array([2.1,-3.3,1.4])
p1 = jnp.array([0.1,0.3,0.4])
r2 = jnp.array([-2.1,-4.3,-4.4])
p2 = jnp.array([-0.23,0.11,-0.32])

# calculate proximity 
phi = proximityMRP(R1,L1,r1,p1,R2,L2,r2,p2)

# calculate proximity gradients 
proximityMRP_grad = grad(proximityMRP, argnums = (0,1,2,3,4,5,6,7))
(g_R1, g_L1, g_r1, g_p1,
 g_R2, g_L2, g_r2, g_p2) = proximityMRP_grad(R1,L1,r1,p1,R2,L2,r2,p2)

# check these gradients with finite diff
from jax.test_util import check_grads
check_grads(proximityMRP,  (R1,L1,r1,p1,R2,L2,r2,p2), order=1)
```
We can also `vmap` over these functions:

```python
key1 = jax.random.PRNGKey(0)
key2, key3 = jax.random.split(key1)
key4, key5 = jax.random.split(key2)
key6, key7 = jax.random.split(key3)
key8, key9 = jax.random.split(key4)

N_capsules = 40
p1s = jax.random.normal(key1, (N_capsules, 3))
r1s = jax.random.normal(key2, (N_capsules, 3))
p2s = jax.random.normal(key3, (N_capsules, 3))
r2s = jax.random.normal(key4, (N_capsules, 3))
R1s = jax.random.normal(key5, (N_capsules,))
L1s = jax.random.normal(key6, (N_capsules,))
R2s = jax.random.normal(key7, (N_capsules,))
L2s = jax.random.normal(key8, (N_capsules,))

# vmap over proxmityMRP
batch_proximity = jax.vmap(proximityMRP, in_axes = (0,0,0,0,0,0,0,0))

phis = batch_proximity(R1s, L1s, r1s, p1s, R2s, L2s, r2s, p2s)

# vmap over proximityMRP_grad
batch_proximity_grad = jax.vmap(proximityMRP_grad, in_axes = (0,0,0,0,0,0,0,0))

(g_R1s, g_L1s, g_r1s, g_p1s,
 g_R2s, g_L2s, g_r2s, g_p2s) = batch_proximity_grad(R1s, L1s, r1s, p1s, R2s, L2s, r2s, p2s)
```

## Basic Usage (Polytopes) 


```python 
import jax
import jax.numpy as jnp 
from jax import jit, grad, vmap 
from jax.test_util import check_grads

import dpax
from dpax.mrp import dcm_from_mrp
from dpax.polytopes import polytope_proximity

 
# rectangular prism in Ax<=b form (halfspace form)
def create_rect_prism(length, width, height):

  A = jnp.array([
      [1,0,0.],
      [0,1,0.],
      [0,0,1.],
      [-1,0,0.],
      [0,-1,0.],
      [0,0,-1.]
  ])

  cs = jnp.array([
      [length/2,0,0.],
      [0,width/2,0.],
      [0.,0,height/2],
      [-length/2,0,0.],
      [0,-width/2,0.],
      [0.,0,-height/2]
  ])

  # b[i] = dot(A[i,:], b[i,:]) 
  b = jax.vmap(jnp.dot, in_axes = (0,0))(A, cs)

  return A, b 

# create polytopes 
A1, b1 = create_rect_prism(1,2,3)
A2, b2 = create_rect_prism(2,4,3)

# position and attitude for each polytope 
r1 = jnp.array([1,3,-2.])
p1 = jnp.array([.1,.3,.4])
Q1 = dcm_from_mrp(p1)

r2 = jnp.array([-1,0.1,2.])
p2 = jnp.array([-.3,.3,-.2])
Q2 = dcm_from_mrp(p2)

# calculate proximity (alpha <= 1 means collision) 
alpha = polytope_proximity(A1,b1,r1,Q1,A2,b2,r2,Q2)

print("alpha: ", alpha)

# calculate all the gradients 
grad_f = jit(grad(polytope_proximity, argnums = (0,1,2,3,4,5,6,7)))
grads = grad_f(A1,b1,r1,Q1,A2,b2,r2,Q2)

# check gradients 
check_grads(polytope_proximity,  (A1,b1,r1,Q1,A2,b2,r2,Q2), order=1, atol = 2e-1)
```
