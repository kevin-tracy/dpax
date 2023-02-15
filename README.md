# diffpills_jax
Differentiable collision detection for capsules in JAX.


```python
import jax
import jax.numpy as jnp 
from jax import jit, grad, vmap 

import diffpills_jax.diffpills as dp 

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

# check these gradients 
from jax.test_util import check_grads
print(check_grads(proximityMRP,  (R1,L1,r1,p1,R2,L2,r2,p2), order=1))
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
batch_proximity_grad = jax.vmap(proximityMRP_grad),in_axes = (0,0,0,0,0,0,0,0))

(g_R1s, g_L1s, g_r1s, g_p1s,
 g_R2s, g_L2s, g_r2s, g_p2s) = batch_proximity_grad(R1s, L1s, r1s, p1s, R2s, L2s, r2s, p2s)
```
