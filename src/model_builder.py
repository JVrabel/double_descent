"""
Contains JAX (stax) model code to instantiate an MLP model.
"""
import jax
from jax import numpy as np
from jax import random

import neural_tangents as nt
from neural_tangents import stax


def MLP_stax(input_shape, hidden_units, output_shape, batch_size):
    nn_init, nn_apply, _ = stax.serial(
                            stax.Dense(hidden_size),
                            stax.Relu(),
                            stax.Dense(output_shape)
                            )

    rng = random.PRNGKey(0)
    in_shape = (batch_size,) + input_shape
    out_shape, params = init_fun(rng, in_shape)

    assert out_shape == (batch_size, output_shape), f"Output shape is {out_shape}, but should be {(batch_size, output_shape)}"
    
    return nn_init, nn_apply
