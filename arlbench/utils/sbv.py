import jax.numpy as jnp
import numpy as np
from flashbax.buffers.train_val_trajectory_buffer import TrainValTrajectoryBuffer

import jax
import functools
from arlbench import AutoRLEnv

def compute_vbs(env: AutoRLEnv) -> float:
    # First, we extract the validation split from the buffer
    assert env._algorithm_state is not None
    buffer_state = env._algorithm_state.buffer_state

    print(buffer_state)
    exit()


    return 0
