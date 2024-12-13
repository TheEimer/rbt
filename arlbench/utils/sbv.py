import jax.numpy as jnp
import numpy as np
from flashbax.buffers.train_val_trajectory_buffer import TrainValTrajectoryBufferState

import jax
import functools
from arlbench import AutoRLEnv

def compute_vbs(env: AutoRLEnv) -> float:
    # First, we extract the validation split from the buffer
    assert env._algorithm_state is not None
    assert isinstance(env._algorithm_state.buffer_state, TrainValTrajectoryBufferState)

    experience = env._algorithm_state.buffer_state.experience
    val_indices = env._algorithm_state.buffer_state.val_indices

    validation_data = {}
    for key in ["last_obs", "obs", "action", "done"]:
        validation_data[key] = np.array(experience[key][val_indices])

    print(validation_data.keys())
    exit()


    return 0
