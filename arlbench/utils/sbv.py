import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from flashbax.buffers.prioritised_trajectory_buffer import PrioritisedTrajectoryBufferState
from arlbench.core.algorithms.prioritised_item_buffer import (
    make_prioritised_item_buffer,
)
import jax
import functools
from arlbench.core.algorithms.buffers import uniform_sample
from arlbench.core.algorithms.common import TimeStep

def split_buffer(
        buffer_state: PrioritisedTrajectoryBufferState,
        random_state: int,
        hpo_config: dict,
        validation_size: float = 0.2,
    ) -> tuple[PrioritisedTrajectoryBufferState, np.ndarray]:
    """Split buffer into train and validation sets."""
    buffer_length = int(buffer_state.experience.obs.shape[1] if buffer_state.is_full else buffer_state.current_index)

    indices = np.arange(buffer_length)
    train_indices, validation_indices = train_test_split(
        indices, test_size=validation_size, random_state=random_state
    )

    train_buffer = make_prioritised_item_buffer(
        max_length=len(train_indices),
        min_length=hpo_config["buffer_batch_size"],
        sample_batch_size=hpo_config["buffer_batch_size"],
        add_sequences=False,
        priority_exponent=hpo_config["buffer_alpha"],
        device="cpu",
    )

    if hpo_config["buffer_prio_sampling"] is False:
        sample_fn = functools.partial(
            uniform_sample,
            batch_size=hpo_config["buffer_batch_size"],
            sequence_length=1,
            period=1,
        )
        train_buffer = train_buffer.replace(sample=sample_fn)

    def get_obs(idx: int) -> TimeStep:
        return TimeStep(
            last_obs=buffer_state.experience.last_obs[idx],
            obs=buffer_state.experience.obs[idx],
            action=buffer_state.experience.action[idx],
            reward=buffer_state.experience.reward[idx],
            done=buffer_state.experience.reward[idx],
        )

    _timestep = get_obs(0)
    train_buffer_state = train_buffer.init(_timestep)

    for idx in train_indices:
        timestep = get_obs(idx)
        train_buffer_state = train_buffer.add(train_buffer_state, timestep)

    print(train_buffer_state)

    exit()