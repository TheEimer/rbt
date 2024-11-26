from flashbax.buffers.prioritised_flat_buffer import PrioritisedTrajectoryBufferState
import numpy as np
import cv2
import gymnasium as gym
from arlbench.core.environments.autorl_env import Environment
from arlbench.core.environments.xland_env import XLandEnv
from arlbench.core.environments.gymnax_env import GymnaxEnv
from arlbench.core.wrappers.wrapper import Wrapper


def visualize_gym(
        observations: np.ndarray,
        arlbench_env: GymnaxEnv,
        n_rendered_frames: int
    ) -> list[np.ndarray]:
    """Visualize the buffer."""
    # Initialize the environment
    env = gym.make(arlbench_env.env_name, render_mode="rgb_array")
    env.reset()

    # Create a list to store rendered frames
    rendered_frames = []

    for observation in observations[n_rendered_frames: ]:
        if np.all(observation == 0):
            continue
        
        env.unwrapped.state = observation 
        frame = env.render() 
        rendered_frames.append(frame)

    env.close()

    return rendered_frames


def visualize_xland(
        _,
        env: XLandEnv,
        n_rendered_frames: int
    ) -> list[np.ndarray]:
    """Visualize the buffer."""
    rendered_frames = []

    for timestep in env.stored_timesteps[n_rendered_frames: ]:   
        frame = env._env.render(env.env_params, timestep)
        rendered_frames.append(frame)

    return rendered_frames


ENV_RENDERERS = {
    "GymnaxEnv": visualize_gym,
    "XLandEnv": visualize_xland
}


def visualize_buffer(
        buffer_state: PrioritisedTrajectoryBufferState,
        env: Environment | Wrapper,
        tag: str,
        n_rendered_frames: int = 0
    ) -> int:
    """Visualize the buffer."""
    if isinstance(env, Wrapper):
        env = env._env

    original_obs_shape = env.observation_space.shape
    observations = np.array(buffer_state.experience.obs).reshape(-1, *original_obs_shape)

    env_type = type(env).__name__

    render_func = ENV_RENDERERS[env_type]
    rendered_frames = render_func(observations, env, n_rendered_frames)

    canvas = None

    for frame in rendered_frames:
        frame = frame.astype(float)
        if canvas is None:
            canvas = frame
        else:
            canvas = cv2.addWeighted(canvas, 0.5, frame, 0.5, 0)  # Overlay

    # Normalize and save the final overlay as PNG
    canvas = (canvas / canvas.max() * 255).astype(np.uint8)
    cv2.imwrite(f"buffer_observations_{tag}.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return len(rendered_frames)
