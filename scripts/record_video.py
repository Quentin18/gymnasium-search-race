import argparse

import gymnasium as gym
from stable_baselines3 import PPO


def record_video(
    model_path: str,
    env_id: str,
    video_folder: str = "videos",
    opponent_path: str | None = None,
) -> None:
    env = gym.make(
        env_id,
        opponent_path=opponent_path,
        render_mode="rgb_array",
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda _: True,
        disable_logger=True,
    )

    model = PPO.load(path=model_path, env=env)

    observation, _info = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(observation=observation, deterministic=True)
        observation, _reward, terminated, truncated, _info = env.step(action)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a video of a trained agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        required=True,
        help="path to model file",
    )
    parser.add_argument(
        "--opponent-path",
        help="path to opponent model file",
    )
    parser.add_argument(
        "--env",
        default="gymnasium_search_race:gymnasium_search_race/MadPodRacing-v0",
        help="environment id",
    )
    parser.add_argument(
        "--video-folder",
        default="videos",
        help="path to videos folder",
    )
    args = parser.parse_args()

    record_video(
        model_path=args.path,
        env_id=args.env,
        video_folder=args.video_folder,
        opponent_path=args.opponent_path,
    )
