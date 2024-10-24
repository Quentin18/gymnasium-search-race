from gymnasium.envs.registration import register

register(
    id="gymnasium_search_race/SearchRace-v1",
    entry_point="gymnasium_search_race.envs:SearchRaceEnv",
    max_episode_steps=600,
)
