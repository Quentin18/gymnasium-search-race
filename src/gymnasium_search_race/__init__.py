from gymnasium.envs.registration import register

register(
    id="gymnasium_search_race/SearchRace-v1",
    entry_point="gymnasium_search_race.envs:SearchRaceEnv",
    max_episode_steps=600,
)

register(
    id="gymnasium_search_race/SearchRaceDiscrete-v1",
    entry_point="gymnasium_search_race.envs:SearchRaceDiscreteEnv",
    max_episode_steps=600,
)

register(
    id="gymnasium_search_race/MadPodRacing-v0",
    entry_point="gymnasium_search_race.envs:MadPodRacingEnv",
    max_episode_steps=600,
)

register(
    id="gymnasium_search_race/MadPodRacingDiscrete-v0",
    entry_point="gymnasium_search_race.envs:MadPodRacingDiscreteEnv",
    max_episode_steps=600,
)
