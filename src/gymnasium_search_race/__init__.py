from gymnasium.envs.registration import register

register(
    id="gymnasium_search_race/SearchRace-v2",
    entry_point="gymnasium_search_race.envs:SearchRaceEnv",
    max_episode_steps=600,
)

register(
    id="gymnasium_search_race/SearchRaceDiscrete-v2",
    entry_point="gymnasium_search_race.envs:SearchRaceDiscreteEnv",
    max_episode_steps=600,
)

register(
    id="gymnasium_search_race/MadPodRacing-v1",
    entry_point="gymnasium_search_race.envs:MadPodRacingEnv",
    max_episode_steps=600,
)

register(
    id="gymnasium_search_race/MadPodRacingBlocker-v1",
    entry_point="gymnasium_search_race.envs:MadPodRacingBlockerEnv",
    max_episode_steps=600,
)

register(
    id="gymnasium_search_race/MadPodRacingDiscrete-v1",
    entry_point="gymnasium_search_race.envs:MadPodRacingDiscreteEnv",
    max_episode_steps=600,
)
