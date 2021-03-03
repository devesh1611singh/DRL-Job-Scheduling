from gym.envs.registration import register

register(
    id='stowkoski-env-v0',
    entry_point='envs.environment:Environment',
)