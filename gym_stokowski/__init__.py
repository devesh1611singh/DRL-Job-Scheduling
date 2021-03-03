from gym.envs.registration import register

'''register(
    id='stowkoski-main-v0',
    entry_point='gym_stowkoski.envs:StowkoskiMain',
)'''
register(
    id='stowkoski-env-v0',
    entry_point='gym_stowkoski.envs:Environment',
)