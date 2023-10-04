'''
Yongcheng Ding

'''
from tensorforce.agents import TRPOAgent
from tensorforce.agents import PPOAgent
from tensorforce.agents import DuelingDQNAgent
from tensorforce.agents import DQNAgent
import DoubleENV
import numpy as np
from tensorforce.execution import Runner
import pickle
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

YY=DoubleENV.YY()#prepare the environment

#hidden layer
network_spec=[
        dict(type='dense',size=64,activation='relu',dropout=0.1),
        dict(type='dense',size=64,activation='relu',dropout=0.1),
        dict(type='dense',size=64,activation='relu',dropout=0.1),
        ]


# np.random.seed(1)

agent=PPOAgent(
        states=YY.states(),
        actions=YY.actions(),
        network=network_spec,
        max_episode_timesteps=YY.max_episode_timesteps(),
        learning_rate = 1*1e-4 ,
        #for pretrain 3*1e-3, for fine tuningh 1e-3
        batch_size = 128,
        saver=dict(directory='model_9')
        )


agent.initialize()
# agent.restore(directory='model_8',filename='agent-best')


runner=Runner(agent=agent,
              environment=YY,
              save_best_agent=True,
              evaluation_environment=YY
              )

runner.run(num_episodes=200000)

"""

with open('reward_1', 'wb') as fp:
    pickle.dump(runner.episode_rewards, fp)

import pickle

with open('reward_1','rb') as fp:
    pickle_1 = pickle.load(fp)
    
"""    
