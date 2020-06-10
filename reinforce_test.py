###################balancing cartpole using reinforcement learning#############################
import gym
env=gym.make('CartPole-v0')# taking cartpole file from gym library
#env.reset()
#for _ in range(100):
#      env.render()
#      env.step(env.action_space.sample())

for _ in range(20):
      observation=env.reset() #reset to start
      for i in range(100):
            env.render()
            print(observation)
            action=env.action_space.sample()
            observation,reward,done,info=env.step(action)
            if done:
                  print("epispde finished after {}timestamp".format(i+1))
                  break