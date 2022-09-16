import warnings
import numpy as np
from tqdm import tqdm

from my_highway_env import MyHighwayEnv
from surround_vehicle import SurroundVehicle
from NashQ_Agent import NashQAgent

warnings.simplefilter("ignore")

sv = SurroundVehicle(type="aggressive")

env = MyHighwayEnv()
obs = env.reset()

nashQ_agent = NashQAgent(
    environment=env,
    learning_rate=0.001,
    max_iter=100,
    discount_factor=0.95,
    decision_strategy="epsilon_greedy",
    epsilon=0.8,
    random_state=42,
)

for e in tqdm(range(1000)):
    if e % 2 == 0:
        sv = SurroundVehicle(type="aggressive")
    else:
        sv = SurroundVehicle(type="gentle")

    nashQ_agent.fit(return_history=False, sv=sv)
