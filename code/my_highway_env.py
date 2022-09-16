from turtle import color
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import LineType, StraightLane

LANES = 3
ANGLE = 0
START = 0
LENGHT = 200
SPEED_LIMIT = 30
SPEED_REWARD_RANGE = [10, 30]
COL_REWARD = -1
HIGH_SPEED_REWARD = 0
RIGHT_LANE_REWARD = 0
DURATION = 5.0


class MyHighwayEnv(AbstractEnv):
    """
    ACTIONS = {
        LANE_LEFT = 0
        IDLE = 1
        LANE_RIGHT = 2
        FASTER = 3
        SLOWER = 4
    }
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                        "vehicles_count": 2,
                        "features": ["x", "y", "vx", "vy"],
                        "absolute": True,
                    },
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                    },
                },
                "reward_speed_range": SPEED_REWARD_RANGE,
                "simulation_frequency": 20,
                "policy_frequency": 20,
                "centering_position": [0.3, 0.5],
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(LANES, speed_limit=SPEED_LIMIT),
            np_random=self.np_random,
            record_history=False,
        )

    def _create_vehicles(self) -> None:

        self.controlled_vehicles = []
        vehicle = Vehicle.create_random(self.road, speed=23, lane_id=1, spacing=0.3)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position,
            vehicle.heading,
            vehicle.speed,
            color=(0, 50, 200),
        )
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        vehicle = Vehicle.create_random(self.road, speed=28, lane_id=0, spacing=-0.35)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position,
            vehicle.heading,
            vehicle.speed,
            color=(200, 200, 0),
        )
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        reward = 0
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        if self.vehicle.crashed:
            reward = -1
        elif lane == 0:
            reward += 1

        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        return (
            self.vehicle.crashed
            or self.time >= DURATION
            or (False and not self.vehicle.on_road)
        )

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)
