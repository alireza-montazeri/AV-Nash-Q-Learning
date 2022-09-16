class SurroundVehicle:
    def __init__(self, type):
        self.type = type

    def act(self, obs):
        if obs[1][1] < 0.47:
            if self.type == "aggressive":
                return 3
            elif self.type == "gentle":
                return 4
        else:
            return 1
