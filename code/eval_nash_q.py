import warnings
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from my_highway_env import MyHighwayEnv
from surround_vehicle import SurroundVehicle

warnings.simplefilter("ignore")


def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):
    # Mess with this to change frame size
    plt.figure(
        figsize=(frames[0].shape[1] / 200.0, frames[0].shape[0] / 200.0), dpi=200
    )

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="Pillow", fps=25)


ev_policy = np.load("ev_policy.npy")

# Choose surround vehicle behaviour
sv = SurroundVehicle(type="aggressive")


env = MyHighwayEnv()

# Simulation of NashQ policy
obs = env.reset()

frames = []
nashQ_ev_history = []
sv_history = []
for _ in range(100):
    sv_action = sv.act(obs[1])
    ev_action = ev_policy[_][0 if len(sv.type) < 8 else 1]

    obs, reward, done, info = env.step((ev_action, sv_action))

    nashQ_ev_history.append(obs[0][0])
    sv_history.append(obs[1][0])

    frames.append(env.render(mode="rgb_array"))
save_frames_as_gif(frames, filename="nashQ_aggressive.gif")

# Simulation of rule base
obs = env.reset()

frames = []
rule_ev_history = []
for _ in range(100):
    sv_action = sv.act(obs[1])
    if _ < 10:
        ego_action = 1
    elif _ < 40:
        ego_action = 0

    obs, reward, done, info = env.step((ego_action, sv_action))

    rule_ev_history.append(obs[0][0])

    frames.append(env.render(mode="rgb_array"))
save_frames_as_gif(frames, filename="rule_aggressive.gif")

# Plot
time = np.arange(0, 5, 0.05)
nashQ_ev_y = [row[1] for row in nashQ_ev_history]
nashQ_ev_x = [row[0] for row in nashQ_ev_history]
nashQ_ev_vx = [row[2] for row in nashQ_ev_history]
nashQ_ev_vy = [row[3] for row in nashQ_ev_history]

rule_ev_y = [row[1] for row in rule_ev_history]
rule_ev_x = [row[0] for row in rule_ev_history]
rule_ev_vx = [row[2] for row in rule_ev_history]
rule_ev_vy = [row[3] for row in rule_ev_history]

sv_y = [row[1] for row in sv_history]
sv_x = [row[0] for row in sv_history]
sv_vx = [row[2] for row in sv_history]
sv_vy = [row[3] for row in sv_history]

plt.figure(figsize=(15, 7), dpi=300)
plt.plot(sv_x, sv_y, "-o")
plt.plot(nashQ_ev_x, nashQ_ev_y, "-o")
plt.plot(rule_ev_x, rule_ev_y, "-o")
plt.legend(["sv", "NQ_EV", "R_EV"])
plt.xlabel("Logitudinal Position")
plt.ylabel("Leteral Position")
plt.savefig("figures/aggressive_position.png")

plt.figure(figsize=(15, 7), dpi=300)
plt.plot(time, sv_vx, "-o")
plt.plot(time, nashQ_ev_vx, "-o")
plt.plot(time, rule_ev_vx, "-o")
plt.legend(["sv", "NQ_EV", "R_EV"])
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.savefig("figures/aggressive_velocity.png")
