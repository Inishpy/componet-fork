import numpy as np


"""
| Difficulty    | Example Tasks (from that evaluation)                                                                                                                                                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Easy**      | Button-Press, Button-Press-Topdown, Button-Press-Wall, Coffee-Button, Dial-Turn, Door-Close, Door-Open, Drawer-Open/Close, Faucet-Open/Close, Handle-Press (various), Plate-Slide (various), Reach, Window-Open/Close, Peg-Unplug-Side, etc. ([alpha-ollama.hf-mirror.com][1]) |
| **Medium**    | Basketball, Bin-Picking, Box-Close, Coffee-Pull/Push, Hammer, Peg-Insert-Side, Push-Wall, Soccer, Sweep, Sweep-Into, etc. ([alpha-ollama.hf-mirror.com][1])                                                                                                                    |
| **Hard**      | Assembly, Hand-Insert, Pick-Out-Of-Hole, Pick-Place, Push, Push-Back, etc. ([alpha-ollama.hf-mirror.com][1])                                                                                                                                                                   |
| **Very Hard** | Shelf-Place, Disassemble, Stick-Pull, Stick-Push, Pick-Place-Wall, etc. ([alpha-ollama.hf-mirror.com][1])                                                                                                                                                                      |

[1]: https://alpha-ollama.hf-mirror.com/RoboHP/metaworld_mt50_act?utm_source=chatgpt.com "RoboHP/metaworld_mt50_act · Hugging Face"

"""

single_tasks = [# Easy tasks
    "button-press-v2",
    "button-press-topdown-v2",
    "button-press-wall-v2",
    "coffee-button-v2",
    "dial-turn-v2",
    "door-close-v2",
    "door-open-v2",
    "drawer-open-v2",
    "drawer-close-v2",
    "faucet-open-v2",
    "faucet-close-v2",
    "handle-press-v2",
    "handle-press-side-v2",
    "handle-pull-v2",
    "handle-pull-side-v2",
    "plate-slide-v2",
    "plate-slide-back-v2",
    "plate-slide-side-v2",
    "reach-v2",
    "window-open-v2",
    "window-close-v2"]

# single_tasks = [
#     "hammer-v2",
#     "push-wall-v2",
#     "faucet-close-v2",
#     "push-back-v2",
#     "stick-pull-v2",
#     "handle-press-side-v2",
#     "push-v2",
#     "shelf-place-v2",
#     "window-close-v2",
#     "peg-unplug-side-v2",
# ]

tasks = single_tasks + single_tasks


def get_task_name(task_id):
    return tasks[task_id]


def get_task(task_id, render=False):
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

    name = tasks[task_id] + "-goal-observable"

    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name]
    env = env_cls(seed=np.random.randint(0, 1024))

    if render:
        env.render_mode = "human"
    env._freeze_rand_vec = False

    return env


if __name__ == "__main__":
    env = get_task(0, render=True)

    for _ in range(200):
        obs, _ = env.reset()  # reset environment
        a = env.action_space.sample()  # sample an action

        # step the environment with the sampled random action
        obs, reward, terminated, truncated, info = env.step(a)

        if terminated:
            break
