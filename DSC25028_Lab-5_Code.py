import json
import os
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Taxi-v3 was renamed Taxi-v4 in newer Gymnasium releases (identical dynamics).
ENV_ID = "Taxi-v3" if "Taxi-v3" in gym.registry else "Taxi-v4"

# --------------------------------------------------------------------------
# 1. Hyper-parameters
# --------------------------------------------------------------------------
SEED = 42
NUM_EPISODES = 10_000        # training episodes
MAX_STEPS = 200              # Taxi-v3 default time limit (truncation)
ALPHA = 0.1                  # learning rate
GAMMA = 0.99                 # discount factor
EPS_START = 1.0              # initial exploration rate
EPS_MIN = 0.01               # minimum exploration rate
EPS_DECAY = 0.9990           # multiplicative decay applied after every episode
EVAL_EPISODES = 1_000        # greedy evaluation episodes
WINDOW = 100                 # moving-average window for plots

OUT_DIR = "results"
SHOT_DIR = os.path.join(OUT_DIR, "env_screenshots")
os.makedirs(SHOT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)


# --------------------------------------------------------------------------
# 2. Agent
# --------------------------------------------------------------------------
class QLearningAgent:
    """Tabular Q-learning agent with an epsilon-greedy behaviour policy."""

    def __init__(self, n_states, n_actions, alpha, gamma, eps, eps_min, eps_decay):
        self.q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay

    def act(self, state, greedy=False):
        if not greedy and rng.random() < self.eps:
            return int(rng.integers(self.n_actions))          # explore
        best = np.flatnonzero(self.q[state] == self.q[state].max())
        return int(rng.choice(best))                           # exploit (random tie-break)

    def update(self, s, a, r, s_next, terminated):
        # Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
        # Bootstrapping is switched off only on true termination (not truncation).
        target = r if terminated else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# --------------------------------------------------------------------------
# 3. Helpers: run episodes
# --------------------------------------------------------------------------
def run_episode(env, agent=None, policy="greedy", seed=None):
    """Run one evaluation episode. policy: 'greedy' (agent) or 'random'."""
    s, _ = env.reset(seed=seed)
    total, steps, success = 0.0, 0, False
    for _ in range(MAX_STEPS):
        a = env.action_space.sample() if policy == "random" else agent.act(s, greedy=True)
        s, r, terminated, truncated, _ = env.step(a)
        total += r
        steps += 1
        if terminated:                       # passenger delivered
            success = True
        if terminated or truncated:
            break
    return total, steps, success


def evaluate(env, agent, n, policy="greedy", seed0=10_000):
    rewards, steps, succ = [], [], []
    for i in range(n):
        R, T, S = run_episode(env, agent, policy, seed=seed0 + i)
        rewards.append(R); steps.append(T); succ.append(S)
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(succ) * 100),
    }


def moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


# --------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------
def train():
    env = gym.make(ENV_ID, max_episode_steps=MAX_STEPS)
    env.action_space.seed(SEED)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                           ALPHA, GAMMA, EPS_START, EPS_MIN, EPS_DECAY)

    ep_reward = np.zeros(NUM_EPISODES)
    ep_steps = np.zeros(NUM_EPISODES)
    ep_success = np.zeros(NUM_EPISODES)
    ep_eps = np.zeros(NUM_EPISODES)

    log("=" * 70)
    log(f" Q-LEARNING TRAINING  -  {ENV_ID}")
    log("=" * 70)
    log(f" Episodes={NUM_EPISODES}  alpha={ALPHA}  gamma={GAMMA}  "
        f"eps: {EPS_START}->{EPS_MIN} (decay {EPS_DECAY})  max_steps={MAX_STEPS}  seed={SEED}")
    log(f" |S| = {env.observation_space.n}   |A| = {env.action_space.n}")
    log("-" * 70)
    log(f" {'Episode':>8} | {'Avg Reward(100)':>15} | {'Success%(100)':>13} | {'Avg Steps(100)':>14} | {'Epsilon':>7}")
    log("-" * 70)

    t0 = time.time()
    for ep in range(NUM_EPISODES):
        s, _ = env.reset(seed=SEED + ep)
        total, steps, success = 0.0, 0, 0
        while True:
            a = agent.act(s)
            s2, r, terminated, truncated, _ = env.step(a)
            agent.update(s, a, r, s2, terminated)
            s, total, steps = s2, total + r, steps + 1
            if terminated:
                success = 1
            if terminated or truncated:
                break
        ep_eps[ep] = agent.eps
        agent.decay_epsilon()
        ep_reward[ep], ep_steps[ep], ep_success[ep] = total, steps, success

        if (ep + 1) % 500 == 0 or ep == 0:
            lo = max(0, ep + 1 - WINDOW)
            log(f" {ep + 1:>8} | {ep_reward[lo:ep+1].mean():>15.2f} | "
                f"{ep_success[lo:ep+1].mean()*100:>12.1f}% | {ep_steps[lo:ep+1].mean():>14.1f} | {ep_eps[ep]:>7.3f}")
    train_time = time.time() - t0
    log("-" * 70)
    log(f" Training finished in {train_time:.1f} s")
    env.close()
    return agent, ep_reward, ep_steps, ep_success, ep_eps, train_time


# --------------------------------------------------------------------------
# 5. Plots
# --------------------------------------------------------------------------
def make_plots(ep_reward, ep_steps, ep_success, ep_eps, ev_trained, ev_random):
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    x = np.arange(1, NUM_EPISODES + 1)
    xm = np.arange(WINDOW, NUM_EPISODES + 1)

    # (a) Learning curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, ep_reward, color="#9ecae1", lw=0.5, alpha=0.7, label="Episode reward")
    ax.plot(xm, moving_avg(ep_reward, WINDOW), color="#08519c", lw=2,
            label=f"Moving average ({WINDOW} episodes)")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Total reward per episode")
    ax.set_title("Q-Learning on Taxi: Reward vs. Episodes")
    ax.set_ylim(-800, 30); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/learning_curve.png", dpi=200); plt.close(fig)

    # (b) Success rate
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xm, moving_avg(ep_success, WINDOW) * 100, color="#238b45", lw=2)
    ax.set_xlabel("Training episode"); ax.set_ylabel(f"Success rate (%) - {WINDOW}-episode window")
    ax.set_title("Task Completion (Successful Drop-off) Rate during Training")
    ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/success_rate_curve.png", dpi=200); plt.close(fig)

    # (c) Steps per episode
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xm, moving_avg(ep_steps, WINDOW), color="#d94801", lw=2)
    ax.set_xlabel("Training episode"); ax.set_ylabel(f"Steps per episode ({WINDOW}-ep. average)")
    ax.set_title("Episode Length during Training")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/steps_curve.png", dpi=200); plt.close(fig)

    # (d) Epsilon decay
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, ep_eps, color="#6a51a3", lw=2)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (Epsilon) Decay")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/epsilon_decay.png", dpi=200); plt.close(fig)

    # (e) Evaluation comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    labels = ["Random\nagent", "Trained\nQ-agent"]
    cols = ["#bdbdbd", "#08519c"]
    for ax, key, ttl in zip(axes, ["avg_reward", "avg_steps", "success_rate"],
                            ["Average reward", "Average steps", "Success rate (%)"]):
        vals = [ev_random[key], ev_trained[key]]
        bars = ax.bar(labels, vals, color=cols)
        ax.set_title(ttl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontweight="bold")
    fig.suptitle(f"Evaluation over {EVAL_EPISODES} episodes: Random vs. Trained agent")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/evaluation_comparison.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------
# 6. Environment screenshots (rgb_array frames)
# --------------------------------------------------------------------------
def capture_screenshots(agent):
    env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=MAX_STEPS)

    def save(frame, name):
        plt.figure(figsize=(6, 4.4)); plt.imshow(frame); plt.axis("off")
        plt.tight_layout(pad=0.2); plt.savefig(f"{SHOT_DIR}/{name}.png", dpi=150); plt.close()

    # Initial state
    s, _ = env.reset(seed=7)
    save(env.render(), "01_initial_state")

    # Trained agent trajectory: save pickup + dropoff frames
    s, _ = env.reset(seed=7)
    frames, picked = [env.render()], None
    for t in range(MAX_STEPS):
        a = agent.act(s, greedy=True)
        s, r, term, trunc, _ = env.step(a)
        frames.append(env.render())
        if r == -10:
            pass
        if a == 4 and r == -1 and picked is None:      # successful pickup
            picked = len(frames) - 1
        if term or trunc:
            break
    save(frames[picked if picked else len(frames) // 2], "02_after_pickup")
    save(frames[len(frames) // 2], "03_en_route")
    save(frames[-1], "04_after_dropoff")

    # Composite montage: 6 evenly spaced frames from the trained trajectory
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.8))
    for ax, i in zip(axs.ravel(), idx):
        ax.imshow(frames[i]); ax.axis("off"); ax.set_title(f"Step {i}")
    fig.suptitle("Trained agent: one complete trip (pick-up -> drop-off)")
    fig.tight_layout(); fig.savefig(f"{SHOT_DIR}/05_trained_trajectory_montage.png", dpi=150); plt.close(fig)
    env.close()
    return len(frames) - 1


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    agent, ep_reward, ep_steps, ep_success, ep_eps, train_time = train()

    eval_env = gym.make(ENV_ID, max_episode_steps=MAX_STEPS)
    eval_env.action_space.seed(SEED)
    ev_trained = evaluate(eval_env, agent, EVAL_EPISODES, "greedy")
    ev_random = evaluate(eval_env, agent, EVAL_EPISODES, "random")
    eval_env.close()

    # Phase-wise training statistics (every 2000 episodes)
    phases = []
    for lo in range(0, NUM_EPISODES, 2000):
        hi = lo + 2000
        phases.append({
            "episodes": f"{lo+1}-{hi}",
            "avg_reward": float(ep_reward[lo:hi].mean()),
            "success_rate": float(ep_success[lo:hi].mean() * 100),
            "avg_steps": float(ep_steps[lo:hi].mean()),
        })

    # Episodes needed to first reach a 100-episode avg reward > 0 / success >= 95%
    ma_r = moving_avg(ep_reward, WINDOW); ma_s = moving_avg(ep_success, WINDOW) * 100
    conv_ep = int(np.argmax(ma_r > 0) + WINDOW) if (ma_r > 0).any() else None
    s95_ep = int(np.argmax(ma_s >= 95) + WINDOW) if (ma_s >= 95).any() else None

    log("\n" + "=" * 70)
    log(" FINAL RESULTS")
    log("=" * 70)
    log(f" Training episodes                 : {NUM_EPISODES}")
    log(f" Training time                     : {train_time:.1f} s")
    log(f" First 100 episodes - avg reward   : {ep_reward[:100].mean():.2f}")
    log(f" Last 100 episodes  - avg reward   : {ep_reward[-100:].mean():.2f}  (epsilon-greedy)")
    log(f" Last 100 episodes  - success rate : {ep_success[-100:].mean()*100:.1f}%")
    log(f" 100-ep avg reward first > 0 at    : episode {conv_ep}")
    log(f" 100-ep success >= 95% first at    : episode {s95_ep}")
    log("-" * 70)
    log(f" GREEDY EVALUATION ({EVAL_EPISODES} unseen episodes)")
    log(f"   Trained agent: reward {ev_trained['avg_reward']:.2f} +/- {ev_trained['std_reward']:.2f}, "
        f"steps {ev_trained['avg_steps']:.2f}, success {ev_trained['success_rate']:.1f}%")
    log(f"   Random agent : reward {ev_random['avg_reward']:.2f} +/- {ev_random['std_reward']:.2f}, "
        f"steps {ev_random['avg_steps']:.2f}, success {ev_random['success_rate']:.1f}%")
    log("=" * 70)

    make_plots(ep_reward, ep_steps, ep_success, ep_eps, ev_trained, ev_random)
    trip_len = capture_screenshots(agent)
    log(f" Demonstration trip (seed 7) length : {trip_len} steps")

    np.save(f"{OUT_DIR}/q_table.npy", agent.q)
    np.save(f"{OUT_DIR}/episode_rewards.npy", ep_reward)
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({
            "hyperparameters": dict(episodes=NUM_EPISODES, alpha=ALPHA, gamma=GAMMA,
                                    eps_start=EPS_START, eps_min=EPS_MIN, eps_decay=EPS_DECAY,
                                    max_steps=MAX_STEPS, seed=SEED),
            "training_time_s": train_time,
            "first100_avg_reward": float(ep_reward[:100].mean()),
            "last100_avg_reward": float(ep_reward[-100:].mean()),
            "last100_success_rate": float(ep_success[-100:].mean() * 100),
            "episode_reward_first_positive_ma": conv_ep,
            "episode_success_ma_ge_95": s95_ep,
            "phases": phases,
            "evaluation_trained": ev_trained,
            "evaluation_random": ev_random,
            "demo_trip_steps": trip_len,
        }, f, indent=2)
    with open(f"{OUT_DIR}/training_log.txt", "w") as f:
        f.write("\n".join(LOG_LINES))
    print(f"\nAll outputs saved in ./{OUT_DIR}/")
