import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time


class PolicyGradientAgent:
    def __init__(self, lr=0.001, gamma=0.99, n_actions=4, input_dims=8):
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.model = self.build_model(input_dims, n_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def build_model(self, input_dims, n_actions):
        model = tf.keras.Sequential([
            layers.Input(shape=(input_dims,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_actions, activation='softmax')
        ])
        return model

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.model(state).numpy()[0]
        action = np.random.choice(self.n_actions, p=probabilities)
        return action

    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        G = np.zeros_like(self.reward_memory, dtype=np.float32)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        G = (G - np.mean(G)) / (np.std(G) + 1e-8)

        with tf.GradientTape() as tape:
            loss = 0
            for g, state, action in zip(G, self.state_memory, self.action_memory):
                state = state[np.newaxis, :]
                probs = self.model(state, training=True)
                action_prob = probs[0, action]
                log_prob = tf.math.log(action_prob + 1e-8)
                loss -= g * log_prob
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


if __name__ == "__main__":
    n_episodes = 3000
    scores = []

    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    agent = PolicyGradientAgent(lr=0.001, gamma=0.99,
                                input_dims=8,  # LunarLander-v2 has 8-dim state
                                n_actions=4)

    for i in range(1, n_episodes + 1):
        record_video = (i == 1 or i == n_episodes)

        if record_video:
            env = gym.make("LunarLander-v2", render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_dir,
                                           episode_trigger=lambda ep: True,
                                           name_prefix=f"episode_{i}")
        else:
            env = gym.make("LunarLander-v2")

        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward)
            state = new_state
            score += reward

        env.close()
        agent.learn()
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print(f"Episode {i} | Score: {score:.2f} | Avg (last 100): {avg_score:.2f}")
        if record_video:
            print(f"🎥 Video for Episode {i} saved in '{video_dir}'")
   