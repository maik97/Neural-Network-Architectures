from stable_baselines3 import PPO

from utils.custom_sb3_ac_policy import CustomActorCriticPolicy


def test_with_ppo(
        env,
        network=None,
        network_kwargs=None,
        last_layer_dim_pi=64,
        last_layer_dim_vf=64,
        total_timesteps=100_000,
        test_timesteps=10_000
):

    policy_kwargs = dict(
        network=network,
        network_kwargs=network_kwargs,
        last_layer_dim_pi=last_layer_dim_pi,
        last_layer_dim_vf=last_layer_dim_vf,
    )

    model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=3e-4)

    print(model.policy)

    model.learn(total_timesteps)

    for i in range(test_timesteps):
        obs = env.reset()
        done = False
        episode_r = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            episode_r += rewards
        print('episode_r', episode_r)


def main():
    import gym

    test_with_ppo(
        env=gym.make('CartPole-v0'),
        network=None,
        network_kwargs=None,
        last_layer_dim_pi=64,
        last_layer_dim_vf=64,
    )

if __name__ == '__main__':
    main()
