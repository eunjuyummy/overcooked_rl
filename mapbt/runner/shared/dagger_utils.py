import time
import numpy as np
import torch

def train_model(model, X_train, y_train, save_path,
                num_epochs=5,
                learning_rate=1e-3,
                lambda_l2=1e-5,
                batch_size=32,
                exp_log=None,
                log_interval=10):

    # 1) Copy initial parameters to track changes
    initial_params = {}
    for net_idx, pi in enumerate(model.pis):
        for name, p in pi.named_parameters():
            initial_params[f"{net_idx}/{name}"] = p.detach().cpu().clone()

    criterion = torch.nn.MSELoss()
    optimizers = [
        torch.optim.SGD(model.pis[i].parameters(), lr=learning_rate, weight_decay=lambda_l2)
        for i in range(model.num_nets)
    ]

    device = model.device
    model.train()

    X_tensor = torch.stack([torch.from_numpy(x).float() for x in X_train], dim=0).to(device)
    X_tensor = X_tensor.div_(255.0) # normalize
    y_tensor = torch.stack([torch.from_numpy(y).float() for y in y_train], dim=0).to(device)

    n_samples = X_tensor.size(0)
    steps_per_epoch = max(1, n_samples // batch_size)

    t_start = time.time()
    for epoch in range(num_epochs):
        # Shuffle data randomly at each epoch
        idx = torch.randperm(n_samples, device=device)
        X_tensor = X_tensor[idx]
        y_tensor = y_tensor[idx]

        epoch_loss = 0.0

        # Train using mini-batches
        for i in range(0, n_samples, batch_size):
            batch_X = X_tensor[i: i + batch_size]
            batch_Y = y_tensor[i: i + batch_size]

            for net_idx in range(model.num_nets):
                preds = model.pis[net_idx](batch_X)
                loss = criterion(preds, batch_Y)
                epoch_loss += loss.item()

                optimizers[net_idx].zero_grad()
                loss.backward()
                optimizers[net_idx].step()

        # Optional logging
        if exp_log is not None and (epoch % log_interval) == 0:
            t_now = time.time()
            exp_log.scalar(
                is_train=True,
                data_set_size=n_samples,
                epoch=epoch,
                epoch_loss=epoch_loss / (steps_per_epoch * model.num_nets),
                ensemble_variance=model.variance(X_tensor.cpu().numpy()),
                epoch_time=(t_now - t_start)
            )
            t_start = t_now

    # 2) Compute and display how much each parameter changed after training
    diffs = {}
    for net_idx, pi in enumerate(model.pis):
        for name, p in pi.named_parameters():
            key = f"{net_idx}/{name}"
            delta = p.detach().cpu() - initial_params[key]
            diffs[key] = delta.norm().item()
    '''
    print("=== Parameter change norms after training ===")
    for k, v in diffs.items():
        print(f"{k:30s}: {v:.6f}")
    '''
    if exp_log is not None:
        exp_log.scalar(is_train=True, parameter_change=diffs)

    if exp_log is not None:
        exp_log.scalar(
            is_train=True,
            last_epoch_loss=epoch_loss / (steps_per_epoch * model.num_nets),
            total_sgd_epoch=num_epochs
        )

    model.save(save_path)

'''
def evaluation(env, model, evaluation_episode_num=30, exp_log=None):
    device = model.device
    model.eval()

    total_reward = 0.0
    total_cost = 0.0
    success_count = 0
    count_episode = 0
    velocity_list = []
    overtake_list = []

    state = env.reset()
    while count_episode < evaluation_episode_num:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = model.act(state_tensor.cpu().numpy().reshape(-1), i=-1)
        next_state, r, done, info = env.step(action.flatten())

        total_reward += r
        total_cost += info.get("native_cost", 0.0)
        velocity_list.append(info.get("velocity", 0.0))

        if done:
            overtake_list.append(info.get("overtake_vehicle_num", 0))
            count_episode += 1
            if info.get("arrive_dest", False):
                success_count += 1
            state = env.reset()
        else:
            state = next_state

    mean_reward = total_reward / count_episode
    mean_cost = total_cost / count_episode
    success_rate = success_count / count_episode
    mean_velocity = float(np.mean(velocity_list)) if velocity_list else 0.0
    mean_overtake = float(np.mean(overtake_list)) if overtake_list else 0.0

    results = {
        "mean_episode_reward": mean_reward,
        "mean_episode_cost": mean_cost,
        "mean_success_rate": success_rate,
        "mean_velocity": mean_velocity,
        "mean_overtake": mean_overtake
    }

    if exp_log is not None:
        exp_log.scalar(is_train=False, **results)

    return results
'''