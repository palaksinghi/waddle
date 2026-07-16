import os        #saving the checkpoints
import argparse     #command line arguments
import numpy as np   #calculations
import torch   # for neural networks
import torch.nn as nn 
import gymnasium as gym  #for environment
import wandb
# Networks
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):   #helper fn which creates the neural network
    layers = []                                            #creates the layers
    for i in range(len(sizes) - 1):                         #for loop 
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)                           #all the activation fn --> inear tanh relu
#gaussian policy act as an actor
#Gaussian policy will observe the nn and will take certain action on the basis will get the mean action 
#--> gaussian distribution and then the sample action step
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):  #constructor 
        super().__init__()
        self.mu_net = mlp([obs_dim, *hidden, act_dim])       #mean action   
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))  #deviation 

#now the below is the exploration part -->randomness
    def forward(self, obs):
        mu = self.mu_net(obs)                  
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)
    
#logpi(a/s) for a given action what would be the state lso known as policy
    def log_prob(self, obs, act):
        dist = self.forward(obs)
        return dist.log_prob(act).sum(axis=-1)

#randomnly samples as an action
    def sample(self, obs):
        dist = self.forward(obs)
        act = dist.sample()
        logp = dist.log_prob(act).sum(axis=-1)
        return act, logp

#we need to create the class of value network
class ValueNetwork(nn.Module):

    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        self.v_net = mlp([obs_dim, *hidden, 1])

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)
# Utility: flatten / unflatten parameters, gradients
def get_flat_params(model): #flatten parameters-->In trpo , layer by layer value does not matter,weights will be added of all layer
    #and this will matter
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):  #set flat parameters-->does the reverse of get falt parameters by doing so we get total se per layer wala network
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_params[idx:idx + n].view_as(p))
        idx += n


def get_flat_grad(loss, model, create_graph=False, retain_graph=False): #computes the policy gradient in huge way rather than in single layer
    grads = torch.autograd.grad(
        loss, model.parameters(), create_graph=create_graph, retain_graph=retain_graph
    )
    return torch.cat([g.contiguous().view(-1) for g in grads])

# GAE

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    values_ext = np.append(values, last_value)
    for t in reversed(range(T)):    #TD error
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae        #estimates thefutire td errror
        adv[t] = last_gae
    returns = adv + values
    return adv, returns

# Rollout buffer / collection--> works to store everything for one episode

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.logps, self.values = [], [], []

    def clear(self):
        self.__init__()

def collect_trajectories(env, policy, value_net, batch_size, gamma, lam, device):   #data collection
    all_states, all_actions, all_logps = [], [], []
    all_advantages, all_returns = [], []
    collected_steps = 0
    ep_rewards_log = []

    while collected_steps < batch_size:
        buf = RolloutBuffer()
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t = policy.sample(state_t)
                value_t = value_net(state_t)

            action = action_t.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buf.states.append(state)
            buf.actions.append(action)
            buf.rewards.append(reward)
            buf.dones.append(float(terminated))  # bootstrap on truncation, not on terminal
            buf.logps.append(logp_t.item())
            buf.values.append(value_t.item())

            state = next_state
            ep_reward += reward
            collected_steps += 1

        # bootstrap value for truncated episodes (0 if truly terminal)
        if terminated:
            last_value = 0.0
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                last_value = value_net(state_t).item()

        rewards = np.array(buf.rewards, dtype=np.float32)
        values = np.array(buf.values, dtype=np.float32)
        dones = np.array(buf.dones, dtype=np.float32)

        adv, ret = compute_gae(rewards, values, dones, last_value, gamma, lam)

        all_states.extend(buf.states)
        all_actions.extend(buf.actions)
        all_logps.extend(buf.logps)
        all_advantages.extend(adv.tolist())
        all_returns.extend(ret.tolist())
        ep_rewards_log.append(ep_reward)

    data = dict(
        states=torch.as_tensor(np.array(all_states), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.array(all_actions), dtype=torch.float32, device=device),
        logps=torch.as_tensor(np.array(all_logps), dtype=torch.float32, device=device),
        advantages=torch.as_tensor(np.array(all_advantages), dtype=torch.float32, device=device),
        returns=torch.as_tensor(np.array(all_returns), dtype=torch.float32, device=device),
    )
    return data, ep_rewards_log


# TRPO core: KL divergence, FVP, conjugate gradient, line search
#KL--> difference between the new and old policy
def gaussian_kl(policy, states, old_mu, old_std):
    dist = policy(states)
    mu, std = dist.mean, dist.stddev
    var_old = old_std.pow(2)
    var_new = std.pow(2)
    kl = (torch.log(std / old_std) +
          (var_old + (old_mu - mu).pow(2)) / (2.0 * var_new) - 0.5)
    return kl.sum(-1).mean()

#size of the fisher vector product=no.of parameters*no.of parameters
def fisher_vector_product(policy, states, vector, damping=1e-2):
    with torch.no_grad():
        dist = policy(states)
        old_mu, old_std = dist.mean, dist.stddev

    kl = gaussian_kl(policy, states, old_mu, old_std)
    # NOTE: create_graph=True builds a NEW graph for the gradient itself, but that
    # new graph still depends on the saved tensors from the ORIGINAL forward pass
    # (the policy(states) call inside gaussian_kl). If retain_graph is left False,
    # those original buffers get freed right after this call, and the very next
    # get_flat_grad() (for the Hessian-vector product) fails with
    # "Trying to backward through the graph a second time". So we must retain it here.
    grad_kl = get_flat_grad(kl, policy, create_graph=True, retain_graph=True)

    grad_vector_product = (grad_kl * vector).sum()
    hvp = get_flat_grad(grad_vector_product, policy, retain_graph=True)
    return hvp + damping * vector


def conjugate_gradient(fvp_fn, b, cg_iters=10, tol=1e-10):
    """Solves Fx = b for x using conjugate gradient, given a function that
    computes the Fisher-vector product F * v."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_old = torch.dot(r, r)

    for _ in range(cg_iters):
        Fp = fvp_fn(p)
        alpha = r_dot_old / (torch.dot(p, Fp) + 1e-10)
        x += alpha * p
        r -= alpha * Fp
        r_dot_new = torch.dot(r, r)
        if r_dot_new < tol:
            break
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x

#loss=ratio*advantage
def surrogate_loss(policy, states, actions, advantages, old_logp):  #new/old policy  -->tries to max this 
    logp = policy.log_prob(states, actions)
    ratio = torch.exp(logp - old_logp)
    return (ratio * advantages).mean()


def trpo_step(policy, states, actions, advantages, old_logp,
              max_kl=0.01, cg_iters=10, damping=1e-2,
              backtrack_coeff=0.8, backtrack_iters=10):

    # 1. Policy gradient of the surrogate objective
    loss = surrogate_loss(policy, states, actions, advantages, old_logp)
    grads = get_flat_grad(loss, policy, retain_graph=True)

    # 2. Conjugate gradient to solve F x = g  ->  x = F^{-1} g (search direction)
    fvp_fn = lambda v: fisher_vector_product(policy, states, v, damping)
    step_dir = conjugate_gradient(fvp_fn, grads, cg_iters=cg_iters)

    # 3. Maximum step size satisfying the KL constraint:
    #    0.5 * step^T F step = max_kl  =>  step = sqrt(2*max_kl / (x^T F x)) * x
    shs = 0.5 * torch.dot(step_dir, fvp_fn(step_dir))
    step_size = torch.sqrt(max_kl / (shs + 1e-10))
    full_step = step_size * step_dir

    expected_improve = torch.dot(grads, full_step)

    # 4. Backtracking line search -->if none code satisfies then it will store the old parameters
    old_params = get_flat_params(policy)
    with torch.no_grad():
        old_dist = policy(states)
        old_mu, old_std = old_dist.mean, old_dist.stddev

    success = False
    final_step_frac = 0.0
    final_kl = 0.0
    for i in range(backtrack_iters):
        frac = backtrack_coeff ** i
        new_params = old_params + frac * full_step
        set_flat_params(policy, new_params)

        with torch.no_grad():
            new_loss = surrogate_loss(policy, states, actions, advantages, old_logp)
            kl = gaussian_kl(policy, states, old_mu, old_std)

        improve = new_loss - loss
        if kl.item() <= max_kl and improve.item() > 0:
            success = True
            final_step_frac = frac
            final_kl = kl.item()
            break

    if not success:
        # No improving step found within trust region: revert to old parameters.
        set_flat_params(policy, old_params)
        final_kl = 0.0

    return {
        "surrogate_loss": loss.item(),
        "expected_improve": expected_improve.item(),
        "line_search_success": success,
        "step_frac": final_step_frac,
        "kl": final_kl,
    }


def train_value_net(value_net, optimizer, states, returns, epochs=5, batch_size=64):
#Train for several episode to min (V(s) - Return)^2
    n = states.shape[0]
    losses = []     #by using optimizer -->adam
    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b_idx = idx[start:start + batch_size]
            pred = value_net(states[b_idx])
            loss = ((pred - returns[b_idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))

# Training loop

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    wandb.init(
    project="humanoid",
    name="run1",
    config=vars(args)| {
        "env":"Humanoid-v5",
        "algorithm":"trpo",
    }
    )
    env = gym.make("Humanoid-v5")   #create env,creates actor,critic
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.value_lr)

    best_reward = -np.inf

    for iteration in range(1, args.iterations + 1):
        #  collect trajectories 
        data, ep_rewards = collect_trajectories(
            env, policy, value_net,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lam=args.lam,
            device=device,
        )

        # normalize advantages
        adv = data["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data["advantages"] = adv

        #TRPO policy update
        trpo_info = trpo_step(
            policy,
            data["states"], data["actions"], data["advantages"], data["logps"],
            max_kl=args.max_kl,
            cg_iters=args.cg_iters,
            damping=args.damping,
            backtrack_coeff=args.backtrack_coeff,
            backtrack_iters=args.backtrack_iters
        )

        # value network training
        value_loss = train_value_net(
            value_net, value_optimizer, data["states"], data["returns"],
            epochs=args.value_epochs,
            batch_size=args.value_batch_size
        )
        mean_reward = float(np.mean(ep_rewards))
        #in wandb graph -->which graph will be seen is mentioned below
        wandb.log({
            "Iteration":iteration,
            "Episode Reward": mean_reward,
            "Policy Loss": trpo_info["surrogate_loss"],
            "Value Loss":value_loss,
            "KL Divergence": trpo_info["kl"],
        })

        # logging / checkpointing-->whichever will be printing on the terminal
        print(
            f"[iter{iteration:4d}] "
            f"mean_reward={mean_reward:8.2f}  "
            f"n_ep={len(ep_rewards):3d}  "
            f"surrogate={trpo_info['surrogate_loss']:.5f}  "
            f"line_search_ok={trpo_info['line_search_success']}  "
            f"step_frac={trpo_info['step_frac']:.3f}  "
            f"value_loss={value_loss:.3f}"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(policy.state_dict(), os.path.join(args.checkpoint_dir, "best_policy.pt"))
            torch.save(value_net.state_dict(), os.path.join(args.checkpoint_dir, "best_value.pt"))
            print(f"  -> new best reward {best_reward:.2f}, checkpoint saved.")

        if iteration % args.save_every == 0:
            torch.save(policy.state_dict(), os.path.join(args.checkpoint_dir, "latest_policy.pt"))
            torch.save(value_net.state_dict(), os.path.join(args.checkpoint_dir, "latest_value.pt"))

    env.close()
    wandb.finish()
    print("Training complete.")

# CLI

def build_argparser():
    p = argparse.ArgumentParser(description="TRPO training on Humanoid-v5 (single environment)")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--iterations", type=int, default=800)          #500-->800
    p.add_argument("--batch_size", type=int, default=10000, help="min env steps collected per iteration")  #4000-->10000
    p.add_argument("--gamma", type=float, default=0.99)       #same
    p.add_argument("--lam", type=float, default=0.97, help="GAE lambda")      #same
    p.add_argument("--max_kl", type=float, default=0.01, help="KL trust region (delta)")     #same
    p.add_argument("--cg_iters", type=int, default=10)   
    p.add_argument("--damping", type=float, default=1e-2)
    p.add_argument("--backtrack_coeff", type=float, default=0.8)
    p.add_argument("--backtrack_iters", type=int, default=10)
    p.add_argument("--value_lr", type=float, default=1e-3)
    p.add_argument("--value_epochs", type=int, default=5)
    p.add_argument("--value_batch_size", type=int, default=64)
    p.add_argument("--save_every", type=int, default=10)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)