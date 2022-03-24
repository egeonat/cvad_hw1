import torch


def polyak_update(net, target_net, polyak):
    for p, p_targ in zip(net.parameters(), target_net.parameters()):
        p_targ.data.mul_(polyak)
        p_targ.data.add_((1 - polyak) * p.data)


def generate_noisy_action_tensor(action_tensor, action_space, stddev, noise_clip):
    bsize = action_tensor.shape[0]
    a_dim = action_tensor.shape[1]
    action_space = torch.tensor(action_space).cuda()
    noise = torch.clamp(torch.randn((bsize, a_dim)) * stddev, -noise_clip, noise_clip).cuda()
    noisy_action_tensor = torch.clamp(
        action_tensor + noise, min=action_space[:, 0], max=action_space[:, 1])
    return noisy_action_tensor


def get_q_params(q_nets):
    param_list = list(q_nets[0].parameters()) + list(q_nets[1].parameters())
    return param_list
