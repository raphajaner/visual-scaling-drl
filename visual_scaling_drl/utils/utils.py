"""Core utilities for agent save/load, timing, network summary, and latency profiling."""

import os
import time
import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchinfo import summary
from torch_pruning.utils.benchmark import measure_latency
from torch.distributions.categorical import Categorical


def save_agent_to_wandb(config, agent, optimizer, obs_rms, return_rms, metadata={}, aliases=["latest"],
                        full_model=True):
    """Save the state_dict of the agent to Weights & Biases."""
    model_dict = {
        'obs_rms': obs_rms,
        'return_rms': return_rms,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'config': config
    }
    if full_model:
        model_dict['model'] = agent
    else:
        model_dict['model_state_dict'] = agent.state_dict()

    torch.save(model_dict, f'{wandb.run.dir}/agent.pt')

    model_artifact = wandb.Artifact(
        f"agent-{wandb.run.id}", type="model", description="DRL agent in PyTorch",
        metadata=metadata
    )

    model_artifact.add_file(f'{wandb.run.dir}/agent.pt')
    wandb.log_artifact(model_artifact, aliases=aliases)
    os.remove(f'{wandb.run.dir}/agent.pt')


def load_agent_from_wandb(agent, device, agent_id=None, full_model=True):
    """Restore trained agent checkpoint from Weights & Biases."""
    if agent_id is None:
        artifact = wandb.use_artifact(f'agent-{wandb.run.id}:latest', type='model')
    else:
        artifact = wandb.use_artifact(f'agent-{agent_id}:latest', type='model')

    artifact_dir = artifact.download(root=wandb.run.dir)
    print(f"Loaded agent {artifact.source_name} (global step {artifact.metadata['global_step']}).")
    checkpoint = torch.load(f'{artifact_dir}/agent.pt', map_location=device, weights_only=False)

    os.remove(f'{wandb.run.dir}/agent.pt')
    if full_model:
        agent = checkpoint['model']
    else:
        agent.load_state_dict(checkpoint['model_state_dict'])

    return agent, checkpoint['obs_rms'], checkpoint['return_rms'], checkpoint['config'], checkpoint[
        'optimizer_state_dict']


def measure_latency_agent(agent, envs, device, repeat=5000, warmup=5000, batch_size_max=256, batch_size_min=None,
                          dataset=None):
    """Measure inference latency across batch sizes on CUDA target device."""
    with torch.inference_mode():
        run_fn = lambda x, y: agent.forward(y)

        torch.cuda.empty_cache()
        if dataset is not None:
            assert batch_size_max <= len(dataset), "batch_size must be less than or equal to the dataset size"
            if not isinstance(dataset, torch.Tensor):
                dataset = torch.tensor(dataset)
            example_input = dataset[:batch_size_max].to(device)
        else:
            example_input = 128 * np.ones((batch_size_max,) + envs.single_observation_space.shape).astype(
                envs.single_observation_space.dtype)
            example_input = torch.tensor(example_input).to(device)
        latency_max, _ = measure_latency(agent, example_input, run_fn=run_fn, repeat=repeat, warmup=warmup)

        latency_min = None
        if batch_size_min is not None:
            time.sleep(2)
            torch.cuda.empty_cache()
            if dataset is not None:
                if not isinstance(dataset, torch.Tensor):
                    dataset = torch.tensor(dataset)
                example_input = dataset[:batch_size_min].to(device)
            else:
                example_input = 128 * np.ones((batch_size_min,) + envs.single_observation_space.shape).astype(
                    envs.single_observation_space.dtype)
                example_input = torch.tensor(example_input).to(device)
            latency_min, _ = measure_latency(agent, example_input, run_fn=run_fn, repeat=repeat, warmup=warmup)
    return latency_max, latency_min


def network_summary(network, input_data, device):
    """Prints network architecture summary and counts total parameters and MAC operations."""
    statistics = summary(
        network,
        input_data=input_data, device=device,
        depth=10,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "params_percent", "mult_adds"),
        verbose=1
    )
    total_params = statistics.total_params
    m_macs = np.round(statistics.total_mult_adds / 1e6, 2)
    param_bytes = statistics.total_param_bytes
    return statistics, total_params, m_macs, param_bytes


class StopTimer:
    """Stopwatch timer tracking wall-clock execution time during training and evaluation."""

    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0.0
        self.running = False

    def start(self):
        """Starts the execution stopwatch timer."""
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            print("Timer is already running!")

    def stop(self):
        """Stops the stopwatch timer and accumulates elapsed execution time."""
        if self.running:
            end_time = time.time()
            self.elapsed_time += end_time - self.start_time
            self.running = False
        else:
            print("Timer is not running!")

    def reset(self):
        """Resets accumulated elapsed time and timer state to zero."""
        self.elapsed_time = 0.0
        self.start_time = None
        self.running = False

    def get_elapsed_time(self):
        """Returns total accumulated elapsed execution time in seconds."""
        if self.running:
            current_time = time.time()
            return self.elapsed_time + (current_time - self.start_time)
        return self.elapsed_time

    def __str__(self):
        return f"Elapsed time: {self.get_elapsed_time()} seconds"
