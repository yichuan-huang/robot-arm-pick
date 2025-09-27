# Franka Path Tracking

A simplified Franka robot arm path tracking environment for IsaacLab, focused on optimal trajectory control from start point to target without grasping tasks.

## Overview

This project implements a streamlined robotic arm control system for path tracking. The system removes grasping complexities while maintaining optimal trajectory planning with constraint satisfaction.

### Key Features
- **Path Control**: Optimal trajectory planning from P0 to Pt
- **Constraint Satisfaction**: Max 5cm path deviation
- **Time Optimization**: Shortest execution time
- **Simplified Design**: No grasping functionality
- **Bezier Trajectories**: Smooth curve generation
- **PPO Training**: Stable-Baselines3 integration

## Mathematical Modeling

### Problem Formulation

The robot arm control problem is formulated as a Markov Decision Process (MDP) where the robot must reach a target position using its gripper center (midpoint between fingertips) while maintaining smooth trajectories.

**State Space**: $\mathcal{S} \subset \mathbb{R}^{28}$
- Joint positions: $q \in \mathbb{R}^7$
- Joint velocities: $\dot{q} \in \mathbb{R}^7$
- Gripper center position: $p_g \in \mathbb{R}^3$
- Target position: $p_t \in \mathbb{R}^3$
- Distance to target: $d_t \in \mathbb{R}$
- Task progress: $\tau \in [0,1]$

**Action Space**: $\mathcal{A} = [-1,1]^7$ (normalized joint position commands)

**Gripper Center Calculation**:
$$p_g = \frac{p_{f1} + p_{f2}}{2}$$
where $p_{f1}$, $p_{f2}$ are the positions of the two fingertip centers.

**Distance Metric**:
$$d_t = \|p_g - p_t\|_2$$

### Reward Function Design

The reward function consists of multiple components designed to encourage efficient target reaching:

#### 1. Distance-Based Reward
$$R_{\text{distance}}(s_t) = -\|p_g - p_t\|_2$$

#### 2. Multi-Tier Approach Reward
$$R_{\text{approach}}(s_t) = \begin{cases}
100.0 & \text{if } d_t \leq 0.01 \text{ (1cm)} \\
50.0 & \text{if } 0.01 < d_t \leq 0.02 \text{ (2cm)} \\
20.0 & \text{if } 0.02 < d_t \leq 0.05 \text{ (5cm)} \\
5.0 & \text{if } 0.05 < d_t \leq 0.10 \text{ (10cm)} \\
0.0 & \text{otherwise}
\end{cases}$$

#### 3. Success Bonus
$$R_{\text{success}}(s_t) = \begin{cases}
1000.0 & \text{if } d_t \leq \theta_{\text{success}} \\
0.0 & \text{otherwise}
\end{cases}$$
where $\theta_{\text{success}} = 0.015$ m (1.5cm threshold).

#### 4. Smoothness Penalties
- Action smoothness: $R_{\text{smooth}} = -0.1 \cdot \|a_t - a_{t-1}\|_2^2$
- Joint limits: $R_{\text{limits}} = -0.1 \cdot \sum_{i=1}^{7} \mathbb{I}[q_i \notin [q_{\min,i}, q_{\max,i}]]$

#### 5. Total Reward
$$R(s_t, a_t) = 10.0 \cdot R_{\text{distance}} + 100.0 \cdot R_{\text{success}} + R_{\text{approach}} + R_{\text{smooth}} + R_{\text{limits}}$$

### Termination Conditions

1. **Success**: $d_t \leq 0.02$ m (2cm precision)
2. **Out of Bounds**: $p_g \notin \mathcal{W}$ (workspace bounds)
3. **Timeout**: $t \geq t_{\max} = 12.0$ s
4. **Safety**: Robot height $< -0.5$ m

## Project Structure

```
robot_arm_pick/
├── scripts/
│   ├── path_control_demo.py    # Standalone demo
│   ├── random_agent.py         # Random policy
│   ├── zero_agent.py           # Zero action baseline
│   └── sb3/
│       ├── train.py            # Training script
│       └── play.py             # Testing script
└── source/robot_arm_pick/
    └── robot_arm_pick/tasks/manager_based/robot_arm_pick/
        ├── franka_path_env.py      # Path tracking environment
        ├── franka_path_env_cfg.py  # Environment configuration
        ├── agents/
        │   └── sb3_ppo_cfg.yaml    # PPO config
        └── mdp/
            ├── observations.py     # Observation functions
            ├── rewards.py          # Reward functions
            └── terminations.py     # Termination conditions
```

## Environment Details

### Observation Space (28-dimensional)
- Joint positions (7D): $q \in \mathbb{R}^7$
- Joint velocities (7D): $\dot{q} \in \mathbb{R}^7$
- Gripper center position (3D): $p_g \in \mathbb{R}^3$
- Target sphere position (3D): $p_t \in \mathbb{R}^3$
- Distance to target (1D): $d_t = \|p_g - p_t\|_2$
- Task progress (1D): $\tau \in [0,1]$

### Action Space (7-dimensional)
- Joint position commands (continuous, normalized [-1,1]): $a \in [-1,1]^7$

### Reward Structure
- **Distance Reward**: $R_{\text{distance}} = -d_t$ (weight: 10.0)
- **Success Bonus**: $R_{\text{success}} = 1000.0$ when $d_t \leq 0.015$m (weight: 100.0)
- **Multi-tier Approach**: Tiered rewards for different distance ranges
- **Action Smoothness**: $R_{\text{smooth}} = -\|a_t - a_{t-1}\|_2^2$ (weight: -0.1)
- **Joint Limits**: Penalty for joint limit violations (weight: -0.1)

### Termination Conditions
- **Success**: $d_t \leq 0.02$m (2cm precision)
- **Out of Bounds**: Gripper center outside workspace
- **Timeout**: $t \geq 12.0$s
- **Safety**: Robot falling below -0.5m

## Quick Start

### Prerequisites
- Isaac Lab
- Python 3.10+

### Installation

```bash
git clone https://github.com/yichuan-huang/robot-arm-pick.git
cd robot-arm-pick
python -m pip install -e source/robot_arm_pick
```

### Training

```bash
# Basic training
python scripts/sb3/train.py --task Isaac-Franka-Path-v0 --num_envs 64 --headless

# With precision control parameters
python scripts/sb3/train.py \
    --task Isaac-Franka-Path-v0 \
    --num_envs 1024 \
    --target_tolerance 0.005 \
    --max_episode_time 12.0 \
    --curriculum_enabled True \
    --headless
```

### Testing

```bash
# Test trained model
python scripts/sb3/play.py \
    --task Isaac-Franka-Path-v0 \
    --checkpoint logs/sb3/model.zip \
    --num_envs 4
```

## Configuration

### Key Parameters
- `target_tolerance`: 0.005m (0.5cm final precision, curriculum: 0.02m → 0.005m)
- `success_threshold`: 0.02m (2cm termination condition)
- `reward_threshold`: 0.015m (1.5cm success bonus)
- `max_episode_time`: 12.0s (extended time limit)
- `curriculum_steps`: 1,000,000 (steps for tolerance transition)

### Training Options
- `--task`: Environment ID
- `--num_envs`: Parallel environments
- `--max_iterations`: Training steps
- `--checkpoint`: Resume training
- `--headless`: No GUI

## Development

### Core Files
- `franka_path_env.py`: Main environment
- `mdp/`: MDP components
- `scripts/sb3/train.py`: Training script

### Dependencies
```bash
isaac-lab
stable-baselines3
torch
gymnasium
```

## License

MIT License
