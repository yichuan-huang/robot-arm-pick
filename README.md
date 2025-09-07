# Robot Arm Pick

A robotic manipulation environment built on Isaac Lab for learning constrained picking tasks with the Franka Panda robot arm.

## 🎯 Overview

This project implements a constrained robotic picking task where a Franka Panda robot learns to pick objects while adhering to predefined trajectory constraints. The environment is designed for reinforcement learning research in robotic manipulation with safety constraints.

### Key Features
- **🤖 Franka Panda Robot**: 7-DOF robot arm with parallel gripper
- **📐 Constrained Trajectories**: Robot must follow predefined safe trajectories
- **🎯 Object Picking**: Pick and place tasks with various objects
- **🏆 Reward Engineering**: Multi-objective rewards for trajectory compliance and task success
- **⚡ High Performance**: Vectorized simulation with Isaac Lab

## 📁 Project Structure

```
robot_arm_pick/
├── 📄 README.md                       # This file
├── 📁 scripts/                        # Executable scripts
│   ├── 🎲 random_agent.py            # Random policy demo
│   ├── 🚫 zero_agent.py              # Zero action baseline
│   ├── 📋 list_envs.py               # Environment discovery
│   └── 📁 sb3/                       # Stable-Baselines3 integration
│       ├── 🏋️ train.py                # Training script
│       └── 🎮 play.py                 # Testing/inference script
├── 📁 source/robot_arm_pick/          # Main package source
│   ├── ⚙️ pyproject.toml             # Package configuration
│   ├── 🔧 setup.py                   # Package setup
│   └── 📁 robot_arm_pick/            # Core package
│       └── 📁 tasks/manager_based/robot_arm_pick/
│           ├── 🌍 franka_picking_env.py           # Environment implementation
│           ├── ⚙️ franka_picking_env_cfg.py       # Environment config
│           ├── 📁 agents/                         # RL agent configs
│           │   └── 🎯 sb3_ppo_cfg.yaml           # PPO hyperparameters
│           └── 📁 mdp/                            # MDP components
│               ├── 👁️ observations.py             # Observation functions
│               ├── 🏆 rewards.py                  # Reward functions
│               └── 🏁 terminations.py             # Termination conditions
└── 📁 .vscode/                       # VS Code configuration
    └── 🛠️ tools/                      # Development tools
```

## 🔬 Environment Details

### Scene Components
- **📋 Table**: Wooden table surface (60cm × 50cm × 5cm)
- **🤖 Robot**: Franka Panda arm with parallel gripper
- **📦 Objects**: Randomly placed target objects
- **💡 Lighting**: Dome lighting for optimal visualization

### Observation Space (17-dimensional)
| Component         | Dimension | Description                |
| ----------------- | --------- | -------------------------- |
| Joint Positions   | 7         | Arm joint angles           |
| Joint Velocities  | 7         | Arm joint velocities       |
| End-effector Pose | 3         | EE position in world frame |

### Action Space (9-dimensional)
- **7** arm joint torques (continuous)
- **2** gripper joint efforts (continuous)
- Actions normalized to [-1, 1] range

### 🎯 Reward Structure
| Component               | Weight      | Description                         |
| ----------------------- | ----------- | ----------------------------------- |
| **Survival**            | -0.01/step  | Encourages task completion          |
| **Trajectory Tracking** | Exponential | Rewards staying near reference path |
| **Target Approach**     | Exponential | Rewards approaching target object   |
| **Success Bonus**       | +100.0      | Large reward for successful grasp   |

### 🏁 Termination Conditions
- ✅ **Success**: End-effector within 5cm of target
- ❌ **Constraint Violation**: Trajectory deviation > 15cm
- ⏰ **Timeout**: Episode exceeds maximum duration

## 🚀 Quick Start

### Prerequisites
- **Isaac Lab** - Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- **Python 3.8+** with conda/mamba environment management

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yichuan-huang/robot-arm-pick.git
   cd robot-arm-pick
   ```

2. **Install the package**
   ```bash
   python -m pip install -e source/robot_arm_pick
   ```

3. **Verify installation**
   ```bash
   python scripts/list_envs.py
   ```

### 🏋️ Training

Train a PPO agent using Stable-Baselines3:

```bash
# Basic training
python scripts/sb3/train.py --task Isaac-Franka-Picking-v0 --headless

# Advanced training with custom parameters
python scripts/sb3/train.py \
    --task Isaac-Franka-Picking-v0 \
    --num_envs 64 \
    --max_iterations 1000 \
    --seed 42 \
    --headless
```

**Training Options:**
- `--task`: Environment identifier
- `--num_envs`: Parallel environments (default: 64)
- `--max_iterations`: Training iterations (default: 1000)
- `--checkpoint`: Resume from checkpoint
- `--video`: Record training videos
- `--headless`: Run without GUI

### 🎮 Testing & Evaluation

Test a trained model:

```bash
# Test trained model
python scripts/sb3/play.py \
    --task Isaac-Franka-Picking-v0 \
    --checkpoint /path/to/model.zip \
    --num_envs 4
```

**Testing Options:**
- `--checkpoint`: Path to trained model [**Required**]
- `--num_envs`: Number of test environments
- `--video`: Record evaluation videos

### 🎲 Demo Scripts

Run baseline demonstrations:

```bash
# Random policy baseline
python scripts/random_agent.py --task Isaac-Franka-Picking-v0 --num_envs 1

# Zero action baseline  
python scripts/zero_agent.py --task Isaac-Franka-Picking-v0 --num_envs 1
```

## ⚙️ Configuration

### Key Environment Parameters

The environment behavior can be customized through configuration files:

| Parameter                    | Default | Description                      |
| ---------------------------- | ------- | -------------------------------- |
| `max_trajectory_deviation`   | 8cm     | Maximum allowed path deviation   |
| `target_tolerance`           | 5cm     | Success distance threshold       |
| `severe_violation_threshold` | 15cm    | Termination distance threshold   |
| `episode_length_s`           | 8.0s    | Maximum episode duration         |
| `num_envs`                   | 64      | Parallel simulation environments |

### Hyperparameter Tuning

Agent hyperparameters are defined in:
```
source/robot_arm_pick/robot_arm_pick/tasks/manager_based/robot_arm_pick/agents/sb3_ppo_cfg.yaml
```

## 📊 Results & Logging

Training progress is automatically logged including:
- 📈 **Episode Rewards**: Cumulative reward per episode
- 🎯 **Success Rate**: Percentage of successful grasps
- ⚠️ **Constraint Violations**: Safety violation frequency
- 📐 **Trajectory Tracking**: Path following accuracy

**Log Directory:** `logs/franka_picking_sb3/[TIMESTAMP]/`

### Visualization

Enable real-time visualization during training:
```bash
python scripts/sb3/train.py --task Isaac-Franka-Picking-v0  # Remove --headless flag
```

## 🔧 Development

### Code Structure
- **Environment Logic**: `franka_picking_env.py`
- **MDP Components**: `mdp/` directory
- **Configuration**: `*_cfg.py` files
- **Training Scripts**: `scripts/sb3/`

### Adding New Features
1. **Custom Rewards**: Modify `mdp/rewards.py`
2. **New Observations**: Update `mdp/observations.py`
3. **Termination Logic**: Edit `mdp/terminations.py`

## 📋 Requirements

### System Requirements
- **OS**: Linux (Ubuntu 22.04+ recommended), Windows 10/11
- **GPU**: NVIDIA RTX series (RTX 4080+ recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### Software Dependencies
```bash
# Core dependencies
isaac-lab              # Physics simulation framework
stable-baselines3      # RL algorithms
torch                  # Deep learning backend
gymnasium             # RL environment interface

# Optional dependencies  
tensorboard           # Training visualization
wandb                 # Experiment tracking
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Isaac Lab Team** for the excellent simulation framework
- **Stable-Baselines3** for robust RL implementations
- **Franka Emika** for the Panda robot design
