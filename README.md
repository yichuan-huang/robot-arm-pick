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
# Basic training with default hyperparameters
python scripts/sb3/train.py --task Isaac-Franka-Picking-v0 --num_envs 64 --headless

# Large-scale training with more environments for faster convergence
python scripts/sb3/train.py \
    --task Isaac-Franka-Picking-v0 \
    --num_envs 128 \
    --max_iterations 4000 \
    --seed 42 \
    --headless

# Extended training with video recording (10M timesteps)
python scripts/sb3/train.py \
    --task Isaac-Franka-Picking-v0 \
    --num_envs 64 \
    --max_iterations 10000 \
    --video \
    --video_interval 250000 \
    --headless

# High-performance training setup (recommended for research)
python scripts/sb3/train.py \
    --task Isaac-Franka-Picking-v0 \
    --num_envs 256 \
    --max_iterations 5000 \
    --log_interval 25000 \
    --video \
    --video_interval 500000 \
    --headless
```

**Training Features for Franka Picking:**
- 🚀 **Auto-optimization**: Automatically applies improved hyperparameters for Franka tasks
- 🧠 **Large networks**: 512-512-256 architecture for complex manipulation learning
- 💾 **Frequent saves**: Model checkpoints every 64K steps for long training sessions
- 📹 **Progress videos**: Optional video recording to monitor learning progress
- ⚙️ **Flexible timesteps**: Configure training duration via --max_iterations parameter

**Training Options:**
- `--task`: Environment identifier
- `--num_envs`: Parallel environments (recommended: 64-256 for large-scale training)
- `--max_iterations`: Training iterations (controls total timesteps = iterations × steps × envs)
- `--checkpoint`: Resume from checkpoint
- `--video`: Record training videos
- `--video_interval`: Steps between video recordings (recommended: 250K-500K for long training)
- `--log_interval`: Logging frequency (default: 50K, recommended: 25K for detailed monitoring)
- `--headless`: Run without GUI (essential for large-scale training)

### 🎮 Testing & Evaluation

Test a trained model:

```bash
# Test trained model
python scripts/sb3/play.py \
    --task Isaac-Franka-Picking-v0 \
    --checkpoint logs/sb3/Isaac-Franka-Picking-v0/YYYY-MM-DD_HH-MM-SS/model.zip \
    --num_envs 16

# Test with video recording
python scripts/sb3/play.py \
    --task Isaac-Franka-Picking-v0 \
    --checkpoint logs/sb3/Isaac-Franka-Picking-v0/YYYY-MM-DD_HH-MM-SS/model.zip \
    --num_envs 4 \
    --video
```

**Testing Options:**
- `--checkpoint`: Path to trained model [**Required**]
- `--num_envs`: Number of test environments (recommended: 4-16)
- `--video`: Record evaluation videos
- `--video_length`: Length of recorded videos (default: 200 steps)

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

| Parameter                    | Default | Large-Scale | Description                      |
| ---------------------------- | ------- | ----------- | -------------------------------- |
| `max_trajectory_deviation`   | 8cm     | 8cm         | Maximum allowed path deviation   |
| `target_tolerance`           | 5cm     | 5cm         | Success distance threshold       |
| `severe_violation_threshold` | 15cm    | 15cm        | Termination distance threshold   |
| `episode_length_s`           | 12.0s   | 12.0s       | Maximum episode duration         |
| `num_envs`                   | 64      | 128-256     | Parallel simulation environments |
| `total_timesteps`            | Config  | 5M-10M      | Total training timesteps         |

### Hyperparameter Tuning

For Franka picking tasks, the training script automatically applies optimized hyperparameters:

| Parameter       | Default | Franka-Optimized | Description                      |
| --------------- | ------- | ---------------- | -------------------------------- |
| `learning_rate` | 3e-4    | 5e-4             | Higher LR for faster convergence |
| `n_steps`       | 2048    | 4096             | Larger rollout buffer            |
| `batch_size`    | 64      | 128              | Larger batches for stability     |
| `n_epochs`      | 10      | 15               | More training epochs             |
| `net_arch`      | [64,64] | [512,512,256]    | Larger network capacity          |
| `gamma`         | 0.99    | 0.995            | Higher discount factor           |
| `gae_lambda`    | 0.95    | 0.98             | Better value estimation          |

## 📊 Results & Logging

Training progress is automatically logged including:
- 📈 **Episode Rewards**: Cumulative reward per episode
- 🎯 **Success Rate**: Percentage of successful grasps
- ⚠️ **Constraint Violations**: Safety violation frequency  
- 📐 **Trajectory Tracking**: Path following accuracy
- 🕒 **Training Metrics**: Timesteps/second, memory usage, GPU utilization
- 💾 **Model Checkpoints**: Saved every 64K steps for large-scale training

**Log Directory:** `logs/sb3/Isaac-Franka-Picking-v0/[TIMESTAMP]/`

### Training Performance Monitoring

Monitor training progress with TensorBoard:
```bash
# Launch TensorBoard to view training metrics
tensorboard --logdir logs/sb3/Isaac-Franka-Picking-v0/

# View specific training run
tensorboard --logdir logs/sb3/Isaac-Franka-Picking-v0/2025-MM-DD_HH-MM-SS/
```

### Expected Training Timeline
| Timesteps  | Training Time* | Expected Performance               |
| ---------- | -------------- | ---------------------------------- |
| 1M steps   | 30-60 min      | Basic learning, ~10% success       |
| 2.5M steps | 1-2 hours      | Moderate performance, ~40% success |
| 5M steps   | 2-4 hours      | Good performance, ~70% success     |
| 10M steps  | 4-8 hours      | Research-grade, 85%+ success       |

*Training times based on RTX 4080+ with 64-128 environments

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
| Component   | Minimum                    | Recommended                | Large-Scale Training     |
| ----------- | -------------------------- | -------------------------- | ------------------------ |
| **OS**      | Ubuntu 20.04+ / Windows 10 | Ubuntu 22.04+ / Windows 11 | Ubuntu 22.04+            |
| **GPU**     | NVIDIA RTX 3080            | NVIDIA RTX 4080+           | NVIDIA RTX 4090 / A6000+ |
| **VRAM**    | 8GB                        | 16GB+                      | 24GB+                    |
| **RAM**     | 16GB                       | 32GB+                      | 64GB+                    |
| **CPU**     | 8 cores                    | 16+ cores                  | 24+ cores                |
| **Storage** | 10GB free                  | 50GB+ free                 | 100GB+ free              |

### Performance Guidelines
- **64 environments**: RTX 3080+ with 16GB RAM
- **128 environments**: RTX 4080+ with 32GB RAM  
- **256+ environments**: RTX 4090/A6000+ with 64GB RAM
- **5M timesteps**: ~2-4 hours training time (depending on hardware)
- **10M timesteps**: ~4-8 hours training time (for research-grade results)

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
