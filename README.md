# Robot Arm Pick

A robotic manipulation environment built on Isaac Lab for learning constrained picking tasks with the Franka Panda robot arm.

## 🎯 Overview

This project implements a constrained robotic picking task where a Franka Panda robot learns to pick objects while adhering to predefined trajectory constraints. The environment is designed for reinforcement learning research in robotic manipulation with safety constraints.

### Key Features
- **🤖 Franka Panda Robot**: 7-DOF robot arm with parallel gripper
- **📐 Constrained Trajectories**: Robot must follow predefined safe trajectories
- **🎯 Object Picking**: Pick and place tasks with various objects
- **🏆 Reward Engineering**: Multi-objective rewards for trajectory compliance and task success
- **📚 Curriculum Learning**: Adaptive difficulty with progressive tolerance tightening
- **🔍 Debug & Monitoring**: Real-time reward analysis and success rate tracking
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
| Component                | Weight | Range     | Description                           |
| ------------------------ | ------ | --------- | ------------------------------------- |
| **Trajectory Tracking**  | 2.0    | [-0.5, 1] | Exponential reward for path following |
| **Target Approach**      | 3.0    | [0, 1]    | Distance-based approach reward        |
| **Success Bonus**        | 1.0    | [20, 30]  | Large bonus for successful grasp      |
| **Grasp Precision**      | 2.0    | [0, 1]    | Precision positioning reward          |
| **Time Efficiency**      | 1.0    | [0, 0.2]  | Early completion incentive            |
| **Trajectory Violation** | 0.3    | [-1, 0]   | Penalty for constraint violations     |
| **Joint Velocity**       | 0.1    | [-1, 0]   | Smoothness penalty                    |
| **Action Smoothness**    | 0.1    | [-1, 0]   | Control smoothness penalty            |

### 📚 Curriculum Learning
The environment implements adaptive difficulty progression:
- **Initial tolerance**: 0.07m (easier for early training)
- **Final tolerance**: 0.025m (final precision requirement)  
- **Decay period**: 500,000 steps (configurable)
- **Adaptive scaling**: Linear progression from easy to hard

### 🏁 Termination Conditions
- ✅ **Success**: End-effector within curriculum tolerance of target + gripper closed
- ❌ **Severe Violation**: Trajectory deviation > 15cm
- ⚠️ **Joint Limits**: Robot joint positions exceed safety limits
- ⏰ **Timeout**: Episode exceeds maximum duration (12 seconds)

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
# Basic training with reward debugging enabled
python scripts/sb3/train.py --task Isaac-Franka-Picking-v0 --num_envs 64 --rew_debug_interval 200 --headless

# Training with custom curriculum learning parameters
python scripts/sb3/train.py \
    --task Isaac-Franka-Picking-v0 \
    --num_envs 128 \
    --curriculum_initial_tolerance 0.08 \
    --curriculum_final_tolerance 0.02 \
    --curriculum_decay_steps 750000 \
    --rew_debug_interval 500 \
    --headless

# Enhanced success bonus for faster learning
python scripts/sb3/train.py \
    --task Isaac-Franka-Picking-v0 \
    --num_envs 64 \
    --success_base_bonus 25.0 \
    --success_time_bonus 15.0 \
    --rew_debug_interval 200 \
    --max_iterations 4000 \
    --headless

# Large-scale training with monitoring
python scripts/sb3/train.py \
    --task Isaac-Franka-Picking-v0 \
    --num_envs 256 \
    --max_iterations 5000 \
    --rew_debug_interval 1000 \
    --curriculum_decay_steps 1000000 \
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

**🔍 Debugging & Monitoring Options:**
- `--rew_debug_interval N`: Print reward term means every N steps (recommended: 200-1000)
- `--success_base_bonus X`: Override base success bonus (default: 20.0)
- `--success_time_bonus Y`: Override time-based success bonus (default: 10.0)

**📚 Curriculum Learning Options:**
- `--curriculum_initial_tolerance Z`: Initial target tolerance in meters (default: 0.07)
- `--curriculum_final_tolerance W`: Final target tolerance in meters (default: 0.025)
- `--curriculum_decay_steps S`: Steps to transition from initial to final (default: 500000)

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

| Parameter                      | Default | Description                           |
| ------------------------------ | ------- | ------------------------------------- |
| `max_trajectory_deviation`     | 0.10m   | Maximum allowed path deviation        |
| `target_tolerance_initial`     | 0.07m   | Initial success distance (curriculum) |
| `target_tolerance_final`       | 0.025m  | Final success distance (curriculum)   |
| `target_tolerance_decay_steps` | 500K    | Steps for curriculum progression      |
| `severe_violation_threshold`   | 0.15m   | Termination distance threshold        |
| `success_base_bonus`           | 20.0    | Base reward for successful grasp      |
| `success_time_bonus`           | 10.0    | Extra reward for fast completion      |
| `reward_debug_print_interval`  | 200     | Steps between reward debug prints     |
| `episode_length_s`             | 12.0s   | Maximum episode duration              |

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

### 🔧 Reward Engineering Guidelines

When tuning rewards, follow these principles:
- **Bounded rewards**: All terms are normalized to consistent ranges for stable training
- **Balanced weights**: Current weights are tuned for the bounded reward ranges
- **Success emphasis**: Large success bonus (20-30) creates clear learning signal
- **Curriculum integration**: Precision rewards automatically adapt to current tolerance

## 📊 Results & Logging

Training progress is automatically logged including:
- 📈 **Episode Rewards**: Cumulative reward per episode
- 🎯 **Success Rate**: Percentage of successful grasps  
- ⚠️ **Constraint Violations**: Safety violation frequency
- 📐 **Trajectory Tracking**: Path following accuracy
- 🔍 **Per-term Rewards**: Individual reward component analysis
- 📚 **Curriculum Progress**: Target tolerance evolution over time
- 🕒 **Training Metrics**: Timesteps/second, memory usage, GPU utilization
- 💾 **Model Checkpoints**: Saved every 64K steps for large-scale training

**Log Directory:** `logs/sb3/Isaac-Franka-Picking-v0/[TIMESTAMP]/`

### 🔍 Debugging Output

When `--rew_debug_interval` is enabled, you'll see detailed reward analysis:

```bash
[reward-debug][step=200] track_trajectory: mean=0.7234
[reward-debug][step=200] approach_target: mean=0.4567
[reward-debug][step=200] success_bonus: mean=0.0234
[reward-debug][step=200] grasp_precision: mean=0.3456
[Curriculum] Step 10000: target_tolerance = 0.0686
[Success Rate] Episodes: 1000, Success Rate: 0.0234, Target Tolerance: 0.0680
[Reward Debug] Step 15000: 3/64 envs succeeded this step
```

This helps identify:
- **Reward balance**: Which terms dominate training
- **Success progression**: How success rate improves over time  
- **Curriculum effectiveness**: Whether difficulty progression is appropriate

### Training Performance Monitoring

Monitor training progress with TensorBoard:
```bash
# Launch TensorBoard to view training metrics
tensorboard --logdir logs/sb3/Isaac-Franka-Picking-v0/

# View specific training run
tensorboard --logdir logs/sb3/Isaac-Franka-Picking-v0/2025-MM-DD_HH-MM-SS/
```

### Expected Training Timeline
| Timesteps  | Training Time* | Expected Performance              | Curriculum Status         |
| ---------- | -------------- | --------------------------------- | ------------------------- |
| 100K steps | 5-15 min       | Initial learning, ~5% success     | Easy tolerance (0.07m)    |
| 500K steps | 25-45 min      | Curriculum midpoint, ~25% success | Mid tolerance (0.05m)     |
| 1M steps   | 45-90 min      | Good learning, ~45% success       | Tighter tolerance (0.04m) |
| 2.5M steps | 2-4 hours      | Strong performance, ~70% success  | Final tolerance (0.025m)  |
| 5M steps   | 4-8 hours      | Research-grade, 85%+ success      | Expert precision          |

*Training times based on RTX 4080+ with 64-128 environments

The curriculum learning system automatically adjusts difficulty, typically showing:
- **Phase 1 (0-200K)**: High success rate with easy tolerance, agent learns basics
- **Phase 2 (200K-600K)**: Success rate may dip as tolerance tightens, agent adapts  
- **Phase 3 (600K+)**: Success rate recovers and improves with final precision

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
