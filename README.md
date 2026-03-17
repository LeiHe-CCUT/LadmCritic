LadmCritic: A Physics-Aware Graph-Based Reinforcement Learning Framework for Autonomous Driving Safety
This repository contains the official implementation of the paper: "LadmCritic: A Physics-Aware Graph-Based Reinforcement Learning Framework with Dynamic Lagrangian Action Minimization for Autonomous Driving Safety".

LadmCritic is a novel "online-training, offline-evaluation" framework that embeds Lagrangian mechanics into a Physics-Aware Graph Attention Network (PA-GAT) and utilizes a Contextual Weighting Network (CWN) to dynamically balance safety, comfort, and efficiency. It serves as both an efficient RL "player-coach" and a standalone, zero-shot safety evaluator.

🌿 Repository Branch Architecture
To keep the repository clean and efficient, the project is organized into two main branches:

main: Contains all the source code, environment configurations, and training/evaluation scripts.

checkpoints: Contains the pre-trained model weights and experiment logs. Switch to this branch or download the weights directly to perform zero-shot offline evaluation without retraining.

📂 Code Structure (main branch)
Based on the repository architecture, the core files and directories are organized as follows:

Plaintext

LadmCritic/
├── agents/                   # Implementations of RL agents (SAC, etc.)
├── configs/                  # Configuration files for hyperparameters and environments
├── dataset/                  # Scenarios and trajectory data for offline evaluation & BC
├── logs/                     # Training logs (TensorBoard)
├── models/                   # Neural network architectures (PA-GAT, CWN, MLPs)
├── trained_models/           # Directory to save locally trained model checkpoints
├── utils/                    # Utility functions (metrics, plotting, physical energy calculations)
│
├── main_train.py             # 🚀 Main script to train our proposed LadmCritic
├── main_train_mlp.py         # Script to train the SAC-MLP Baseline (pure data-driven)
├── train_ablation_ladm_reward.py # Script for the SAC-MLP + LadmReward ablation study
│
├── train_bc.py               # Script for Behavioral Cloning (BC) pre-training
├── main_train_bc_finetune.py # Script to fine-tune models from BC Warm-Start
│
├── main_evaluate.py          # 📊 Script for offline safety evaluation and metric scoring
├── collect_scenarios.py      # Script to collect diverse driving scenarios
├── collect_dummy_scenarios.py# Script to generate dummy/test scenarios
├── data_process.py           # Tools for data processing and state normalization
├── export_results.py         # Script to export evaluation results and generate figures
│
├── app.py                    # 🖥️ Interactive visualization engine / dashboard
└── use_highway_env.py        # Custom wrappers for highway-env interaction
⚙️ Installation
Clone the repository:

Bash

git clone https://github.com/LeiHe-CCUT/LadmCritic.git
cd LadmCritic
Create a virtual environment and install dependencies:

Bash

conda create -n ladmcritic python=3.8
conda activate ladmcritic
pip install -r requirements.txt
(Note: Ensure you have pytorch, stable-baselines3, and highway-env installed.)

🚀 Usage
1. Online Training Phase (Policy Learning)
As described in the paper, all agents can utilize a Behavioral Cloning (BC) Warm-Start strategy before RL fine-tuning to ensure safe initial exploration.

Step 1: BC Warm-Start (Optional but recommended)

Bash

python train_bc.py
Step 2: Train the specific models
Train the proposed LadmCritic (Ours):

Bash

python main_train.py
# Or with BC fine-tuning: python main_train_bc_finetune.py
Train the SAC-MLP Baseline:

Bash

python main_train_mlp.py
Train the Ablation Model (SAC-MLP + LadmReward):

Bash

python train_ablation_ladm_reward.py
2. Offline Evaluation Phase (Safety Metric)
To evaluate the models and generate the "Safety vs. Efficiency" metrics, you can use the pre-trained weights from the checkpoints branch.

Evaluate LadmCritic on specific scenarios:

Bash

python main_evaluate.py 
You can configure the specific scenario IDs (e.g., 0038follow38, 1095cutin44) inside the configuration files or via command-line arguments to reproduce the scores in Table 4 of our paper.

3. Data Processing & Visualization
To process collected data and export evaluation charts (like the Radar Chart or Learning Curves):

Bash

python data_process.py
python export_results.py
To launch the real-time continuous Safety and Comfort Fields visualization engine (Figure 7 in the paper):

Bash

python app.py
📦 Checkpoints & Pre-trained Models (checkpoints branch)
If you switch to the checkpoints branch, you will find a highly organized structure containing all the experimental runs and pre-trained weights mentioned in the paper:

Plaintext

checkpoints/
├── ladm_experiment_[1-12]/           # Pre-trained weights & logs for LadmCritic (Ours)
├── mlp_baseline_experiment_[2-7]/    # Pre-trained weights & logs for SAC-MLP Baseline
├── ablation_ladm_reward_[1-2]/       # Pre-trained weights & logs for Ablation models
└── bc_finetune_experiment_[1-5]/     # Pre-trained weights for BC fine-tuning strategies
You can download these specific experiment folders and place them in your trained_models/ or logs/ directory in the main branch to evaluate them directly using main_evaluate.py.
