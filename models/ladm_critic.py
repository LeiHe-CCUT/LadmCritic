# models/ladm_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KinematicsDecoder(nn.Module):
    """ Section 3.2.1: Decodes scene embedding and action into physical quantities. """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # This network's output size depends on how many physical quantities you model.
        # Example: 1 ego accel, 1 ego jerk, 1 speed deviation, 
        # 5 pairs of (rel_dist, rel_vel) for 5 nearest vehicles = 1 + 1 + 1 + 5*2 = 13
        output_dim = 13
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state_embedding, action):
        x = torch.cat([state_embedding, action], dim=1)
        physical_params = self.net(x)
        # It's good practice to return a dictionary for clarity
        return {
            'a_ego': physical_params[:, 0],
            'j_ego': physical_params[:, 1],
            'delta_v_des': physical_params[:, 2],
            'rel_dist': physical_params[:, 3:8],
            'rel_vel': physical_params[:, 8:13],
        }

class RiskEnergyModule(nn.Module):
    """ A generic module for DSE, DCE, DEE as described in Section 3.2.2. """
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ContextualWeightingNetwork(nn.Module):
    """ Section 3.2.3: Outputs adaptive weights (w_s, w_c, w_e) based on the scene. """
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3), # Outputs 3 raw scores for safety, comfort, efficiency
            nn.Softmax(dim=1) # Ensures weights sum to 1
        )
    
    def forward(self, state_embedding):
        return self.net(state_embedding)

class LagrangianHead(nn.Module):
    """ Section 3.2.4: Aggregates weighted features into the final Q-value. """
    def __init__(self, concatenated_feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(concatenated_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs the final scalar Q-value
        )

    def forward(self, x):
        return self.net(x)

class LadmCritic(nn.Module):
    """ The complete LadmCritic architecture from Section 3.2. """
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        dse_dim = config['dse_feature_dim']
        dce_dim = config['dce_feature_dim']
        dee_dim = config['dee_feature_dim']

        # 1. Kinematics Decoder
        self.kinematics_decoder = KinematicsDecoder(state_dim, action_dim)

        # 2. Risk-Energy Modules
        # Input dim for DSE is 2 (rel_dist, rel_vel) per vehicle. Assuming 5 vehicles.
        self.dse_module = RiskEnergyModule(input_dim=5 * 2, feature_dim=dse_dim) 
        # Input dim for DCE is 2 (accel, jerk)
        self.dce_module = RiskEnergyModule(input_dim=2, feature_dim=dce_dim)
        # Input dim for DEE is 1 (speed deviation)
        self.dee_module = RiskEnergyModule(input_dim=1, feature_dim=dee_dim)

        # 3. Contextual Weighting Network
        self.contextual_weighting_net = ContextualWeightingNetwork(state_dim)

        # 4. Lagrangian Head
        concatenated_dim = dse_dim + dce_dim + dee_dim
        self.lagrangian_head = LagrangianHead(concatenated_dim)
        
        print("LadmCritic model initialized successfully.")

    def forward(self, state, action):
        # Assume state is the final scene embedding h_final
        
        # Step 1: Decode kinematics
        phys_params = self.kinematics_decoder(state, action)
        
        # Step 2: Get risk-energy features
        # Note: We need to reshape the inputs for the modules
        dse_input = torch.cat([phys_params['rel_dist'], phys_params['rel_vel']], dim=1)
        f_dse = self.dse_module(dse_input)
        
        dce_input = torch.cat([phys_params['a_ego'].unsqueeze(1), phys_params['j_ego'].unsqueeze(1)], dim=1)
        f_dce = self.dce_module(dce_input)
        
        dee_input = phys_params['delta_v_des'].unsqueeze(1)
        f_dee = self.dee_module(dee_input)

        # Step 3: Get contextual weights
        weights = self.contextual_weighting_net(state) # Shape: (batch, 3)
        w_s, w_c, w_e = weights[:, 0], weights[:, 1], weights[:, 2]

        # Step 4: Weight the features and compute Q-value
        f_dse_weighted = w_s.unsqueeze(1) * f_dse
        f_dce_weighted = w_c.unsqueeze(1) * f_dce
        f_dee_weighted = w_e.unsqueeze(1) * f_dee
        
        concatenated_features = torch.cat([f_dse_weighted, f_dce_weighted, f_dee_weighted], dim=1)
        
        q_value = self.lagrangian_head(concatenated_features)
        return q_value