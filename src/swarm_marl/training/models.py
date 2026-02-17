import numpy as np
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query, keys, values):
        # query: (B, 1, D)
        # keys: (B, K, D)
        # values: (B, K, D)
        context, _ = self.attn(query, keys, values)
        return context.squeeze(1) # (B, D)


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Multi-agent model that implements a centralized (V) critic.
    Optionally uses an Attention mechanism for the Actor to process neighbors.

    - Actor (Action): Uses local observations (potentially with Attention).
    - Critic (Value): Uses global state (concatenated positions/velocities/etc).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.use_attention = custom_config.get("use_attention", False)

        # ~~~ Actor (Policy) ~~~
        if self.use_attention:
            self.neighbor_k = custom_config.get("neighbor_k", 3)
            self.sensed_obstacles = custom_config.get("sensed_obstacles", 4)
            self.embed_dim = 32

            # Embeddings
            self.neighbor_embed = nn.Sequential(
                nn.Linear(4, self.embed_dim),
                nn.ReLU()
            )
            # Own state is 9 dim (pos(3)+vel(3)+goal(3))
            self.own_embed = nn.Sequential(
                nn.Linear(9, self.embed_dim),
                nn.ReLU()
            )
            self.attn_block = AttentionBlock(self.embed_dim)

            # Calculate new input dim for the FCNet
            # Original obs: [Own(9) | Neighbors(K*4) | Obstacles(M*4)]
            # Processed:    [Own(9) | AttnContext(32)| Obstacles(M*4)]
            new_obs_dim = 9 + self.embed_dim + (self.sensed_obstacles * 4)
            
            # Create a dummy space for the FCNet
            flat_obs_space = spaces.Box(-np.inf, np.inf, shape=(new_obs_dim,), dtype=np.float32)
            
            self.action_model = TorchFC(
                flat_obs_space,
                action_space,
                num_outputs,
                model_config,
                name + "_actor_attn",
            )
        else:
            self.action_model = TorchFC(
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name + "_actor",
            )

        # ~~~ Critic (Value) ~~~
        self.global_state_dim = custom_config.get("global_state_dim")
        if not self.global_state_dim:
            raise ValueError(
                "CentralizedCriticModel requires 'global_state_dim' in custom_model_config"
            )

        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.global_state_dim,), dtype=np.float32
        )
        
        self.value_model = TorchFC(
            self.state_space,
            action_space,
            1,  # Value function output is scalar
            model_config,
            name + "_critic",
        )

        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        # 1. Compute Action
        if self.use_attention:
            # Slice and Process
            obs = input_dict["obs"]
            # Shape (B, TotalDim)
            
            # Indices
            idx_own = 9
            idx_neighbors = idx_own + (self.neighbor_k * 4)
            
            own_state = obs[:, :idx_own] # (B, 9)
            neighbors_flat = obs[:, idx_own:idx_neighbors]
            obstacles_flat = obs[:, idx_neighbors:]
            
            # Reshape Neighbors
            neighbors = neighbors_flat.reshape(-1, self.neighbor_k, 4) # (B, K, 4)
            
            # Embed
            neighbors_emb = self.neighbor_embed(neighbors) # (B, K, H)
            own_emb = self.own_embed(own_state).unsqueeze(1) # (B, 1, H)
            
            # Attention (Query=Own, Key=Neighbor, Value=Neighbor)
            context = self.attn_block(own_emb, neighbors_emb, neighbors_emb) # (B, H)
            
            # Concat
            new_obs = torch.cat([own_state, context, obstacles_flat], dim=1)
            
            # Pass to FCNet
            # Creating a view dict for TorchFC
            # We must be careful not to mutate the original input_dict in a way that affects other calls?
            # Creating a new dict is safer.
            fc_input = {"obs": new_obs}
            logits, _ = self.action_model(fc_input, state, seq_lens)
            
        else:
            # Standard FCNet
            logits, _ = self.action_model(input_dict, state, seq_lens)

        # 2. Compute Value using Global State
        if "global_state" in input_dict:
            global_state = input_dict["global_state"]
            critic_input = {"obs": global_state}
            value_out, _ = self.value_model(critic_input, state, seq_lens)
            self._cur_value = value_out.squeeze(-1)
        else:
            self._cur_value = torch.zeros_like(logits[:, 0])

        return logits, state

    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
