from typing import Dict, Any
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class GlobalStateCallback(DefaultCallbacks):
    """
    Callback to add 'global_state' from info to the sample batch.
    Required for CentralizedCriticModel.
    """

    def on_postprocess_trajectory(
        self,
        *,
        worker: Any,
        episode: Any,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Dict] = None,
        **kwargs,
    ):
        # The 'global_state' is in the info dict of the last step.
        # 'postprocessed_batch' contains a batch of data. 
        # We need to extract 'global_state' from the 'infos' column if available.
        # Note: RLlib's 'infos' column is a list of dicts.
        
        # However, SampleBatch might not have 'infos' fully populated as tensors yet?
        # Typically 'infos' is available. 
        
        # Alternative: We can use `episode.last_info_for(agent_id)` but that's only the last one.
        # We need the global state for EVERY step in the batch.
        
        # Efficient way: The environment puts 'global_state' in info.
        # RLlib collects infos. We can iterate and extract.
        
        # 1. Initialize with zeros to ensure column always exists (prevents concatenation errors)
        # Access the model from the policy to get dimensions
        model = policies[policy_id].model
        if hasattr(model, "global_state_dim") and model.global_state_dim:
            import numpy as np
            batch_len = postprocessed_batch.count
            postprocessed_batch["global_state"] = np.zeros(
                (batch_len, model.global_state_dim), dtype=np.float32
            )

        # 2. Populate with actual data from infos if available
        if SampleBatch.INFOS in postprocessed_batch:
            infos = postprocessed_batch[SampleBatch.INFOS]
            
            if len(infos) > 0 and "global_state" in infos[0]:
                # Extract global_state from each info dict
                global_states = [info["global_state"] for info in infos]
                postprocessed_batch["global_state"] = np.array(global_states, dtype=np.float32)
