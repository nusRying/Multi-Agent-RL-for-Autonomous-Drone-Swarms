"""
Fixed version of drone_physics_env.py step method termination logic.
This fixes the "Batches sent to postprocessing must only contain steps from a single trajectory" error.

Key change: Episode ends on ANY collision (not ALL terminated), and all agents are marked consistently.
"""

# REPLACE lines 198-227 in drone_physics_env.py with:

```python
        # Terminations - check each agent
        any_collision = False
        all_goals_reached = True

        for agent_id in active_agents:
            idx = self.agent_ids.index(agent_id)
            pos, _ = p.getBasePositionAndOrientation(self.drone_ids[idx])
            dist = np.linalg.norm(np.array(pos) - self.goal)

            reward = -dist * 0.1  # Simple dense reward
            rewards[agent_id] = reward

            # Check collision
            if collided[agent_id]:
                rewards[agent_id] -= 10.0
                any_collision = True
            # Check goal reached
            elif dist < self.cfg.goal_radius:
                rewards[agent_id] += 50.0
            else:
                all_goals_reached = False

        # Episode termination logic
        time_limit = self.step_count >= self.cfg.max_steps
        episode_done = any_collision or all_goals_reached or time_limit

        if episode_done:
            # Mark ALL agents with consistent done flags
            for agent_id in self.agent_ids:
                if time_limit and not any_collision and not all_goals_reached:
                    # Pure time limit (no collision, not all goals)
                    terminated[agent_id] = False
                    truncated[agent_id] = True
                else:
                    # Collision or all goals reached
                    terminated[agent_id] = True
                    truncated[agent_id] = False

            terminated["__all__"] = True if (any_collision or all_goals_reached) else False
            truncated["__all__"] = True if (time_limit and not any_collision and not all_goals_reached) else False
            self.agents = []
        else:
            # Episode continues
            for agent_id in self.agent_ids:
                terminated[agent_id] = False
                truncated[agent_id] = False
            terminated["__all__"] = False
            truncated["__all__"] = False
```
