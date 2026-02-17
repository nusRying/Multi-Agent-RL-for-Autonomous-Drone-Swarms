import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class SwarmVisualizer:
    def __init__(self, world_size: float = 100.0, num_drones: int = 3, num_obstacles: int = 0):
        self.world_size = world_size
        self.num_drones = num_drones
        self.num_obstacles = num_obstacles
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize scatters
        self.drone_scatters = [
            self.ax.scatter([], [], [], c='blue', marker='^', s=50, label='Drones' if i == 0 else "")
            for i in range(num_drones)
        ]
        self.obstacle_scatter = self.ax.scatter([], [], [], c='red', marker='o', s=100, label='Obstacles')
        self.goal_scatter = self.ax.scatter([], [], [], c='green', marker='*', s=150, label='Goal')
        self.trails = [self.ax.plot([], [], [], c='blue', alpha=0.3)[0] for _ in range(num_drones)]
        
        self.ax.set_xlim(-world_size/2, world_size/2)
        self.ax.set_ylim(-world_size/2, world_size/2)
        self.ax.set_zlim(-world_size/2, world_size/2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.set_title("Drone Swarm Visualization")

    def animate(self, history: list[dict], interval: int = 50, save_path: str = None):
        """
        Animate the swarm history.
        
        Args:
            history: List of dictionaries containing 'positions', 'obstacles', 'goals'.
                     positions: (N, 3)
                     obstacles: (M, 3)
                     goal: (3,)
            interval: Time between frames in ms.
            save_path: If provided, save animation to this path (e.g. 'video.mp4').
        """
        
        def update(frame):
            data = history[frame]
            positions = data['positions']
            obstacles = data['obstacles']
            goal = data['goal']
            
            # Update Drones
            for i, scatter in enumerate(self.drone_scatters):
                scatter._offsets3d = (positions[i:i+1, 0], positions[i:i+1, 1], positions[i:i+1, 2])
            
            # Update Trails
            hist_len = 50 # Keep last 50 frames for trails
            start_frame = max(0, frame - hist_len)
            for i, trail in enumerate(self.trails):
                # Extract history for drone i
                past_pos = np.array([history[f]['positions'][i] for f in range(start_frame, frame+1)])
                if len(past_pos) > 0:
                    trail.set_data(past_pos[:, 0], past_pos[:, 1])
                    trail.set_3d_properties(past_pos[:, 2])

            # Update Obstacles
            if len(obstacles) > 0:
                self.obstacle_scatter._offsets3d = (obstacles[:, 0], obstacles[:, 1], obstacles[:, 2])
            
            # Update Goal
            self.goal_scatter._offsets3d = (goal[0:1], goal[1:2], goal[2:3])
            
            self.ax.set_title(f"Drone Swarm - Step {frame}/{len(history)}")
            return self.drone_scatters + [self.obstacle_scatter, self.goal_scatter] + self.trails

        ani = animation.FuncAnimation(
            self.fig, update, frames=len(history), interval=interval, blit=False
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer='ffmpeg', fps=30)
            print("Done.")
        else:
            plt.show()
