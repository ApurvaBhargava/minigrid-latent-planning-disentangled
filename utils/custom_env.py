"""Custom DoorKey-5x5 environment with configurable positions."""

from collections import deque
import gymnasium as gym

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from .environment import get_full_obs


class CustomDoorKey5x5(MiniGridEnv):
    """Custom DoorKey-5x5 environment with configurable key/door/goal positions.
    
    Grid layout (5x5):
    - Outer walls at x=0,4 and y=0,4
    - Vertical wall at x=2
    - Left room: x=1, y in [1,3]
    - Right room: x=3, y in [1,3]
    
    Valid positions:
    - Key: x=1, y in {1,2,3} (left room)
    - Door: x=2, y in {1,2,3} (on wall)
    - Goal: x=3, y in {1,2,3} (right room)
    - Agent start: x=1, y in {1,2,3} (left room, not on key)
    - Agent dir: 0=Right, 1=Down, 2=Left, 3=Up
    """
    
    def __init__(
        self,
        key_pos=(1, 3),
        door_pos=(2, 1),
        goal_pos=(3, 3),
        agent_start=(1, 1),
        agent_dir=0,
        max_steps=100,
        **kwargs
    ):
        """Initialize custom DoorKey environment.
        
        Args:
            key_pos: Position of the key (x, y)
            door_pos: Position of the door (x, y)
            goal_pos: Position of the goal (x, y)
            agent_start: Starting position of the agent (x, y)
            agent_dir: Starting direction (0=Right, 1=Down, 2=Left, 3=Up)
            max_steps: Maximum steps before truncation
        """
        self.key_pos = key_pos
        self.door_pos = door_pos
        self.goal_pos = goal_pos
        self.agent_start_pos = agent_start
        self.agent_start_dir = agent_dir
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        super().__init__(
            mission_space=mission_space,
            grid_size=5,
            max_steps=max_steps,
            **kwargs
        )
    
    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Vertical wall at x=2
        for y in range(0, height):
            self.grid.set(2, y, Wall())
        
        # Place door, key, goal
        self.grid.set(self.door_pos[0], self.door_pos[1], Door('yellow', is_locked=True))
        self.grid.set(self.key_pos[0], self.key_pos[1], Key('yellow'))
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
        
        # Place agent with specified position AND direction
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        
        self.mission = self._gen_mission()


def make_custom_doorkey(key_pos, door_pos, goal_pos, agent_start=(1, 1), agent_dir=0):
    """Create a custom DoorKey environment with specified positions.
    
    Args:
        key_pos: Position of the key (x, y)
        door_pos: Position of the door (x, y)
        goal_pos: Position of the goal (x, y)
        agent_start: Starting position of the agent (x, y)
        agent_dir: Starting direction (0=Right, 1=Down, 2=Left, 3=Up)
        
    Returns:
        CustomDoorKey5x5 environment instance
    """
    env = CustomDoorKey5x5(
        key_pos=key_pos,
        door_pos=door_pos,
        goal_pos=goal_pos,
        agent_start=agent_start,
        agent_dir=agent_dir,
        render_mode='rgb_array'
    )
    return env


def bfs_solve_custom_env(env):
    """BFS solver for custom environment instance.
    
    Args:
        env: Custom environment instance (already reset)
        
    Returns:
        List of actions to reach the goal, or None if unsolvable
    """
    unwrapped = env.unwrapped
    
    DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    start_pos = tuple(unwrapped.agent_pos)
    start_dir = unwrapped.agent_dir
    grid = unwrapped.grid
    
    # Find key and door positions
    key_pos = None
    door_pos = None
    for gx in range(grid.width):
        for gy in range(grid.height):
            cell = grid.get(gx, gy)
            if cell and cell.type == 'key':
                key_pos = (gx, gy)
            if cell and cell.type == 'door':
                door_pos = (gx, gy)
    
    queue = deque([(start_pos[0], start_pos[1], start_dir, [], None, False)])
    visited = set([(start_pos[0], start_pos[1], start_dir, None, False)])
    
    while queue:
        x, y, d, actions, carrying, door_open = queue.popleft()
        
        dx, dy = DIR_TO_VEC[d]
        fx, fy = x + dx, y + dy
        
        if 0 <= fx < grid.width and 0 <= fy < grid.height:
            cell = grid.get(fx, fy)
            
            # Goal reached
            if cell and cell.type == 'goal':
                return actions + [unwrapped.actions.forward]
            
            # Pick up key
            if cell and cell.type == 'key' and carrying is None:
                new_carrying = 'key'
                new_state = (x, y, d, new_carrying, door_open)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((x, y, d, actions + [unwrapped.actions.pickup], new_carrying, door_open))
            
            # Open door (must have key)
            if cell and cell.type == 'door' and not door_open and carrying == 'key':
                new_state = (fx, fy, d, carrying, True)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((fx, fy, d, actions + [unwrapped.actions.toggle, unwrapped.actions.forward], carrying, True))
            
            # Move forward
            cell_is_key_picked = (key_pos == (fx, fy) and carrying == 'key')
            can_move = (cell is None) or \
                       (cell and cell.type == 'door' and door_open) or \
                       (cell and cell.type == 'goal') or \
                       cell_is_key_picked
            
            if can_move:
                state = (fx, fy, d, carrying, door_open)
                if state not in visited:
                    visited.add(state)
                    queue.append((fx, fy, d, actions + [unwrapped.actions.forward], carrying, door_open))
        
        # Turn left/right
        for turn in [-1, 1]:
            td = (d + turn) % 4
            state = (x, y, td, carrying, door_open)
            if state not in visited:
                visited.add(state)
                action = unwrapped.actions.left if turn == -1 else unwrapped.actions.right
                queue.append((x, y, td, actions + [action], carrying, door_open))
    
    return None


def generate_custom_configs(exclude_standard=True):
    """Generate all valid custom DoorKey configurations.
    
    Args:
        exclude_standard: Whether to exclude standard MiniGrid-DoorKey-5x5-v0 pattern
        
    Returns:
        List of configuration dictionaries
    """
    key_positions = [(1, 1), (1, 2), (1, 3)]
    door_positions = [(2, 1), (2, 2), (2, 3)]
    goal_positions = [(3, 1), (3, 2), (3, 3)]
    
    custom_configs = []
    
    for key_pos in key_positions:
        for door_pos in door_positions:
            for goal_pos in goal_positions:
                # Exclude standard MiniGrid-DoorKey-5x5-v0 pattern
                if exclude_standard and door_pos == (2, 2) and goal_pos == (3, 3):
                    continue
                
                # Find valid agent start position (not on key)
                valid_agent_starts = [(1, y) for y in [1, 2, 3] if (1, y) != key_pos]
                agent_start = valid_agent_starts[0]
                
                custom_configs.append({
                    "key_pos": key_pos,
                    "door_pos": door_pos,
                    "goal_pos": goal_pos,
                    "agent_start": agent_start
                })
    
    return custom_configs

