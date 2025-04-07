# Path Planning Algorithms for Autonomous Robots

*Published: July 30, 2023*

## Introduction

Path planning is a fundamental challenge in robotics that involves finding a viable trajectory from a starting position to a goal position while avoiding obstacles. Effective path planning enables autonomous robots to navigate safely and efficiently in complex environments. This article explores key path planning algorithms, their implementations, and their applications in modern robotics.

## Classical Path Planning Algorithms

### A* Algorithm

The A* algorithm is one of the most popular path planning methods due to its efficiency and optimality guarantees. It uses a heuristic function to guide the search towards the goal while accounting for the cost incurred so far.

```python
import heapq
import numpy as np

def a_star(grid, start, goal):
    """
    A* path planning algorithm implementation
    
    Args:
        grid: 2D binary array (1: obstacle, 0: free)
        start: tuple (x, y) of start position
        goal: tuple (x, y) of goal position
        
    Returns:
        path: list of (x, y) positions from start to goal
    """
    # Define possible movements (8-connectivity)
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0), 
             (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    # Cost of each move (diagonal moves cost more)
    costs = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
    
    # Initialize data structures
    open_set = []  # Priority queue
    closed_set = set()
    g_scores = {start: 0}  # Cost from start to current
    f_scores = {start: heuristic(start, goal)}  # Estimated total cost
    came_from = {}  # For path reconstruction
    
    # Push start node to open set
    heapq.heappush(open_set, (f_scores[start], start))
    
    while open_set:
        # Get node with lowest f_score
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Goal reached, reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse to get start-to-goal
        
        closed_set.add(current)
        
        # Explore neighbors
        for i, (dx, dy) in enumerate(moves):
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check boundaries
            if (neighbor[0] < 0 or neighbor[0] >= grid.shape[0] or 
                neighbor[1] < 0 or neighbor[1] >= grid.shape[1]):
                continue
                
            # Check if obstacle or already visited
            if grid[neighbor] == 1 or neighbor in closed_set:
                continue
                
            # Calculate tentative g_score
            tentative_g = g_scores[current] + costs[i]
            
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                # This path is better than any previous one
                came_from[neighbor] = current
                g_scores[neighbor] = tentative_g
                f_scores[neighbor] = g_scores[neighbor] + heuristic(neighbor, goal)
                
                if neighbor not in [node[1] for node in open_set]:
                    heapq.heappush(open_set, (f_scores[neighbor], neighbor))
    
    # No path found
    return None

def heuristic(a, b):
    """Euclidean distance heuristic"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
```

### Rapidly-exploring Random Tree (RRT)

RRT is a sampling-based algorithm particularly effective for high-dimensional configuration spaces. It builds a tree by randomly sampling points and expanding towards them.

```python
import random
import numpy as np

def rrt(start, goal, obstacle_check, bounds, max_iterations=1000, step_size=0.1):
    """
    RRT path planning algorithm
    
    Args:
        start: Start configuration
        goal: Goal configuration
        obstacle_check: Function that returns True if a configuration is collision-free
        bounds: Tuple of (min_bounds, max_bounds) for random sampling
        max_iterations: Maximum number of iterations
        step_size: Distance to extend tree in each step
        
    Returns:
        path: List of configurations from start to goal
    """
    # Initialize tree with start node
    tree = {tuple(start): None}  # Node: Parent
    
    for i in range(max_iterations):
        # Sample random configuration
        if random.random() < 0.05:  # Bias towards goal
            random_point = goal
        else:
            random_point = sample_free_space(bounds)
            
        # Find nearest node in tree
        nearest_node = find_nearest(tree, random_point)
        
        # Extend tree towards random point
        new_node = extend(nearest_node, random_point, step_size)
        
        # Check if extension is valid
        if obstacle_check(new_node):
            tree[tuple(new_node)] = tuple(nearest_node)
            
            # Check if goal is reached
            if np.linalg.norm(np.array(new_node) - np.array(goal)) < step_size:
                # Construct path
                tree[tuple(goal)] = tuple(new_node)
                path = []
                node = tuple(goal)
                
                while node is not None:
                    path.append(node)
                    node = tree[node]
                    
                return path[::-1]  # Return path from start to goal
    
    # No path found within iterations
    return None

def sample_free_space(bounds):
    """Sample random point within bounds"""
    min_bounds, max_bounds = bounds
    return [random.uniform(min_bounds[i], max_bounds[i]) for i in range(len(min_bounds))]

def find_nearest(tree, point):
    """Find nearest node in tree to the given point"""
    return min(tree.keys(), key=lambda node: np.linalg.norm(np.array(node) - np.array(point)))

def extend(from_node, to_node, step_size):
    """Extend from 'from_node' towards 'to_node' with step_size"""
    from_array = np.array(from_node)
    to_array = np.array(to_node)
    
    direction = to_array - from_array
    norm = np.linalg.norm(direction)
    
    if norm < step_size:
        return tuple(to_array)
    else:
        return tuple(from_array + step_size * direction / norm)
```

## Advanced Path Planning Techniques

### Probabilistic Roadmap (PRM)

PRM pre-computes a roadmap of the environment by randomly sampling collision-free configurations and connecting them with a local planner. This makes it efficient for multi-query planning.

### Potential Field Methods

Potential field methods model the robot as a point in a potential field where the goal generates an attractive force while obstacles generate repulsive forces.

```python
def potential_field_planner(start, goal, obstacles, attraction_gain=1.0, repulsion_gain=100.0, 
                           max_iterations=1000, step_size=0.1, obstacle_influence_range=2.0):
    """
    Potential field path planner
    
    Args:
        start: Start position [x, y]
        goal: Goal position [x, y]
        obstacles: List of obstacle positions [[x1, y1], [x2, y2], ...]
        attraction_gain: Weight of attractive potential
        repulsion_gain: Weight of repulsive potential
        max_iterations: Maximum number of steps
        step_size: Size of each step
        obstacle_influence_range: Range of influence of obstacles
        
    Returns:
        path: List of positions from start to goal
    """
    path = [start]
    current = np.array(start)
    
    for _ in range(max_iterations):
        # If close to goal, return path
        if np.linalg.norm(current - np.array(goal)) < step_size:
            path.append(goal)
            return path
        
        # Calculate forces
        f_att = attraction_force(current, goal, attraction_gain)
        f_rep = np.array([0.0, 0.0])
        
        for obs in obstacles:
            f_rep += repulsion_force(current, obs, repulsion_gain, obstacle_influence_range)
        
        # Calculate total force and normalize
        f_total = f_att + f_rep
        norm = np.linalg.norm(f_total)
        
        if norm < 1e-5:  # If force is too small
            break
            
        f_total = f_total / norm
        
        # Update position
        current = current + step_size * f_total
        path.append(current.tolist())
    
    return path

def attraction_force(current, goal, gain):
    """Calculate attraction force towards goal"""
    return gain * (np.array(goal) - current)

def repulsion_force(current, obstacle, gain, influence_range):
    """Calculate repulsion force from obstacle"""
    diff = current - np.array(obstacle)
    dist = np.linalg.norm(diff)
    
    if dist > influence_range:
        return np.array([0.0, 0.0])
    
    # Repulsion increases as distance decreases
    return gain * (1.0/dist - 1.0/influence_range) * diff / (dist**3) if dist > 0 else np.array([1.0, 0.0]) * gain
```

## Dynamic and Adaptive Planning

### D* Lite Algorithm

D* Lite is particularly useful for planning in partially known or changing environments, allowing efficient replanning when new obstacles are detected.

### Model Predictive Control (MPC)

MPC integrates path planning with control by continuously replanning over a receding horizon while considering vehicle dynamics.

## Multi-Robot Path Planning

Coordinating multiple robots introduces new challenges, requiring algorithms that avoid collisions between robots while optimizing overall efficiency.

### Prioritized Planning

In prioritized planning, robots are assigned priorities and plan sequentially, with higher-priority robots planning first and lower-priority robots avoiding them.

### Conflict-Based Search (CBS)

CBS is a multi-agent pathfinding algorithm that uses a two-level search to find optimal paths for all agents while resolving conflicts.

## Implementation Considerations

### Environment Representation

Different environment representations suit different algorithms:
- **Occupancy Grids**: Discretize space into cells (occupied/free)
- **Visibility Graphs**: Connect obstacle vertices with visible lines
- **Voronoi Diagrams**: Create paths equidistant from obstacles

### Kinodynamic Constraints

Real robots have physical limitations on velocity, acceleration, and turning radius that must be incorporated into path planning.

### Dealing with Uncertainty

Robust planning must account for:
- Sensor uncertainty
- Motion uncertainty
- Environmental changes

## Applications in Modern Robotics

### Autonomous Vehicles

Self-driving cars use hierarchical planning systems combining high-level route planning, behavioral planning, and low-level trajectory planning.

### Warehouse Robots

Warehouse automation relies on efficient path planning for hundreds of robots operating simultaneously.

### Drone Navigation

UAVs require 3D path planning that accounts for aerodynamic constraints and varying altitudes.

## Conclusion

Path planning is a rich field with diverse approaches suited to different robotics applications. Classical algorithms like A* provide optimality guarantees for structured environments, while sampling-based methods like RRT excel in high-dimensional spaces. Dynamic replanning algorithms adapt to changing conditions, making them ideal for real-world applications. As robots continue to advance and operate in increasingly complex environments, path planning algorithms will remain a critical component of autonomous systems.

## References

1. LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press.
2. Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. *The International Journal of Robotics Research*, 30(7), 846-894.
3. Koenig, S., & Likhachev, M. (2002). D* Lite. *AAAI Conference on Artificial Intelligence*, 476-483.
4. Kavraki, L. E., Svestka, P., Latombe, J. C., & Overmars, M. H. (1996). Probabilistic roadmaps for path planning in high-dimensional configuration spaces. *IEEE Transactions on Robotics and Automation*, 12(4), 566-580.

---

*Tags: robotics, path planning, autonomous navigation, algorithms, A*, RRT* 