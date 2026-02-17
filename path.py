import heapq
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import json

with open("map.json") as f:
    grid = json.load(f)

start = (0, 0)
goal = (len(grid) - 1, len(grid[0]) - 1)

def bfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([start])
    visited = {start}
    came_from = {start: None}

    nodes_explored = 0
    max_frontier_size = 0

    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    while queue:
        max_frontier_size = max(max_frontier_size, len(queue))

        current = queue.popleft()
        nodes_explored += 1

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], nodes_explored, max_frontier_size

        for dx, dy in directions:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] == 0 and neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

    return None, nodes_explored, max_frontier_size


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def A_star(grid, start, goal, heuristic=manhattan):
    rows, cols = len(grid), len(grid[0])

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))

    open_set = {start}
    closed_set = set()

    came_from = {start: None}
    g_score = {start: 0}

    nodes_explored = 0
    max_frontier_size = 0

    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    while open_heap:
        max_frontier_size = max(max_frontier_size, len(open_heap))

        _, current_g, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        nodes_explored += 1

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], nodes_explored, max_frontier_size

        open_set.discard(current)
        closed_set.add(current)

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor[0]][neighbor[1]] != 0:
                continue
            if neighbor in closed_set:
                continue

            tentative_g = current_g + 1

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))
                    open_set.add(neighbor)

    return None, nodes_explored, max_frontier_size


bfs_path, bfs_nodes, bfs_frontier = bfs(grid, start, goal)
astar_path, astar_nodes, astar_frontier = A_star(grid, start, goal)

grid_np = np.array(grid)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(grid_np, cmap="Greys")
if bfs_path:
    bp = np.array(bfs_path)
    plt.plot(bp[:,1], bp[:,0], color="blue", linewidth=2)
plt.scatter(start[1], start[0], color="green", s=80)
plt.scatter(goal[1], goal[0], color="red", s=80)
plt.title(f"BFS\nNodes: {bfs_nodes}\nMax frontier: {bfs_frontier}")

plt.subplot(1, 2, 2)
plt.imshow(grid_np, cmap="Greys")
if astar_path:
    ap = np.array(astar_path)
    plt.plot(ap[:,1], ap[:,0], color="red", linewidth=2)
plt.scatter(start[1], start[0], color="green", s=80)
plt.scatter(goal[1], goal[0], color="red", s=80)
plt.title(f"A*\nNodes: {astar_nodes}\nMax frontier: {astar_frontier}")

plt.suptitle("BFS vs A* – Correct Time & Space Comparison")
plt.tight_layout()
plt.show()
