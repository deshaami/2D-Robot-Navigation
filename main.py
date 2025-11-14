import pygame
import numpy as np
import heapq

# -------------------------------
# Grid Map and Start/Goal
# -------------------------------
grid = np.array([
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0]
])

start = (0, 0)  # (row, col)
goal = (5, 5)   # (row, col)

# -------------------------------
# Pygame Setup
# -------------------------------
pygame.init()

CELL_SIZE = 60
GRID_HEIGHT, GRID_WIDTH = grid.shape
WINDOW_SIZE = (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("2D Robot Navigation Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# -------------------------------
# Draw Functions
# -------------------------------
def draw_grid():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if grid[y][x] == 0 else BLACK  # 0 = free, 1 = obstacle
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # grid lines

def draw_robot(position):
    y, x = position
    pygame.draw.circle(screen, RED, (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3)

def draw_goal(position):
    y, x = position
    pygame.draw.rect(screen, GREEN, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

# -------------------------------
# A* Algorithm
# -------------------------------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            tentative_g = gscore[current] + 1

            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                continue

            if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return []  # no path

# -------------------------------
# Initialize Robot and Variables
# -------------------------------
robot_pos = start
running = True

# -------------------------------
# Main Loop
# -------------------------------
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            gx = mx // CELL_SIZE
            gy = my // CELL_SIZE
            # toggle obstacle on click
            grid[gy][gx] = 1 - grid[gy][gx]

    screen.fill(WHITE)
    draw_grid()
    draw_goal(goal)

    path = astar(grid, robot_pos, goal)

    if path and len(path) > 1:
        next_pos = path[1]
        if grid[next_pos[0]][next_pos[1]] == 0:
            robot_pos = next_pos

    draw_robot(robot_pos)

    pygame.display.flip()
    clock.tick(5)  # slow down to 5 FPS for visible movement

pygame.quit()
