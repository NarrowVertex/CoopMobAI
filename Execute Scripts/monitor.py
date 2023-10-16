import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1920, 1080
TILE_SIZE = 40
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load the map data from a text file
def load_map(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [list(map(int, line.strip())) for line in lines]


# Initialize the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Map Drawing")

# Load the map datau
map_data = load_map("../Game Data/map/empty_map.txt")

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw the map
    for y, row in enumerate(map_data):
        for x, tile in enumerate(row):
            if tile == 1:
                pygame.draw.rect(screen, BLACK, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
