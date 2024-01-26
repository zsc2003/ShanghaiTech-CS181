import pygame

# infinate
POSI_INFI = 2147483647
NEGA_INFI = -2147483647

# colors
WHITE = (255,255,255)
BLUE  = (0,0,255)
BLACK = (127,0,0)
GREY1 = (209,203,180)
GREY2 = (194,194,194)
GREEN = (114,139,114)

# chess board info
WIDTH, HEIGHT = 640, 680  # size for window
ROWS, COLS = 8, 8         # size for the chess board
SIZE = 80                 # width of a single chess borad piece

# images for crowns
WHITE_CROWN = pygame.transform.scale(pygame.image.load('images/crown_new.png'), (44, 25))
BLACK_CROWN = pygame.transform.scale(pygame.image.load('images/crown_new.png'), (44, 25))

# movements for pieces
dir_move = [[-1,-1], [-1,1], [1,-1], [1,1]]
dir_jump = [[-2,-2], [-2,2], [2,-2], [2,2]]