import pygame
from utils.config import ROWS, COLS, SIZE, BLACK, WHITE, WIDTH, HEIGHT, BLUE, GREEN, GREY1, GREY2, WHITE_CROWN, BLACK_CROWN

class Piece:
    PADDING = 10 # distance between chess pieces and the outline of the chess grid

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.is_king = False
        self.x = SIZE * self.col + SIZE // 2 # chess piece's x-axis coordinate
        self.y = SIZE * self.row + SIZE // 2 # chess piece's y-axis coordinate

    def draw_piece(self, window):
        pygame.draw.circle(window, GREY2, (self.x, self.y), SIZE // 2 - self.PADDING + 2)
        pygame.draw.circle(window,self.color, (self.x,self.y), SIZE // 2 - self.PADDING)
        if self.is_king:
            if self.color == WHITE: 
                window.blit(WHITE_CROWN, (self.x - WHITE_CROWN.get_width() // 2, self.y - WHITE_CROWN.get_height() // 2 - 3))
            else:
                window.blit(BLACK_CROWN, (self.x - BLACK_CROWN.get_width() // 2, self.y - BLACK_CROWN.get_height() // 2 - 3))
    
    def move_piece(self, row, col):
        self.row = row
        self.col = col
        self.x = SIZE * self.col + SIZE // 2
        self.y = SIZE * self.row + SIZE // 2
    
    def get_dist(self):
        return 1 / (8 - self.row)

    def get_color(self):
        return self.color