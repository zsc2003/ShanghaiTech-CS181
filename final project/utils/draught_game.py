import pygame
from utils.config import ROWS, COLS, SIZE, BLACK, WHITE, WIDTH, HEIGHT, BLUE, GREEN, GREY1, GREY2, WHITE_CROWN, BLACK_CROWN
from utils.draught_board import Board

class Game:
    def __init__(self, window, my_turn):
        self.init(window, my_turn)

    def init(self, window, my_turn):
        self.selected_piece  = None     # whether the piece is selected by user
        self.turn            = WHITE    # current turn
        self.all_valid_moves = {}       # valid moves for current turn
        self.valid_moves     = {}       # current selected piece's valid moves
        self.board           = Board()
        self.window          = window
        self.my_turn         = my_turn
    
    def winner(self):
        return self.board.winner()

    # update game window
    def update(self):
        self.board.draw_board(self.window)
        self.draw_valid_moves(self.valid_moves)
        self.draw_turn()
        self.draw_path_num()
        pygame.display.update()

    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            pygame.draw.circle(self.window, BLUE, (col * SIZE + SIZE // 2, row * SIZE + SIZE // 2), 15)
    
    def draw_turn(self):
        if self.turn == WHITE:
            turn_text = 'Turn : WHITE'
        else:
            turn_text = 'Turn : BLACK'
        font = pygame.font.SysFont('simhei', 30)
        turn = font.render(turn_text, True, (0, 0, 0))
        self.window.blit(turn, (30, HEIGHT - 35))
    
    # calculate the number that the algorithm has calculated
    def draw_path_num(self,num = 0):
        if self.turn == self.my_turn:
            text = 'Path Num : -'
        else:
            text = 'Path Num : ' + str(num)
        font = pygame.font.SysFont('simhei', 30)
        text_num = font.render(text, True, (0, 0, 0))
        self.window.blit(text_num, (WIDTH/2 + 30, HEIGHT - 35))
    
    def draw_winner(self):
        if self.board.winner() == None:
            return
        if  self.board.winner() == WHITE:
            text = 'The WHITE side wins! GAME OVER !!!'
        else:
            text = 'The BLACK side wins! GAME OVER !!!'
        font = pygame.font.SysFont('simhei', 32)
        gameover_text = font.render(text, True, (255, 0, 0))
        self.window.blit(gameover_text, (WIDTH / 4,HEIGHT / 2 - 20))
        pygame.display.update()
        pygame.time.delay(5000)
        
    
    def after_click(self, row, col): 
        '''
        If there are already selected pieces on the chessboard, move that piece;
        On the contrary, select the chess piece.
        When the move fails (the selected move position does not comply with the game rules):
        Reselect chess pieces
        ''' 
        if self.selected_piece:
            if not self.move(row, col):
                self.selected_piece = None
                self.after_click(row, col)
        else:                   
            piece = self.board.pieces[row][col]
            if piece != 0 and (piece.color == self.turn):
                self.selected_piece = piece
                if piece in self.all_valid_moves:
                    self.valid_moves = self.all_valid_moves[piece]
                else:
                    self.valid_moves = {}

    def get_moves(self):
        self.all_valid_moves = self.board.get_valid_moves(self.turn)

    # movement logic
    def move(self, row, col):
        '''
        If the chess piece cannot be moved to the selected position, the move fails and returns False
        On the contrary, move the chess piece to the selected position and check if it has been eaten.
        The eaten chess piece will be deleted. If the move is successful, return True
        '''
        piece = self.board.pieces[row][col]
        if self.selected_piece and piece == 0 and self.valid_moves and (row, col) in self.valid_moves:
            self.board.move_piece(self.selected_piece, row, col)
            if self.board.is_jump:
                skipped = self.valid_moves[(row, col)]
                self.board.remove_pieces(skipped)
            self.change_turn()
            return True
        return False
    
    def change_turn(self):
        self.board.max_eat = 0
        self.valid_moves = {}
        self.all_valid_moves = {}
        self.selected_piece = None
        if self.turn == WHITE:
            self.turn = BLACK
        else:
            self.turn = WHITE
    
    def ai_move(self, board):
        self.board = board
        self.change_turn()    