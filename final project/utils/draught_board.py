import pygame
from utils.config import ROWS, COLS, SIZE, BLACK, WHITE, WIDTH, HEIGHT, BLUE, GREEN, GREY1, GREY2, WHITE_CROWN, BLACK_CROWN
from utils.config import dir_move, dir_jump 
from utils.draught_piece import Piece
import sys
sys.path.append(r'C:\Users\wangquan\Desktop\workspace\cs181\ShanghaiTech-CS181-Final-Project\algorithm')
from algorithms.neural_feature import feature_network
import torch

class Board:
    def __init__(self):
        self.pieces      = []                    # store all pieces
        self.white_kings = self.black_kings = 0  # pieces of being king
        self.white_left  = self.black_left  = 12 # pieces remained
        self.is_jump     = False                 # whether a piece have to eat another piece
        self.max_eat     = 0                     # the maximum pieces that can be eaten
        self.init_pieces()
        self.use_feature_network = False
        self.neural_weight = 0.1
        
        if self.use_feature_network:
            self.feature_network = feature_network()
            self.feature_network.cuda()
            self.feature_network.load_state_dict(torch.load(r'C:\Users\wangquan\Desktop\workspace\cs181\ShanghaiTech-CS181-Final-Project\algorithms\ckpt\feature_net_epoch_20.pth'))
    
    def init_pieces(self):
        for row in range(ROWS):
            self.pieces.append([])
            for col in range(COLS):
                if (row + 1) % 2 == col % 2: # white, black, white, black, ......
                    if row < 3:
                        self.pieces[row].append(Piece(row, col, BLACK))
                    elif row < 5:
                        self.pieces[row].append(0)
                    else:
                        self.pieces[row].append(Piece(row, col, WHITE))
                else:
                    self.pieces[row].append(0)
    
    def draw_board(self, window):
        window.fill(GREY1)
        
        # white, black, white, black, ......
        for row in range(ROWS):
            for col in range((row + 1) % 2, ROWS, 2): 
                pygame.draw.rect(window, GREEN, (row * SIZE, col * SIZE, SIZE, SIZE))

        for i in range(len(self.pieces)):
            for j in range(len(self.pieces[i])):
                if self.pieces[i][j]:
                    self.pieces[i][j].draw_piece(window)
   
    # piece raise to king 
    def make_king(self, row, piece):
        if row == ROWS - 1 or row == 0:
            piece.is_king = True
            if piece.color == WHITE:
                self.white_kings += 1
            else:
                self.black_kings += 1 
    
    def move_piece(self, piece, row, col):
        '''
        move the piece to where it will move
        if it reach the end, it raise to the king
        '''
        self.pieces[piece.row][piece.col], self.pieces[row][col] = self.pieces[row][col], self.pieces[piece.row][piece.col]
        piece.move_piece(row, col)
        self.make_king(row, piece)

    def remove_pieces(self, skipped):
        '''
        remove the pieces if they were eaten
        '''
        for p in skipped:
            self.pieces[p.row][p.col] = 0
            if p.color == WHITE:
                self.white_left -= 1
            if p.color == BLACK:
                self.black_left -= 1
    
    # judge whether still in the board
    def is_in_board(self, row, col):
        if row < 0 or row >= ROWS or col < 0 or col >= COLS:
            return False
        return True
    
    # calculate the piece after eating other pieces 
    def try_to_jump(self, row, col, is_king, color, step_num, skipped = []):
        global dir_jump
        moves = {}
        piece = self.pieces[row][col]

        # calculate where the piece could go
        if is_king:
            start = 0
            stop = 4
        elif color == WHITE:
            start = 0
            stop = 2
        else:
            start= 2
            stop = 4

        for i in range(start,stop):
            new_row = row + dir_jump[i][0]
            new_col = col + dir_jump[i][1]
            mid_row = (row + new_row) //2
            mid_col = (col + new_col) //2
            if self.is_in_board(new_row,new_col):
                new_piece = self.pieces[new_row][new_col]
            else:
                continue
            
            # could eat other pieces
            if new_piece == 0 and self.pieces[mid_row][mid_col] and self.pieces[mid_row][mid_col].color != color:        
                if self.pieces[mid_row][mid_col] in skipped: 
                   continue
                self.is_jump = True

                # must eat the most
                if step_num > self.max_eat:
                    self.max_eat = step_num
                last = skipped.copy()
                last += [self.pieces[mid_row][mid_col]]

                moves.update(self.try_to_jump(new_row, new_col,is_king,color,step_num + 1,last))
                
        if skipped:
            moves[(row,col)] = skipped
            
        return moves
    
    # pieces could go
    def try_to_move(self, row, col):
        global dir_move
        moves = {}
        piece = self.pieces[row][col]

        if piece.is_king:
            start = 0
            stop = 4
        elif piece.color == WHITE:
            start = 0
            stop = 2
        else:
            start = 2
            stop = 4
        for i in range(start, stop):
            new_row = row + dir_move[i][0]
            new_col = col + dir_move[i][1]
            if self.is_in_board(new_row, new_col):
                new_piece = self.pieces[new_row][new_col]
            else:
                continue
            if new_piece == 0:
                moves[(new_row,new_col)] = []

        return moves
    
    
    # decide if exist a winner
    def winner(self):
        if self.white_left <= 0:
            return BLACK
        elif self.black_left <= 0:
            return WHITE
        else:
            return None
    
    # access all pieces
    def get_all_pieces(self, color):
        pieces = []
        for row in self.pieces:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces
    
    # get all valid movements
    def get_valid_moves(self, color):
        valid_moves = {} 
        self.is_jump = False

        # try to eat
        for piece in self.get_all_pieces(color):
            moves = {}
            moves.update(self.try_to_jump(piece.row, piece.col, piece.is_king, piece.color, 1))
            if moves:
                valid_moves[piece] = moves

        # cannot eat
        if not self.is_jump: 
            for piece in self.get_all_pieces(color):
                moves = {}
                moves.update(self.try_to_move(piece.row, piece.col))
                if moves:
                    valid_moves[piece] = moves
            
        all_valid_moves = {} # paths that eat the most
        if self.is_jump:
            for piece, moves in valid_moves.items():
                all_valid_moves_update = {}
                for move, skipped in moves.items():
                    new_moves = {}
                    if skipped:
                        # take the path that eat the most
                        if len(skipped) == self.max_eat:
                            new_moves[move] = skipped  
                            all_valid_moves_update.update(new_moves) 
                            all_valid_moves[piece] = all_valid_moves_update
        else:
            all_valid_moves = valid_moves
        return all_valid_moves

    # evaluate score function
    def evaluate(self, color):
        score = self.white_left - self.black_left + (self.white_kings * 0.5 - self.black_kings * 0.5)
        if color == WHITE:
            return -score
        else:
            return score
        
    def evaluate_neural(self, color):
        neural_board = torch.zeros((1, 1, 8, 8)).cuda()
        for i in range(8):
            for j in range(8):
                if self.pieces[i][j] == 0:
                    neural_board[0][0][i][j] = 0
                elif self.pieces[i][j].color == WHITE:
                    if self.pieces[i][j].is_king:
                        neural_board[0][0][i][j] = 3
                    else:
                        neural_board[0][0][i][j] = 1
                else:
                    if self.pieces[i][j].is_king:
                        neural_board[0][0][i][j] = 4
                    else:
                        neural_board[0][0][i][j] = 2
        neural_board = neural_board.float()
        neural_turn = torch.tensor([1 if color == WHITE else 2]).float().cuda()
        neural_score = self.feature_network(neural_board, neural_turn)
        # to numpy
        neural_score = neural_score.cpu().detach().numpy()[0][0]
        score = self.white_left - self.black_left + (self.white_kings * 0.5 - self.black_kings * 0.5)
        score += self.neural_weight * neural_score
        if color == WHITE:
            return -score
        else:
            return score
        
    # evaluate the board score without consider turn
    def board_mcts_eval(self):
        score = self.white_left + 2 * self.white_kings - self.black_left - 2 * self.black_kings
        return score

    def distance(self):
        #white
        result = 0
        for i in range(8):
            for j in range(8):
                if(self.pieces[i][j] != 0 and self.pieces[i][j].get_color() == WHITE):
                    result += self.pieces[i][j].get_dist()
        return result


    def evaluate_dist(self,color):
        # raise RuntimeError(self.pieces)
        result = 0

        if(self.white_left == 0):
            return -10000
        if(self.black_left == 0):   
            return 10000

        for i in range(8):
            for j in range(8):
                if(self.pieces[i][j] != 0 and self.pieces[i][j].get_color() == WHITE):
                    result += self.pieces[i][j].get_dist()
        
        score = self.white_left - self.black_left + (self.white_kings * 2 - self.black_kings * 2)
        return result + score