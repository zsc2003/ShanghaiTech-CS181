import pygame
from copy import deepcopy

def get_all_moves(path_num, board, color, game):
    boards = []
    valid_moves = board.get_valid_moves(color)
    for piece in board.get_all_pieces(color): 
        if piece in valid_moves:
            path_num[0] += 1
            moves = valid_moves[piece]
            show_path(path_num[0], piece, board, game, moves)
            for move,skipped in moves.items():
                temp_board = deepcopy(board)
                temp_piece = temp_board.pieces[piece.row][piece.col]
                new_board = simulate_move(temp_piece, move,temp_board, skipped)
                boards.append(new_board)
    return boards


def simulate_move(piece, move, board, skip):
    '''
    simulate the calculate process for the computer
    obtain all the chessboards generated after AI simulation movement
    '''
    board.move_piece(piece, move[0], move[1])
    if skip:
        board.remove_pieces(skip)
    return board
    

def show_path(path_num, piece, board, game, valid_moves):
    board.draw_board(game.window)
    game.draw_turn()
    game.draw_path_num(path_num)
    pygame.draw.circle(game.window, (0, 255, 0), (piece.x, piece.y), 40, 4)
    game.draw_valid_moves(valid_moves.keys())
    pygame.display.update()