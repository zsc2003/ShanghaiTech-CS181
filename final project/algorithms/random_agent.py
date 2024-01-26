from algorithms.algorithms import *
import random

def random_algorithm(path_num, current_board, color, depth, game):
    # someone wins
    if current_board.winner() != None:
        return current_board.evaluate(game.my_turn), current_board
    moves = get_all_moves(path_num, current_board, color, game)
    num = len(moves)
    move = moves[random.randint(0, num - 1)]
    return move.evaluate(game.my_turn), move