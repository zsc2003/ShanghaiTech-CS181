from utils.config import WHITE, BLACK, POSI_INFI, NEGA_INFI
from utils.config import NEGA_INFI, POSI_INFI
from algorithms.algorithms import get_all_moves, simulate_move, show_path

def minimax(path_num, current_board, color, depth, game):
    if depth == 0 or current_board.winner():
        return current_board.evaluate(game.my_turn), current_board
    
    best_move = None

    # convert turn (color)
    if color == WHITE:
        other_color = BLACK
    else:
        other_color = WHITE
    if color != game.my_turn:
        max_score = NEGA_INFI
        for move in get_all_moves(path_num, current_board, color, game):
            score = minimax(path_num, move, other_color, depth - 1, game)[0]
            max_score = max(score, max_score)
            if max_score == score:
                best_move = move
        return max_score, best_move
    else:
        min_score = POSI_INFI
        for move in get_all_moves(path_num, current_board, color, game):
            score = minimax(path_num, move, other_color, depth - 1, game)[0]
            min_score = min(score,min_score)
            if min_score == score:
                best_move = move
        return min_score, best_move

    
def alpha_beta_pruning(path_num, current_board, color, alpha, beta, depth, game):
    '''
    alpha-beta pruning (based on negamax)
    '''

    # depth = 0 or someone wins
    if depth == 0 or current_board.winner() != None:
        return current_board.evaluate(game.my_turn), current_board
    
    # convert turn (color)
    if color == WHITE:
        other_color = BLACK
    else:
        other_color = WHITE
    best_move = None
    moves = get_all_moves(path_num, current_board, color, game)
    if moves:
        for move in moves:
            score = -alpha_beta_pruning(path_num, move, other_color, -beta, -alpha, depth - 1, game)[0]
            if score >= beta:
                return beta, best_move
            if score > alpha:
                alpha = score
                best_move = move
        return alpha, best_move
    else:
        # no place to move -> -inf
        return 1 + NEGA_INFI, best_move