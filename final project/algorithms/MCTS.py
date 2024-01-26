from algorithms.algorithms import *
import numpy as np
import random, pickle
import copy
import math
from utils.config import *

class mcts_node():
    def __init__(self, color = WHITE):
        self.parent = None
        self.children = []
        self.next_moves = None
        self.turn = color
        self.board = None
        self.N = 0
        self.Q = 0
        
    def reverse_node(self):
        if self.turn == WHITE:
            return BLACK
        elif self.turn == BLACK:
            return WHITE
        else:
            raise ValueError("Error turn in reverse_node")
        
    def get_next(self):
        # get all moves
        valid_moves = self.board.get_valid_moves(self.turn)
        child_nodes = []
        # loop all valid moves and generate new board
        for piece in self.board.get_all_pieces(self.turn):
            # simulate move
            if piece in valid_moves:
                moves = valid_moves[piece]
                for move, skipped in moves.items():
                    child_board = copy.deepcopy(self.board)
                    child_piece = child_board.pieces[piece.row][piece.col]
                    child_board.move_piece(child_piece, move[0], move[1])
                    if skipped:
                        child_board.remove_pieces(skipped)
                    
                    # generate new node
                    child_node = mcts_node()
                    child_node.turn = self.reverse_node()
                    child_node.parent = self
                    child_node.N = 0
                    child_node.Q = 0
                    child_node.board = child_board
                    # append
                    child_nodes.append(child_node)
        
        # print(child_nodes)
        
        # return
        return child_nodes

# uct algorithm
def uct(node):
    best_score = float('-inf')
    best_node = None
    # loop all node in the children
    for child_node in node.children:
        # compute the ucb value
        ucb1_v1 = child_node.Q / child_node.N
        ucb1_v2 = (1 / math.sqrt(2)) * math.sqrt(2 * math.log(child_node.parent.N) / child_node.N)
        score = ucb1_v1 + ucb1_v2
        if score > best_score:
            best_score = score
            best_node = child_node
    return best_node

def selection(node):
    while node.children and all(child.N > 0 for child in node.next_moves):
        if node.next_moves is None:
            node.next_moves = node.get_next()

        node = uct(node)

    return node

def expansion(parent):
    # if no child
    if not parent.next_moves:
        return parent

    # if child not in children
    for move in parent.next_moves:
        if all(move.board.pieces != child.board.pieces for child in parent.children):
            move.next_moves = move.get_next() if move.next_moves is None else move.next_moves
            move.turn = parent.reverse_node() if move.next_moves else parent.turn
            parent.children.append(move)
            return move

    return None

def simulation(node):
    for _ in range(20):
        if node.board.winner() is not None:
            break

        valid_moves = node.get_next()
        node = random.choice(valid_moves) if valid_moves else node

    return node.board.board_mcts_eval() > 0
    
def backpropagation(node, winner):
    while node:
        node.N += 1
        if (node.turn and not winner) or (not node.turn and winner):
            node.Q += 1
        node = node.parent
        
class mcts_agent():
    def __init__(self, board, color):
        self.root = mcts_node(color)
        self.root.board = copy.deepcopy(board)
        self.root.next_moves = self.root.get_next()
        
    def step(self, board, iterations):
        self.root.board = copy.deepcopy(board)
        self.root.next_moves = self.root.get_next()
        self.root.children = []
        
        # simulation
        for _ in range(iterations):
            # perform mcts 4 steps
            current_node = selection(self.root)
            new_node = expansion(current_node) or current_node
            white_win = simulation(new_node)
            backpropagation(new_node, white_win)

        # find the most explored
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda child: child.N)
        return best_child.board