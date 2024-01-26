import pygame
import pygame_menu
import sys
from utils.config import *
from utils.draught_game import Game
from utils.draught_board import Board
from algorithms.minimax import minimax
from algorithms.algorithms import *

import random
import pickle

depth = 4
ai_turn = BLACK
algorithm = 1

def select_algorithm(value, index):
    global algorithm
    algorithm = index

def select_turn(value, index):
    global ai_turn
    if index == 2:
        ai_turn = WHITE
    else:
        ai_turn = BLACK

def set_depth(value):
    global depth
    if value.isdigit():
        value_ = int(value)
        if value_ % 2:
            depth = value_ - 1
        else:
            depth = value_

alpha = 0.4
discount_factor = 0.8
depth = 2

def train_qlearning():
    global qvalue_grid
    qvalue_grid = {}    # Q-values
    epochs = 50         # train epoch
    epsilon = 0.4       # exploration 
    i = 0

    while(i < epochs):
        i += 1
        game = Game(win,WHITE)
        is_run = True
        count_step = 0
        while is_run:
            count_step += 1
            if(count_step > 200):
                break     
            valid_moves = get_all_moves([0],game.board,WHITE,game)

            max_action = []
            max_q_value = -9999
            for num_board in range(len(valid_moves)):
                new_board = valid_moves[num_board]
                if((game.board, new_board) not in qvalue_grid.keys()):
                    qvalue_grid[(game.board, new_board)] = 0

                max_for_next_step = -9999   
                valid_moves_1 = get_all_moves([0],new_board,WHITE,game)
                for num_board_1 in range(len(valid_moves_1)):
                    new_board_1 = valid_moves_1[num_board_1]
                    if((new_board, new_board_1) not in qvalue_grid.keys()):
                        qvalue_grid[(new_board, new_board_1)] = 0
                    if(qvalue_grid[(new_board, new_board_1)] > max_for_next_step):
                        max_for_next_step = qvalue_grid[(new_board, new_board_1)]

                if(qvalue_grid[(game.board, new_board)] > max_q_value):
                    max_q_value = qvalue_grid[(game.board, new_board)]
                    max_action = [new_board]
                elif(qvalue_grid[(game.board, new_board)] == max_q_value):
                    max_action.append(new_board)
                
            if(len(valid_moves) == 0):
                if game.board.winner() != None:
                    is_run = False
                continue

            flip_coin = random.random()
            if(flip_coin > epsilon): 

                final_action = random.choice(max_action)
                value_state = (game.board,final_action)
                game.ai_move(final_action)
            else:
                final_action = random.choice(valid_moves)
                value_state = (game.board,final_action)
                game.ai_move(final_action)
            
            reward_white = (final_action.evaluate_dist(WHITE) - game.board.evaluate_dist(WHITE))

            q_1 = -99999
            valid_moves_2 = get_all_moves([0],final_action,WHITE,game)
            for num_board_2 in range(len(valid_moves_2)):
                new_board_2 = valid_moves_2[num_board_2]
                if((final_action, new_board_2) not in qvalue_grid.keys()):
                    qvalue_grid[(final_action, new_board_2)] = 0
                if(qvalue_grid[(final_action, new_board_2)] > q_1):
                    q_1 = qvalue_grid[(final_action, new_board_2)]

            valid_moves = get_all_moves([0],game.board,ai_turn,game)
            if(len(valid_moves) == 0):
                if game.board.winner() != None:
                    is_run = False
                continue
            new_board = random.choice(valid_moves)
            game.ai_move(new_board)

            #evaluate function
            reward_black = (new_board.evaluate_dist(BLACK) - game.board.evaluate_dist(BLACK))

            qvalue_grid[value_state] = (1 - alpha) * qvalue_grid[value_state] + alpha * ((reward_white) + discount_factor * q_1)
            # qvalue_grid[value_state] = (1 - alpha) * qvalue_grid[value_state] + alpha * ((reward_white - reward_black) + discount_factor * q_1)
        
            if game.board.winner() != None:
                is_run = False
            game.update()
        if(i > 800):
            epsilon *= 0.99
    print("train_finish")

def get_action_from_q(game):
    valid_moves = get_all_moves([0],game.board,WHITE,game)
    # raise RuntimeError(valid_moves)
    if(len(valid_moves) == 0):
        return None

    max_action = []
    max_q_value = -9999
    for num_board in range(len(valid_moves)):
        try:
            new_board = valid_moves[num_board]
            if(qvalue_grid[(game.board, new_board)] == max_q_value):
                max_action.append(new_board)
            elif(qvalue_grid[(game.board, new_board)] > max_q_value):
                max_action = [new_board]
        except:
            new_board = valid_moves[num_board]
            if(max_q_value < 0):
                max_action = [new_board]
            elif(max_q_value == 0):
                max_action.append(new_board)
            else:
                continue

    final_action = random.choice(max_action)
    return final_action


def test():
    epochs_1 = 100
    # epsilon_1 = 0.3
    i_1 = 0
    count_win = 0
    count_win_black = 0

    while(i_1 < epochs_1):
        i_1 += 1
        # if(i % 50 == 0):
        #     print(i)
        # print(i)
        
        game = Game(win,WHITE)
        is_run_1 = True
        count_step = 0
        count_1 = 0
        while is_run_1:
            count_1 +=1

            if(count_1 > 150):
                break
            if game.turn == ai_turn:
                # score,new_board = minimax([0],game.board, ai_turn, depth_1, game)  
                valid_moves = get_all_moves([0],game.board,ai_turn,game)
                if(len(valid_moves) == 0):
                    if game.board.winner() != None:
                        # game.draw_winner()
                        is_run = False
                    continue
                new_board = random.choice(valid_moves)
                # random agent
                # raise RuntimeError(new_board)
                # print(4)        
                if new_board:
                    game.ai_move(new_board)
                else:
                    if game.board.winner() != None:
                        # game.draw_winner()
                        is_run = False
                    continue

                # pygame.time.delay(100)
            else:
                # if(count_step % 4 = )
                count_step += 1
                if(count_step > 150):
                    break
                # if(game.board == four_step_before):
                #     is_run = False
                #     continue
                # if(count_step % 10 == 0):
                #     four_step_before = game.board
                valid_moves = get_all_moves([0],game.board,WHITE,game)
                # raise RuntimeError(valid_moves)
                if(len(valid_moves) == 0):
                    if game.board.winner() != None:
                        # game.draw_winner()
                        is_run = False
                    continue

                # max Q-value
                max_action = []
                max_q_value = -9999
                for num_board in range(len(valid_moves)):
                    try:
                        new_board = valid_moves[num_board]
                    
                
                        if(qvalue_grid[(game.board, new_board)] == max_q_value):
                            # max_q_value = qvalue_grid[(game.board, new_board)]
                            max_action.append(new_board)
                        elif(qvalue_grid[(game.board, new_board)] > max_q_value):
                            max_action = [new_board]
                    except:
                        # continue
                        new_board = valid_moves[num_board]
                        if(max_q_value < 0):
                            max_action = [new_board]
                        elif(max_q_value == 0):
                            max_action.append(new_board)
                        else:
                            continue


                if(len(valid_moves) == 0):
                    if game.board.winner() != None:
                        # game.draw_winner()
                        is_run = False
                    continue

                final_action = random.choice(max_action)
                game.ai_move(final_action)


            if game.board.winner() != None:
                # game.draw_winner()
                if(game.board.winner() == WHITE):
                    count_win += 1
                elif(game.board.winner() == BLACK):
                    count_win_black += 1
                    # print(count_win)
                is_run = False
                
            
            game.update()
        # epsilon *= 0.99
    print(count_win)
    print(count_win_black)
    # print(len(list(qvalue_grid.values())))
        
# train_qlearning()
# raise RuntimeError("train_finish")

def run_game():
    global depth,ai_turn,algorithm
    is_run = True

    if ai_turn == BLACK:
       game = Game(win,WHITE)
    else:
        game = Game(win,BLACK)
    
    while is_run:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                is_run = False 
            
            #white  ai_ture 000   gameturn 255 255 255
            #black  ai_ture 255 255 255   gameturn 255 255 255
            # print(ai_turn)
            # raise RuntimeError(game.turn)
            if game.turn == ai_turn:
                # print(1)
                # if algorithm == 1:
                #     score,new_board = negamax([0],game.board, ai_turn, depth, game)
                # else:
                #     score,new_board = alpha_beta_pruning([0],game.board, ai_turn,NEGA_INFI , POSI_INFI, depth, game)    
                score,new_board = minimax([0],game.board, ai_turn, depth, game)  
                # print(4)        
                if new_board:
                    game.ai_move(new_board)
                # pygame.time.delay(100)
            else:
                # print(2)
                game.get_moves()
                # if algorithm == 1:
                #     score,new_board = negamax([0],game.board, game.turn, depth, game)
                # else:
                #     score,new_board = alpha_beta_pruning([0],game.board, game.turn, NEGA_INFI , POSI_INFI, depth, game)         
                # print(3)     
                # score,new_board = minimax([0],game.board, game.turn, depth, game)
                # score,new_board = alpha_beta_pruning([0],game.board, game.turn, NEGA_INFI , POSI_INFI, depth, game)      
                # if new_board:
                #     game.ai_move(new_board)
                # pygame.time.delay(100)
            
            if game.board.winner() != None:
                # game.draw_winner()
                is_run = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x , y = pos
                game.after_click(y//SIZE,x//SIZE)
        
        game.update()

if __name__ == '__main__':
    pygame.init()
    win = pygame.display.set_mode((WIDTH,HEIGHT))
    menu = pygame_menu.Menu('Welcome',
                            400,
                            400,
                            theme=pygame_menu.themes.THEME_BLUE)

    menu.add.selector('Algorithm :', [('minimax',1), ('a-b pruning',2)],
                    onchange=select_algorithm)
    menu.add.selector('My Turn :', [('WHITE',1), ('BLACK',2)],
                    onchange=select_turn)
    menu.add.text_input('Depth :', default= 4, onchange=set_depth)
    menu.add.button('Play', run_game)
    menu.add.button('Quit', pygame_menu.events.EXIT)

    train_qlearning()
    print(" ============== training finish ============== ")

    print(" ============== start evaluate ============== ")
    test()
    print(" ============== evaluate finish ============== ")