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

alpha = 0.5
discount_factor = 0.9

def q_func(board : Board, board2 : Board):
    # wHITE
    f1 = 0
    if(board.white_left == 0):
        f1 = -10
    if(board.black_left == 0):
        f1 = 10
    
    f2 = board.white_left - board.black_left + (board.white_kings * 2 - board.black_kings * 2)

    f3 = board.distance()

    f4 = 0
    if(board2.white_left == 0):
        f4 = -10
    if(board2.black_left == 0):
        f4 = 10
    
    f5 = board2.white_left - board2.black_left + (board2.white_kings * 2 - board2.black_kings * 2)

    f6 = board2.distance()

    if board.use_feature_network:
        f7 = board.evaluate_neural(WHITE)
    else:
        f7 = 0

    if board2.use_feature_network:
        f8 = board2.evaluate_neural(BLACK)
    else:
        f8 = 0

    return f1, f2, f3, f4, f5, f6, f7, f8

def train():
    # training approximate Q-learning

    global w1, w2, w3, w4, w5, w6, w7, w8, depth

    # pretrain
    # w1, w2, w3, w4, w5, w6 = 10, 10, 10, 10, 10, 10
    # epsilon = 0.3
    
    # finetune
    w1, w2, w3, w4, w5, w6, w7, w8 = 9.86152761377497, 0.023915788615273777, 0.48000919549243837, 9.9674619481094, -0.11002902967481094, 0.158369039828599, 3.46109195499481, 4.96745273781
    epsilon = 0.05

    print("================= start training =================")
    epochs = 3000
    
    i = 0
    count_win = 0
    four_step_before = None
    while(i < epochs):

        i += 1
        if(i % 50 == 0):
            # print(list(qvalue_grid.values()))
            print(i)
    
            # write w1, w2, ... , w6 to log.txt
            with open("log.txt", "a") as f:
                f.write(f"iter : {i}\n")
                f.write(f"w1 = {w1}, w2 = {w2}, w3 = {w3}, w4 = {w4}, w5 = {w5}, w6 = {w6}, w7 = {w7}, w8 = {w8}\n")
        
        game = Game(window, WHITE)
        is_run = True
        count_step = 0
        while is_run:
            if game.turn == ai_turn:
                count_step += 1
                if count_step > 300:
                    break
                score, new_board = minimax([0], game.board, ai_turn, depth, game)  

                if new_board:
                    game.ai_move(new_board)
                if new_board == None:
                    if game.board.winner() != None:
                        game.draw_winner()
                        is_run = False
                    continue
            else:
                count_step += 1
                # avoid draw 
                if(count_step > 300):
                    break

                valid_moves = get_all_moves([0], game.board, WHITE, game)
                # raise RuntimeError(valid_moves)
                max_action = None
                max_q_value = -9999
                for num_board in range(len(valid_moves)):
                    new_board = valid_moves[num_board]

                    # if((game.board, new_board) not in qvalue_grid.keys()):
                    #     qvalue_grid[(game.board, new_board)] = 0
                    f11, f12, f13, f14, f15, f16, f17, f18 = q_func(game.board, new_board)
                    q1 = w1 * f11 + w2 * f12 + w3 * f13 + w4 * f14 + w5 * f15 + w6 * f16 + w7 * f17 + w8 * f18
                    #start difference 
                    max_for_next_step = -99999
                    reward = 0
                    valid_moves_1 = get_all_moves([0], new_board, WHITE, game)
                    for num_board_1 in range(len(valid_moves_1)):
                        new_board_1 = valid_moves_1[num_board_1]
                        # if((new_board, new_board_1) not in qvalue_grid.keys()):
                        #     qvalue_grid[(new_board, new_board_1)] = 0
                        f21, f22, f23, f24, f25, f26, f27, f28 = q_func(new_board, new_board_1)
                        q2 = w1 * f21 + w2 * f22 + w3 * f23 + w4 * f24 + w5 * f25 + w6 * f26 + w7 * f27 + w8 * f28
                        if(q2 > max_for_next_step):
                            max_for_next_step = q2
                            reward = new_board.black_left - new_board_1.black_left
                        # if(qvalue_grid[(new_board, new_board_1)] > max_for_next_step):
                        #     max_for_next_step = qvalue_grid[(new_board, new_board_1)]
                    # update Q-value
                    # print(f'q1 = {q1}, q2 = {q2}, reward = {reward}, w1 = {w1}, w2 = {w2}, w3 = {w3}, w4 = {w4}, w5 = {w5}, w6 = {w6}')
                    # lr = 0.01
                    lr = 0.001
                    w1 = w1 + lr * ((reward + discount_factor * q2) - q1) * f11
                    w2 = w2 + lr * ((reward + discount_factor * q2) - q1) * f12
                    w3 = w3 + lr * ((reward + discount_factor * q2) - q1) * f13
                    w4 = w4 + lr * ((reward + discount_factor * q2) - q1) * f14
                    w5 = w5 + lr * ((reward + discount_factor * q2) - q1) * f15
                    w6 = w6 + lr * ((reward + discount_factor * q2) - q1) * f16
                    w7 = w7 + lr * ((reward + discount_factor * q2) - q1) * f17
                    w8 = w8 + lr * ((reward + discount_factor * q2) - q1) * f18
                    
                    # qvalue_grid[(game.board, new_board)] = (1 - alpha) * qvalue_grid[(game.board, new_board)] + alpha * (-(new_board.evaluate_dist(WHITE) - game.board.evaluate_dist(WHITE)) + discount_factor * max_for_next_step)

                    if(q1 > max_q_value):
                        max_q_value = q1
                        max_action = new_board

                if(len(valid_moves) == 0):
                    if game.board.winner() != None:
                        game.draw_winner()
                        is_run = False
                    continue
                
                # epsilon-greedy
                flip_coin = random.random()
                if flip_coin > epsilon and max_action != None:
                    game.ai_move(max_action)
                else:
                    game.ai_move(random.choice(valid_moves))
                # epsilon *= 0.99

                # get_all_moves

                # game.change_turn()

            # print(game.board)
                        
            if game.board.winner() != None:
                # game.draw_winner()
                if(game.board.winner() == WHITE):
                    # count_win += 1
                    print(count_win)
                is_run = False
                
            game.update()

        if(i > 1000):
            epsilon *= 0.99

    # with open("D:/CS181/project/draughtAI-main/draughtAI-main/ckpts.pkl",'wb')as f:
    #     pickle.dump(qvalue_grid,f)
    #  =================== train ===================
    # print("load_begin")
    # with open("D:/CS181/project/draughtAI-main/draughtAI-main/ckpts.pkl", 'rb')as f1:
    #     qvalue_grid = pickle.load(f1)
    # print("load_finish")
    # qvalue_grid

def evaluate():
    # test winning rate
    w1, w2, w3, w4, w5, w6 = 9.86152761377497, 0.023915788615273777, 0.48000919549243837, 9.9674619481094, -0.11002902967481094, 0.1583690398285995

    epochs_1 = 100
    epsilon = 0.3
    i = 0
    count_win = 0
    four_step_before = None
    depth_1 = 2

    while(i < epochs_1):
        i += 1
        # if(i % 50 == 0):
        #     print(i)
        # print(i)
        
        game = Game(window, WHITE)
        is_run = True
        count_step = 0
        count_1 = 0
        while is_run:
            count_1 +=1

            if(count_1 > 150):
                break
            if game.turn == ai_turn:
                # score,new_board = minimax([0],game.board, ai_turn, depth_1, game)  
                valid_moves = get_all_moves([0], game.board, ai_turn, game)
                if(len(valid_moves) == 0):
                    if game.board.winner() != None:
                        game.draw_winner()
                        is_run = False
                    continue
                new_board = random.choice(valid_moves)

                if new_board:
                    game.ai_move(new_board)

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
                        game.draw_winner()
                        is_run = False
                    continue

                # take the max qvalue
                max_action = valid_moves[0]
                max_q_value = -9999
                for num_board in range(len(valid_moves)):

                    new_board = valid_moves[num_board]
                    f1,f2,f3,f4,f5,f6 = q_func(game.board, new_board)
                    q1 = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5 + w6 * f6
                    if(q1 > max_q_value):
                        # max_q_value = qvalue_grid[(game.board, new_board)]
                        max_action = new_board

                if(len(valid_moves) == 0):
                    if game.board.winner() != None:
                        game.draw_winner()
                        is_run = False
                    continue
                flip_coin = random.random()
                if(flip_coin >= 0):
                    try:
                        
                        game.ai_move(max_action)

                    except:
                        pygame.time.delay(100)
                else:
                    try:
                        game.ai_move(random.choice(valid_moves))
                    except:
                        # raise RuntimeError(valid_moves)
                        pygame.time.delay(100)
                # epsilon *= 0.99


                # get_all_moves


                # game.change_turn()

            if game.board.winner() != None:
                # game.draw_winner()
                if(game.board.winner() == WHITE):
                    count_win += 1
                    # print(count_win)
                is_run = False
            
            game.update()
        # epsilon *= 0.99
    print(count_win)
    # print(len(list(qvalue_grid.values())))
    print(f"w1 = {w1}, w2 = {w2}, w3 = {w3}, w4 = {w4}, w5 = {w5}, w6 = {w6}")


# train_qlearning()
# raise RuntimeError("train_finish")

def run_game():
    global depth,ai_turn,algorithm
    is_run = True

    window = pygame.display.set_mode((WIDTH,HEIGHT))


    if ai_turn == BLACK:
       game = Game(window, WHITE)
    else:
        game = Game(window, BLACK)
    
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
                score,new_board = minimax([0], game.board, ai_turn, depth, game)  
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
                game.draw_winner()
                is_run = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x, y = pos
                game.after_click(y // SIZE, x // SIZE)
        
        game.update()


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

if __name__ == '__main__':
    global window
    pygame.init()
    window = pygame.display.set_mode((WIDTH,HEIGHT))
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

    train()
    print(" ============== training finish ============== ")

    print(" ============== start evaluate ============== ")
    evaluate()
    print(" ============== finished evaluate ============== ")