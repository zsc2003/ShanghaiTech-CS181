import pygame
import pygame_menu
from utils.config import WIDTH ,HEIGHT, SIZE, BLACK, WHITE, POSI_INFI, NEGA_INFI
from utils.draught_game import Game

from algorithms.minimax import minimax, alpha_beta_pruning
from algorithms.random_agent import random_algorithm
# from algorithms.MCTS import mcts_agent
from algorithms.mcts import mcts_agent
# from algorithms.reinforcement_learning import reinforcement_learning

menu = None

depth = 3         # search depth
ai_turn = BLACK   # the color for the ai piece
algorithm = 1     # search algorithm
color = 1
p1 = 1
p1_algorithm = 4

# select algorithm for AI
def select_algorithm(value, index):
    global algorithm
    algorithm = index
    # print(index)


# select algorithm for p1-AI
def select_p1_algorithm(value, index):
    global p1_algorithm, p1
    if p1 == 1: # human
        p1_algorithm = 4
    elif p1 == 2: # AI
        if index == 4:
            index = 1
        p1_algorithm = index
    else:
        raise ValueError("Unconstructed player")


# select the color for the player
def select_turn(value, index):
    global ai_turn, color
    color = index
    if index == 2:
        ai_turn = WHITE
    else:
        ai_turn = BLACK
    # print(index)


# select the depth for searching
def select_depth(value, index):
    global depth
    depth = index
    # print(index)


def select_p1(value, index):
    global p1, p1_algorithm
    p1 = index

    if index == 1: # p1 is human
        p1_algorithm = 4
    elif index == 2: # p1 is AI
        p1_algorithm = 1        
    else:
        raise ValueError("Unconstructed player")

# run draught
def run_game():
    global depth, ai_turn, algorithm
    is_run = True

    # initialize
    if ai_turn == BLACK:
       game = Game(window, WHITE)
    else:
        game = Game(window, BLACK)
        
    mcts_class_white = mcts_agent(game.board, WHITE)
    mcts_class_black = mcts_agent(game.board, BLACK)
    
    while is_run:
        for event in pygame.event.get():
            # Esc to quit the game
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                is_run = False 
            
            # judge the turn
            if game.turn == ai_turn:
                if algorithm == 1: # random
                    score, new_board = random_algorithm([0], game.board, ai_turn, depth, game)
                elif algorithm == 2: # search
                    score, new_board = alpha_beta_pruning([0], game.board, ai_turn, NEGA_INFI, POSI_INFI, depth, game)              
                elif algorithm == 3: # MCTS
                    # print("Before board: ", game.board)
                    # new_board = mcts_class.step(game.board, 20)
                    if game.turn == WHITE:
                        new_board = mcts_class_white.step(game.board, 20)
                    else:
                        new_board = mcts_class_black.step(game.board, 20)
                    # print("After board: ", new_board)
                    
                else:
                    raise ValueError("Unconstructed algorithm")
                
                if new_board:
                    game.ai_move(new_board)
            else:
                
                # TODO add AI vs AI
                if p1 == 1: # human
                    game.get_moves()
                elif p1 == 2: # AI
                    if p1_algorithm == 2: # alpha beta pruning
                        score, new_board = alpha_beta_pruning([0], game.board, WHITE, NEGA_INFI, POSI_INFI, depth, game)
                    elif p1_algorithm == 3:
                        if game.turn == WHITE:
                            new_board = mcts_class_white.step(game.board, 20)
                        else:
                            new_board = mcts_class_black.step(game.board, 20)
                    else:
                        score, new_board = random_algorithm([0], game.board, WHITE, depth, game)
                    if new_board:
                        game.ai_move(new_board)

            # judge if anyone wins
            if game.board.winner() != None:
                game.draw_winner()
                is_run = False
            
            # select and move the pieces
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x , y = pos
                game.after_click(y // SIZE, x // SIZE)
        
        game.update()


def add_selectors():
    global menu, algorithm, color, depth, p1
    menu.add.selector('Algorithm : ', algorithm_list,
                      onchange=select_algorithm, font_name=font, default=algorithm - 1)
    menu.add.selector('My Turn : ', [('WHITE',1), ('BLACK',2)],
                      onchange=select_turn, font_name=font, default=color - 1)
    menu.add.selector('Depth : ', [(f'{i}', i) for i in range(1, 11)],
                      onchange=select_depth, font_name=font, default=depth - 1)

    menu.add.selector("Player1 : ", [('human', 1), ['AI', 2]],
                      onchange=select_p1, font_name=font, default=p1 - 1)
    menu.add.selector("Player1 Algorithm: ", algorithm_list + [('--', 4)],
                      onchange=select_p1_algorithm, font_name=font, default=p1_algorithm - 1)

def add_buttons():
    menu.add.button('Play', run_game, font_name=font)
    menu.add.button('Quit', pygame_menu.events.EXIT, font_name=font)

if __name__ == '__main__':
    RUN_APPROXIMATE_Q_LEARNING = True
    if RUN_APPROXIMATE_Q_LEARNING:
        from approximate_q_learning import approximate_q_learning
        approximate_q_learning()
        exit()

    pygame.init()

    window = pygame.display.set_mode((WIDTH,HEIGHT))

    # initialize menu
    font = pygame_menu.font.FONT_NEVIS
    menu = pygame_menu.Menu('Welcome',
                            500,
                            400,
                            theme=pygame_menu.themes.THEME_SOLARIZED,
                            )

    algorithm_list = [('random', 1),
                      ('search', 2),
                      ('MCTS', 3),
                     ]

    add_selectors()
    add_buttons()

    menu.mainloop(window)

    run_game()