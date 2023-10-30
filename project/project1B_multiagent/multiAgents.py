# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(newPos)

        # distance between newPos and newGhostStates
        # I am more likely to use the maze_distant here, but it needs to much work .......
        dists_to_ghost = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        
        # print(dists_to_ghost)
        # print(newGhostStates[0].getPosition())

        old_food = currentGameState.getFood()
        # I am more likely to use the maze_distant here, but it needs to much work .......
        dists_to_food = [manhattanDistance(newPos, food) for food in old_food.asList()]
        dists_to_capsules = [manhattanDistance(newPos, capsule) for capsule in currentGameState.getCapsules()]
        
        # print(newFood.asList())
        # print(dists_to_food)
        # print(currentGameState.getCapsules())

        # As features, try the reciprocal of important values (such as distance to food) rather than just the values themselves.
        # The evaluation function you're writing is evaluating state-action pairs; in later parts of the project, you'll be evaluating states.
        
        # print(newScaredTimes)
        
        if newPos in currentGameState.getCapsules(): # try to get the capsules
            return 100
        
        for i, scared_time in enumerate(newScaredTimes):
            if scared_time > 0 and newPos == newGhostStates[i].getPosition(): # eat the ghost
                return 100
            if scared_time <= 0 and newPos == newGhostStates[i].getPosition(): # be eaten by the ghost
                return -500

        min_dist_to_ghost = min(dists_to_ghost) if len(dists_to_ghost) > 0 else 0
        min_dist_to_food = min(dists_to_food) if len(dists_to_food) > 0 else 0
        min_dist_to_capsule = min(dists_to_capsules) if len(dists_to_capsules) > 0 else 1926

        # print(newScaredTimes)
        if newPos in old_food.asList():
            # print('1111')
            return 10
        
        if min_dist_to_ghost >= 5 and action == Directions.STOP:
            return -1

        weight = 1
        if min_dist_to_ghost < 2:
            weight = 100
        
        # reciprocal
        score = 10 / min_dist_to_food - weight / min_dist_to_ghost
        if min_dist_to_capsule <= 2:
            score += 100 / min_dist_to_capsule
        # print(score)
        # if min_dist_to_ghost < 2:
            # score = -500 * min_dist_to_ghost

        # print(score)
        return score
        # print(childGameState.getScore())
        # return childGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    inf = 1926081719491001

    # pacman max_node
    def max_node(self, gameState : GameState, depth, ghost_num):
        max_val = -self.inf

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # action for pacman, agent_id = 0
        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            max_val = max(max_val, self.min_node(next_state, depth, 1, ghost_num)) # the ghost's id start from 1
        
        return max_val
    
    # adversarial(ghost) min_node
    def min_node(self, gameState : GameState, depth, agent_id, ghost_num):
        min_val = self.inf

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # action for ghost, agent_num in [1, ghost_num]
        for action in gameState.getLegalActions(agent_id):
            next_state = gameState.getNextState(agent_id, action)

            if agent_id == ghost_num: # last ghost, go to the next depth for pacman
                min_val = min(min_val, self.max_node(next_state, depth + 1, ghost_num))
            else: # consider the next ghost
                min_val = min(min_val, self.min_node(next_state, depth, agent_id + 1, ghost_num))
        
        return min_val

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        ghost_num = gameState.getNumAgents() - 1 # 0 for pacman, [1, ghost_num] for ghosts
        # print("ghost_num = ", ghost_num)

        # the nodes connected to the root(pacman) are min_nodes
        # pacman is the root, all the ghosts are the first depth

        max_val = -self.inf
        act = None

        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            val = self.min_node(next_state, 0, 1, ghost_num)
            if max_val < val:
                act = action
                max_val = val
            
        return act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    inf = 1926081719491001

    # pacman max_node
    def max_node(self, gameState : GameState, depth, ghost_num, alpha, beta):
        max_val = -self.inf

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # action for pacman, agent_id = 0
        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            max_val = max(max_val, self.min_node(next_state, depth, 1, ghost_num, alpha, beta)) # the ghost's id start from 1

            # alpha-beta pruning
            if max_val > beta:
                return max_val
            alpha = max(alpha, max_val)

        return max_val
    
    # adversarial(ghost) min_node
    def min_node(self, gameState : GameState, depth, agent_id, ghost_num, alpha, beta):
        min_val = self.inf

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # action for ghost, agent_num in [1, ghost_num]
        for action in gameState.getLegalActions(agent_id):
            next_state = gameState.getNextState(agent_id, action)

            if agent_id == ghost_num: # last ghost, go to the next depth for pacman
                min_val = min(min_val, self.max_node(next_state, depth + 1, ghost_num, alpha, beta))
            else: # consider the next ghost
                min_val = min(min_val, self.min_node(next_state, depth, agent_id + 1, ghost_num, alpha, beta))
            
            # alpha-beta purning
            if min_val < alpha:
                return min_val
            beta = min(beta, min_val)
        return min_val

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        ghost_num = gameState.getNumAgents() - 1 # 0 for pacman, [1, ghost_num] for ghosts
        # print("ghost_num = ", ghost_num)

        # the nodes connected to the root(pacman) are min_nodes
        # pacman is the root, all the ghosts are the first depth

        act = None

        alpha = -self.inf
        beta = self.inf

        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            val = self.min_node(next_state, 0, 1, ghost_num, alpha, beta)
            
            # not prune on equality !!
            if val > beta:
                return act
            if val > alpha:
                act = action
                alpha = val
            
        return act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    inf = 1926081719491001

    # pacman max_node
    def max_node(self, gameState : GameState, depth, ghost_num):
        max_val = -self.inf

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # action for pacman, agent_id = 0
        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            max_val = max(max_val, self.expect_node(next_state, depth, 1, ghost_num)) # the ghost's id start from 1
        
        return max_val
    
    # adversarial(ghost) min_node
    def expect_node(self, gameState : GameState, depth, agent_id, ghost_num):
        expect_val = 0

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # action for ghost, agent_num in [1, ghost_num]
        actions = gameState.getLegalActions(agent_id)
        for action in actions:
            next_state = gameState.getNextState(agent_id, action)

            if agent_id == ghost_num: # last ghost, go to the next depth for pacman
                expect_val += self.max_node(next_state, depth + 1, ghost_num)
            else: # consider the next ghost
                expect_val += self.expect_node(next_state, depth, agent_id + 1, ghost_num)
        
        return expect_val / len(actions)
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        ghost_num = gameState.getNumAgents() - 1 # 0 for pacman, [1, ghost_num] for ghosts

        # the nodes connected to the root(pacman) are min_nodes
        # pacman is the root, all the ghosts are the first depth

        max_val = -self.inf
        act = None

        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            val = self.expect_node(next_state, 0, 1, ghost_num)
            if max_val < val:
                act = action
                max_val = val
            
        return act

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    start with the naive score we calculated in the reflex agent

    regard the root as an expected-node
    consider the capsule, food, ghost
    the food has 10 points, the scared ghost has 100 points
    when meeting the un-scared ghost, regard is as -500 points
    also regard the capsule as high valued food, try different value of it

    after testing, take the capsules' value to be 25

    if the ghost is too far from the pacman(dis > 3), just ignore it

    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()


    food_score = 10
    capsules_score = 25
    ghost_score = -500
    scared_ghost_score = 100 # try to catch the scared ghost

    # val = 0
    val = currentGameState.getScore()

    pacman_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood().asList()
    capsule_pos = currentGameState.getCapsules()

    dis_to_food = [manhattanDistance(pacman_pos, food) for food in food_pos]
    if dis_to_food != []:
        val += food_score / min(dis_to_food)
    
    dis_to_capsule = [manhattanDistance(pacman_pos, capsule) for capsule in capsule_pos]
    if dis_to_capsule != []:
        # if pacman_pos in capsule_pos:
            # val += capsules_score
        # else:
        val += capsules_score / min(dis_to_capsule)
    
    # dis_to_ghost = [manhattanDistance(pacman_pos, ghost) for ghost in ghost_pos]
    for ghost in currentGameState.getGhostStates():
        dis_to_ghost = manhattanDistance(pacman_pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            val += scared_ghost_score / dis_to_ghost
        else:
            if dis_to_ghost <= 2: # ignore the far ghost
                val += ghost_score

    return val


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState: GameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()


        