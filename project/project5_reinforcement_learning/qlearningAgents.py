# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        self.Q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        if (state, action) not in self.Q_values:
            return 0.0
        return self.Q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        legal_actions = self.getLegalActions(state)

        # no legal actions, which is the case at the terminal state
        if len(legal_actions) == 0:
            return 0.0

        q_values = [self.getQValue(state, action) for action in legal_actions]
        return max(q_values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        legal_actions = self.getLegalActions(state)

        # no legal actions, which is the case at the terminal state
        if len(legal_actions) == 0:
            return None

        #  For computeActionFromQValues, you should break ties randomly for better behavior. The random.choice() function will help
        possible_actions = []

        best_q_value = self.computeValueFromQValues(state)
        for action in legal_actions:
            if abs(self.getQValue(state, action) - best_q_value) < 1e-15:
                possible_actions.append(action)
                return action
        
        if len(possible_actions) == 0:
          return None

        return random.choice(possible_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        strategy = util.flipCoin(self.epsilon)
        # random action
        if strategy == True:
            action = random.choice(legalActions)        
        else: # best policy action
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # Q(s,a) = (1-alpha) * Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a'))
        # Q(s,a) = (1-alpha) * Q(s,a) + alpha * (reward + gamma * V(s'))
        # V(s') = max_a' Q(s',a')

        V_s = self.computeValueFromQValues(nextState)
        self.Q_values[(state, action)] = (1 - self.alpha) * self.Q_values[(state, action)] \
                                       + self.alpha * (reward + self.discount * V_s)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        w = self.getWeights()
        featureVector = self.featExtractor.getFeatures(state, action)
        q_value = 0.0

        for feature in featureVector:
            q_value += w[feature] * featureVector[feature]
        return q_value
    
    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # difference = reward + gamma * max_a' Q(s',a') - Q(s,a)
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)

        # wi = wi + alpha * difference * fi(s,a)
        featureVector = self.featExtractor.getFeatures(state, action)
        for feature in featureVector:
            self.weights[feature] += self.alpha * difference * featureVector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


# MazeDistance from pa1
def mazeDistance(point1, goals, gameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    walls = gameState.getWalls()

    # do bfs from point1 until first time find the point2
    from game import Directions
    from util import Queue
    q = Queue()
    visited = set()
    q.push((point1, 0))
    visited.add(point1)
    while not q.isEmpty():
        point, distance = q.pop()
        if point in goals:
            return distance / (walls.width * walls.height)
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = point
            dx, dy = Actions.directionToVector(action)
            new_point = (int(x + dx), int(y + dy))
            if new_point not in visited and not walls[new_point[0]][new_point[1]]:
                visited.add(new_point)
                q.push((new_point, distance + 1))
    return None

class BetterExtractor(FeatureExtractor):
    "Your extractor entry goes here.  Add features for capsuleClassic."
    
    import pacman
    def getFeatures(self, state:pacman.GameState, action):
        features = SimpleExtractor().getFeatures(state, action)
        # Add more features here
        "*** YOUR CODE HERE ***"

        foods = state.getFood()
        food_list = foods.asList()
        capsules = state.getCapsules()
        
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        ghostStates = state.getGhostStates()
        scare_times = [ghostState.scaredTimer for ghostState in ghostStates]

        pacman_pos = state.getPacmanPosition()
        newPos = Actions.getSuccessor(pacman_pos, action)
        newPos = (int(newPos[0]), int(newPos[1]))

        min_food_distance = mazeDistance(newPos, food_list, state)
        min_ghost_distance = mazeDistance(newPos, ghosts, state)
        min_capsule_distance = mazeDistance(newPos, capsules, state)

        scared_ghost = [ghost_state for ghost_state in ghostStates if ghost_state.scaredTimer > 0]
        unscared_ghost = [ghost_state for ghost_state in ghostStates if ghost_state.scaredTimer == 0]

        features.divideAll(1e8)
        w = {'scared_ghost' : -2.01, 'eat_ghost' : 4.85, 'capsule' : 4, 'near_ghost' : -7.8, 'close_ghost' : 2.56, }
        bias = {'scared_ghost' : -3.4, 'eat_ghost' : -0.47, 'capsule' : 6, 'near_ghost' : -6.57, 'close_ghost' : -6.8, }

        # capsule
        if scared_ghost == [] and min_capsule_distance != None:
            features['capsule'] = w['capsule'] * min_capsule_distance + bias['capsule']

        # ghosts
        features['scared_ghost'] = w['scared_ghost'] * len(scared_ghost) + bias['scared_ghost']
        
        scared_ghost_pos = []
        for ghost_state in scared_ghost:
            ghost_new_pos = Actions.getSuccessor(ghost_state.getPosition(), ghost_state.getDirection())
            ghost_new_pos = (int(ghost_new_pos[0]), int(ghost_new_pos[1]))
            scared_ghost_pos.append(ghost_new_pos)
        min_dis_scared_ghost = mazeDistance(newPos, scared_ghost_pos, state)

        if min_dis_scared_ghost != None:
            features['eat_ghost'] = w['eat_ghost'] * min_dis_scared_ghost + bias['eat_ghost']


        neibors = Actions.getLegalNeighbors(newPos, walls)
        # ghost in pacman's neibor
        num_near_ghost = sum(ghost_state.getPosition() in neibors for ghost_state in unscared_ghost)

        # ghost in pacman's neibor's neibor
        num_close_ghost = 0
        new_neibors = [(newPos[0] + 1, newPos[1]), (newPos[0], newPos[1] + 1), (newPos[0] - 1, newPos[1]), (newPos[0], newPos[1] - 1)]
        for neibor in new_neibors:
            num_close_ghost += sum(neibor in Actions.getLegalNeighbors(ghostState.getPosition(), walls) \
                                   for ghostState in unscared_ghost)

        features['near_ghost'] = w['near_ghost'] * num_near_ghost + bias['near_ghost']
        features['close_ghost'] = w['close_ghost'] * num_close_ghost + bias['close_ghost']

        features['bias'] = -1

        # print("====================================")
        # print(f"====  sum = {features.totalCount()}  ====")
        # print("====================================")

        max_feature = -19260817
        for feature in features:
            max_feature = max(max_feature, abs(features[feature]))

        # print("====================================")
        # print(f"====  maxn = {max_feature}  ====")
        # print("====================================")
        
        # learning will work much better if your features have a maximum absolute value of 1.
        # Shrinking the whole feature vector so that the sum of weights is less than 1 can help even further
        features.divideAll(min(max_feature + 4, 10))
        return features