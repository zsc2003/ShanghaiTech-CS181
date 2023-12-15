# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for t in range(self.iterations):
            V_new = util.Counter()
            # V_k+1(s)=max_a(sum_s' T(s,a,s')[R(s,a,s')+gamma*V_k(s')])
            #         =max_a(Q_k(s,a))
            for state in self.mdp.getStates():
                # print(self.mdp.getPossibleActions(state))
                Q_values = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
                # for action in self.mdp.getPossibleActions(state):
                    # V_new[state] = max(V_new[state], self.computeQValueFromValues(state, action))
                    # print(f"Q_value of action {action} = ", self.computeQValueFromValues(state, action))
                if len(Q_values) == 0:
                    V_new[state] = 0.0
                else:
                    V_new[state] = max(Q_values)

            self.values = V_new

        return

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # returns the Q-value of the (state, action) pair given by the value function given by self.values

        # Q(s,a)=sum_s' T(s,a,s')[R(s,a,s')+gamma*V(s')]
        Q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            Q_value += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # computes the best action according to the value function given by self.values
        # print(state)
        # print(self.values)
        # print("-------------------")
        action_values = [(action, self.computeQValueFromValues(state, action)) for action in self.mdp.getPossibleActions(state)]
        # print(action_values)
        if len(action_values) == 0:
            return None
        action = max(action_values, key=lambda x: x[1])[0]
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Asynchronous
        # (异步)

        # update only one state in each iteration
        state_num = len(self.mdp.getStates())
        for t in range(self.iterations):
            state = self.mdp.getStates()[t % state_num]
            
            # If the state picked for updating is terminal, nothing happens in that iteration
            if not self.mdp.isTerminal(state):
                Q_values = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
                if len(Q_values) == 0:
                    self.values[state] = 0.0
                else:
                    self.values[state] = max(Q_values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Compute predecessors of all states.
        # when you compute predecessors of a state, make sure to store them in a set, not a list, to avoid duplicates.
        predecessor = {state : set() for state in self.mdp.getStates()}
        
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessor[next_state].add(state)
        
        # Initialize an empty priority queue
        priority_queue = util.PriorityQueue()

        # For each non-terminal state s, do:
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Find the absolute value of the difference between the current value of s in self.values 
                # and the highest Q-value across all possible actions from s (this represents what the value should be);
                # call this number diff.
                Q_values = [abs(self.values[state] - self.computeQValueFromValues(state, action)) 
                            for action in self.mdp.getPossibleActions(state)]
                if len(Q_values) == 0:
                    diff = abs(self.values[state] - 0.0)
                else:
                    diff = abs(self.values[state] - max(Q_values))

                # Push s into the priority queue with priority -diff (note that this is negative).
                # We use a negative because the priority queue is a min heap,
                # but we want to prioritize updating states that have a higher error.
                priority_queue.push(state, -diff)
    
        # For iteration in 0, 1, 2, ..., self.iterations - 1, do: 
        for t in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if priority_queue.isEmpty():
                break
                
            # Pop a state s off the priority queue.
            s = priority_queue.pop()

            # Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                Q_values = [self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]
                if len(Q_values) == 0:
                    self.values[s] = 0.0
                else:
                    self.values[s] = max(Q_values)
            
            # For each predecessor p of s, do: 
            for p in predecessor[s]:

                # Find the absolute value of the difference between the current value of p in self.values and 
                # the highest Q-value across all possible actions from p  (this represents what the value should be); 
                # call this number diff
                diff = abs(self.values[p] - self.computeQValueFromValues(p, self.computeActionFromValues(p)))

                # If diff > theta, push p into the priority queue with priority -diff (note that this is negative),
                if diff > self.theta:
                    priority_queue.update(p, -diff)