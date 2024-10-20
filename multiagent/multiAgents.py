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


from util import manhattanDistance, matrixAsList
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        chosen_move = legalMoves[chosenIndex]
        print(chosen_move)
        return chosen_move 

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        # An ascii layout of the gameboard where pacman is represented as <>^v
        # depending on the direction that he is facing.
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Coordinate position of pacman in the successor gamestate
        newPos = successorGameState.getPacmanPosition()

        # Ascii table representing the locations in the successor state where
        # there is food on the board.
        newFood = successorGameState.getFood()

        # A list containing the state information for each ghost on the board.
        # Contains the coordinate position of the ghosts as well as the
        # direction that they are facing.
        newGhostStates = successorGameState.getGhostStates()

        # A list containing the number of moves that each ghost will remain
        # scared for.
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # 
        # TODO:
        # Would probably be good to prefer positions where pacman is not facing
        # opposite a ghosts direction of motion. This should be weighted by how
        # close pacman is to a ghost

        # TODO:
        # Prefer positions which are further away from ghosts
        ghost_dist_score = 0
        for i, state in enumerate(newGhostStates):

            pos = state.getPosition()
            dist = manhattanDistance(pos, newPos)
            ghost_dist_score += (-3 + newScaredTimes[i])*(12/(dist+.01))**2

        # TODO:
        # Consider the amount of walls between pacman and ghosts, pacman should
        # pay significantly more attention to ghosts who do not have any walls
        # between them and pacman

        # TODO:
        # Pacman should prefer positions which contain food and by extension, he
        # should prefer positions that are close to food. Counting the number of
        # pellets within a certain distance could be useful.
        food_dist_score = 0
        for y, row in enumerate(newFood):
            for x, food in enumerate(row):
                if food:
                    dist = manhattanDistance((x,y), newPos)
                    food_dist_score += 1/(dist+.1)

        # TODO:
        # Highly incentivise positions that increase score
        diff_score = 0
        score_diff = successorGameState.getScore() - currentGameState.getScore()
        if score_diff > 0:
            diff_score = 100

        # TODO:
        # Incentivise, to the utmost, avoiding positions which ghosts are in

        # TODO:
        # Prefer non-stopped actions
        direction_score = 0
        if action == 'Stop': direction_score += (-100)

        return sum([ghost_dist_score,
                    food_dist_score,
                    direction_score,
                    successorGameState.getScore()])

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
