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
import random, util, math

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self):
        super().__init__()
        self.last_positions = []

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

        self.last_positions.append(gameState.getPacmanPosition())
        if len(self.last_positions) > 4: 
            self.last_positions.pop()

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

        return self.evaluation_function_rec(currentGameState, action, 2)

    def evaluation_function_rec(self, currentGameState, action, look_ahead_depth):

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

        # Penalize agent for returning to the previous postition
        prev_pos_score = 0
        if newPos in self.last_positions:
            prev_pos_score = -100*((self.last_positions.index(newPos)+1)**3)
        else:
            prev_pos_score = 100

        # Penalize deadends
        dead_end_score = 0
        num_moves = len(successorGameState.getLegalActions())
        if num_moves == 2: dead_end_score = -100

        # Prefer positions which are further away from ghosts
        ghost_dist_score = 0
        for i, state in enumerate(newGhostStates):
            ghost_pos = state.getPosition()
            dist_to_ghost = manhattanDistance(ghost_pos, newPos)
            ghost_dist_score += (-3 + newScaredTimes[i])\
                *(10/(dist_to_ghost+.01)**3)

        # Pacman should prefer positions which contain food and by extension, he
        # should prefer positions that are close to food. Counting the number of
        # pellets within a certain distance could be useful.
        food_dist_score = 0
        man_food_dists = []
        num_food = 0
        for x, row in enumerate(newFood):
            for y, food in enumerate(row):
                if food:
                    num_food += 1
                    man_dist_to_food = manhattanDistance((x,y), newPos)
                    man_food_dists.append(man_dist_to_food)
        if len(man_food_dists) > 0: food_dist_score += \
            (1 / (min(man_food_dists) + .1)) * 200

        # Highly incentivise positions that increase score
        diff_score = successorGameState.getScore()\
            - currentGameState.getScore()
        if diff_score > 0: 
            diff_score = 200000
        else:
            diff_score = 0

        # Prefer non-stopped actions
        direction_score = 0
        if action == 'Stop': direction_score += (-200)

        # Score successor states a depth of n
        surrounding_states_score = 0
        if look_ahead_depth > 0:

            legalMoves = successorGameState.getLegalActions()
            surrounding_states_score = sum(
                [self.evaluation_function_rec(successorGameState,
                                              move,
                                              look_ahead_depth-1)
                 for move in legalMoves]
            )
            if len(legalMoves) > 0:
                surrounding_states_score /= len(legalMoves)

        noise = random.uniform(-.1,.1)

        return sum([ghost_dist_score,
                    diff_score,
                    prev_pos_score,
                    food_dist_score,
                    direction_score,
                    dead_end_score,
                    surrounding_states_score,
                    noise])


def opposite(action):
    match action:
        case 'North':
            return 'South'
        case 'East':
            return 'West'
        case 'West':
            return 'East'
        case 'South':
            return 'North'
        case 'Stop':
            return 'Stop'
        case _:
            return 'Stop'


def euclidean_distance(x, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


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

        action, _ = self.get_action_rec(gameState, self.depth)
        return action

    def get_action_rec(self, game_state, depth, agent_index=0, action=None):

        # if state is a leaf, evaluate it a bubble it up
        if (depth == 0) or game_state.isLose() or game_state.isWin():
            return (action, self.evaluationFunction(game_state))

        # evaluate pacman agent successors
        if (agent_index == 0):

            # Get legal pacman actions
            pacman_actions = game_state.getLegalActions(agent_index)
            max_value = -math.inf
            chosen_action = None

            # Evaluate the successors for each action
            for action in pacman_actions:

                # Generate successor state
                pacman_successor = game_state.generateSuccessor(
                    agent_index, action
                )

                # evaluates action and returns its value
                _, value = self.get_action_rec(
                    pacman_successor, depth, agent_index+1, action
                )

                # keep track of which action yields the max value
                if value > max_value:
                    max_value = value
                    chosen_action = action

            return chosen_action, max_value

        # Evaluate successors for agents 1 thru gameState.getNumAgents() - 1
        else:

            # Get legal ghost actions
            ghost_actions = game_state.getLegalActions(agent_index)
            min_value = math.inf
            chosen_action = None

            # Evaluate successors for each ghost action
            for action in ghost_actions:

                # Generate successor
                ghost_successor = game_state.generateSuccessor(
                    agent_index, action
                )

                # Determine the next agent's successors to evaluate
                next_agent_index = agent_index + 1
                next_depth = depth

                # If there are no more ghosts, yield next turn to pacman
                if agent_index == game_state.getNumAgents() - 1:
                    next_depth = depth - 1
                    next_agent_index = 0

                # Determine the value of each of the successor states
                _, value = self.get_action_rec(
                    ghost_successor, next_depth, next_agent_index, action
                )

                # Keep track of which action yields the min value
                if value < min_value:
                    min_value = value
                    chosen_action = action

            return chosen_action, min_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        action, _ = self.get_action_rec(gameState, self.depth)
        return action

    def get_action_rec(self, game_state, depth, agent_index=0, action=None,
        alpha=-math.inf, beta=math.inf):

        # if state is a leaf, evaluate it a bubble it up
        if (depth == 0) or game_state.isLose() or game_state.isWin():
            return (action, self.evaluationFunction(game_state))

        # evaluate pacman agent successors
        if (agent_index == 0):

            # Get legal pacman actions
            pacman_actions = game_state.getLegalActions(agent_index)
            max_value = -math.inf
            chosen_action = None

            # Evaluate the successors for each action
            for action in pacman_actions:

                # Generate successor state
                pacman_successor = game_state.generateSuccessor(
                    agent_index, action
                )

                # evaluates action and returns its value
                _, value = self.get_action_rec(
                    pacman_successor, depth, agent_index+1, action, alpha, beta
                )

                # keep track of which action yields the max value
                if value > max_value:
                    max_value = value
                    chosen_action = action

                # Prune if conditions met
                if max_value > beta: return chosen_action, max_value

                # Update alpha
                alpha = max(alpha, value)

            return chosen_action, max_value

        # Evaluate successors for agents 1 thru gameState.getNumAgents() - 1
        else:

            # Get legal ghost actions
            ghost_actions = game_state.getLegalActions(agent_index)
            min_value = math.inf
            chosen_action = None

            # Evaluate successors for each ghost action
            for action in ghost_actions:

                # Generate successor
                ghost_successor = game_state.generateSuccessor(
                    agent_index, action
                )

                # Determine the next agent's successors to evaluate
                next_agent_index = agent_index + 1
                next_depth = depth

                # If there are no more ghosts, yield next turn to pacman
                if agent_index == game_state.getNumAgents() - 1:
                    next_depth = depth - 1
                    next_agent_index = 0

                # Determine the value of each of the successor states
                _, value = self.get_action_rec(
                    ghost_successor, next_depth, next_agent_index, action,
                    alpha, beta
                )

                # Keep track of which action yields the min value
                if value < min_value:
                    min_value = value
                    chosen_action = action

                # Prune if conditions met
                if min_value < alpha: return chosen_action, min_value

                # Update beta
                beta = min(beta, min_value)

            return chosen_action, min_value


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

        action, _ = self.get_action_rec(gameState, self.depth)
        return action

    def get_action_rec(self, game_state, depth, agent_index=0, action=None):

        # if state is a leaf, evaluate it a bubble it up
        if (depth == 0) or game_state.isLose() or game_state.isWin():
            return (action, self.evaluationFunction(game_state))

        # evaluate pacman agent successors
        if (agent_index == 0):

            # Get legal pacman actions
            pacman_actions = game_state.getLegalActions(agent_index)
            max_value = -math.inf
            chosen_action = None

            # Evaluate the successors for each action
            for action in pacman_actions:

                # Generate successor state
                pacman_successor = game_state.generateSuccessor(
                    agent_index, action
                )

                # evaluates action and returns its value
                _, value = self.get_action_rec(
                    pacman_successor, depth, agent_index+1, action
                )

                # keep track of which action yields the max value
                if value > max_value:
                    max_value = value
                    chosen_action = action

            return chosen_action, max_value

        # Evaluate successors for agents 1 thru gameState.getNumAgents() - 1
        else:

            # Get legal ghost actions
            ghost_actions = game_state.getLegalActions(agent_index)
            total_value = 0
            chosen_action = random.choice(ghost_actions)

            # Evaluate successors for each ghost action
            for action in ghost_actions:

                # Generate successor
                ghost_successor = game_state.generateSuccessor(
                    agent_index, action
                )

                # Determine the next agent's successors to evaluate
                next_agent_index = agent_index + 1
                next_depth = depth

                # If there are no more ghosts, yield next turn to pacman
                if agent_index == game_state.getNumAgents() - 1:
                    next_depth = depth - 1
                    next_agent_index = 0

                # Determine the value of each of the successor states
                _, value = self.get_action_rec(
                    ghost_successor, next_depth, next_agent_index, action
                )

                total_value += value

            expected_value = total_value / len(ghost_actions)

            return chosen_action, expected_value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:

    Calculated and returned the sum of the following values:

    ghost_dist_score
        This value represents the inverse of the distance between pacman
        and ghosts in the game.
    food_dist_score
        This value is the inverse of the distance between pacman and food
        one the board
    num_food_score
        This value increases greatly as food is removed from the board to
        make the value of consuming food much higher than being close to
        food
    capsule_dist_score
        This value is the inverse of the distance between pacman and the
        closest power pellet. Weighted much less than the distance to food
        but enough to make pacman grab one if he is pretty close to one
    num_capsule_score
        This value increases greatly as power pellets are removed from the 
        board to incentivise pacman to pick them up as they provide extra
        game-score opportunities for pacman.
    dead_end_score
        This value is based off of the number of available moves that pacman
        has in a particular state. In order to try and avoid deadends, pacman
        is greatly penalized for states which contain only two possible
        actions ('stop' and one other option)
    currentGameState.getScore()
        This value is added to incentivise pacman to choose states which
        increase his score.
    noise
        A small amount of uniform noise is added to the total score in order
        to try and avoid pacman getting stuck between two states.
    """

    # Coordinate position of pacman in the successor gamestate
    newPos = currentGameState.getPacmanPosition()

    # Ascii table representing the locations in the successor state where
    # there is food on the board.
    newFood = currentGameState.getFood()

    capsules = currentGameState.getCapsules()

    # A list containing the state information for each ghost on the board.
    # Contains the coordinate position of the ghosts as well as the
    # direction that they are facing.
    newGhostStates = currentGameState.getGhostStates()

    # A list containing the number of moves that each ghost will remain
    # scared for.
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Penalize deadends
    dead_end_score = 0
    num_moves = len(currentGameState.getLegalActions())
    if num_moves == 2: dead_end_score = -100

    # Prefer positions which are further away from ghosts
    ghost_dist_score = 0
    for i, state in enumerate(newGhostStates):
        ghost_pos = state.getPosition()
        dist_to_ghost = manhattanDistance(ghost_pos, newPos)
        ghost_dist_score += (-3 + newScaredTimes[i])\
            *(10/(dist_to_ghost+.01)**3)

    # Pacman should prefer positions which contain food and by extension, he
    # should prefer positions that are close to food. Counting the number of
    # pellets within a certain distance could be useful.
    food_dist_score = 0
    man_food_dists = []
    num_food = 0
    for x, row in enumerate(newFood):
        for y, food in enumerate(row):
            if food:
                num_food += 1
                man_dist_to_food = manhattanDistance((x,y), newPos)
                man_food_dists.append(man_dist_to_food)
    if len(man_food_dists) > 0: food_dist_score += \
        (1 / (min(man_food_dists) + .1)) * 1000

    num_food_score = 40000000 / (num_food**2 +.1)

    capsule_dist_score = 0
    man_capsule_dists = []
    num_capsules = 0
    for x, row in enumerate(capsules):
        for y, cap in enumerate(row):
            if cap:
                num_capsules += 1
                man_dist_to_capsule = manhattanDistance((x,y), newPos)
                man_capsule_dists.append(man_dist_to_capsule)
    if len(man_capsule_dists) > 0: capsule_dist_score += \
        (1 / (min(man_capsule_dists) + .1))

    num_capsule_score = 40000000 / (num_capsules**2 +.1)


    noise = random.uniform(-.1,.1)

    return sum([ghost_dist_score,
                food_dist_score,
                num_food_score,
                capsule_dist_score,
                num_capsule_score,
                dead_end_score,
                currentGameState.getScore(),
                noise])


# Abbreviation
better = betterEvaluationFunction
