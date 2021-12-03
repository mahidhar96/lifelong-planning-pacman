# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from datetime import datetime
from game import Directions
from game import Actions
import math
from util import PriorityQueueLPA

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print(problem)

    openSet = util.Stack()#openSet is a Stack in this case, to maintain fringe, unvisited nodes
    closedSet = []#to maintain visited nodes
    actions = []#maintain a list of actions, so you dont have to backtrack to reach the goal for pacman
    startState = problem.getStartState()

    if problem.isGoalState(startState):#return empty list, no actions after this goal reached
        return []

    openSet.push((startState,actions))#push start state and actions to reach state to stack

    while not openSet.isEmpty():
        current_state,actions = openSet.pop()
        if current_state not in closedSet:
            closedSet.append(current_state)
            if problem.isGoalState(current_state):#return actions, reached goal
                return actions
            for nextState, action,cost in problem.getSuccessors(current_state):#fetch new actions and add to actions
                actionsToState = list(actions)
                actionsToState.append(action)
                openSet.push((nextState,actionsToState))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    openSet = util.Queue()#openSet is a Queue in this case, to maintain fringe, unvisited nodes
    ##rest of the code same as dfs
    closedSet = []#to maintain visited nodes
    actions = []#maintain a list of actions, so you dont have to backtrack to reach the goal for pacman
    startState = problem.getStartState()

    if problem.isGoalState(startState):#return empty list, no actions after this goal reached
        return []

    openSet.push((startState,actions))#push start state and actions to reach state to stack

    while not openSet.isEmpty():
        current_state,actions = openSet.pop()
        if current_state not in closedSet:
            closedSet.append(current_state)
            if problem.isGoalState(current_state):#return actions, reached goal
                return actions
            for nextState, action,cost in problem.getSuccessors(current_state):#fetch new actions and add to actions
                actionsToState = list(actions)
                actionsToState.append(action)
                openSet.push((nextState,actionsToState))
    return actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    openSet = util.PriorityQueue()
    closedSet = []
    actions = []
    cost=0
    startState = problem.getStartState()

    if problem.isGoalState(startState):
        return []

    openSet.push((startState, actions, cost), cost)#saving priority in the node as pop doesn't return priority 

    while not openSet.isEmpty():
        currentState, actions, parentNodeCost = openSet.pop()
        if currentState not in closedSet:
            closedSet.append(currentState)
            if problem.isGoalState(currentState):
                return actions
            for nextState, action, cost in problem.getSuccessors(currentState):
                updatedPriority = cost + parentNodeCost
                actionsToState = list(actions)
                actionsToState.append(action)
                openSet.push((nextState, actionsToState, updatedPriority), updatedPriority)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    openSet = util.PriorityQueue()
    closedSet = []
    actions = []
    cost=0
    startState = problem.getStartState()

    if problem.isGoalState(startState):
        return []

    openSet.push((startState, actions, cost), cost)#saving priority in the node as pop doesn't return priority 

    while not openSet.isEmpty():
        currentState, actions, parentNodeCost = openSet.pop()
        if currentState not in closedSet:
            closedSet.append(currentState)
            if problem.isGoalState(currentState):
                # print(actions)
                return actions
            for nextState, action, cost in problem.getSuccessors(currentState):
                updatedPriority = cost + parentNodeCost
                actionsToState = list(actions)
                actionsToState.append(action)
                pathHeuristicCost = updatedPriority + heuristic(nextState, problem)#only difference between UCS and A* is
                #priorities are not just based on costs, but also on the added heuristics
                openSet.push((nextState, actionsToState, updatedPriority), pathHeuristicCost)

def dStarSearch(problem, heuristic=nullHeuristic):
    start_time = datetime.now()
    openset = util.PriorityQueue()
    rhs, gVal = {}, {}
    km = 0
    #setting startState and goalState
    start, goal = problem.getStartState(), problem.getGoalState()
    endpath = []
    actions = []

    #setting gVal and rhs values of nodes to infinity
    def initilize(allStates):#allStates = problem.getAllStates()
        for state in allStates:
            rhs[state] = gVal[state] = math.inf
        #goal node (from where the seach will start, will have rhs as 0)
        rhs[goal] = 0
        #pushing goal state in Openset priority queue(considered incosistent as this is where the search starts from)
        openset.push(goal,calculateKey(goal))

    #calculating the keys for everyn node in the new format
    #first with heuristic and key modifier, second with only distance
    def calculateKey(s):
        #using non local km to access km initialized in dstar
        nonlocal km
        #HEURISTIC: change the heuristic here for evaluations
        heuristic = util.manhattanDistance(s, start)
        direct = (min(gVal[s], rhs[s]) + heuristic + km, min(gVal[s], rhs[s]))
        return direct

    #check for inconsistencies and update gVal and rhs for the state
    def updateVertex(next_state):
        if not(next_state == goal):
            temp = math.inf
            next_states = problem.getSuccessors(next_state,walls = False)
            for successor,direction, cost in next_states:
                new_gval = gVal[successor] + cost
                temp = min(temp, new_gval)
            rhs[next_state] = temp
        #removing node from openset queue as it is consistent and discovered
        openset.remove(next_state)
        if gVal[next_state] != rhs[next_state]:#checking inconsistency
            #pushing node back openset queue if node inconsistent, calculate key again
            openset.push(next_state, calculateKey(next_state))

    #Initial shortest path from goal to start, without any knowledge
    #KnownWalls = []
    def computeShortestPath():
        # print(rhs[start], gVal[start],openset.peek()[0], calculateKey(start),)     
        while openset.peek()[0] < calculateKey(start) or rhs[start] != gVal[start]:
            kold = openset.peek()[0]
            u = openset.pop()
            if kold < calculateKey(u): #correct estimate push it
                openset.push(u,calculateKey(u))
            elif gVal[u] > rhs[u]: # overestimate fix it
                gVal[u] = rhs[u]
                next_states = problem.getSuccessors(u, walls = False)
                for successor, _,_ in next_states:
                    updateVertex(successor)
            else:  # underestimate recalculate and propagate to next states as well
                gVal[u] = math.inf
                updateVertex(u)
                next_states = problem.getSuccessors(u, walls = False)
                #propagation of changes
                for successor, _,_ in next_states:
                    updateVertex(successor)

    slast = start
    nextAction = None
    initilize(problem.getAllStates())
    computeShortestPath()
    print("Start",start)
    print("Goal", goal)
    # IMPORTANT:
    # This is where the code slightly differs from the paper
    # We are not using chaging the path or wall structure as
    # Pacman is moving, instead we are blinding pacman, by 
    # restricting him to only immediate actions and not letting
    # pacman see walls. 
    while start != goal:
        minimum = math.inf
        # maintain optimal state/ state with min cost
        optimalState = None
        # walls is set to falls for algorithm implementation
        # getSuccessors will return all actions including walls
        next_states = problem.getSuccessors(start,walls = False)
        for successor, action ,cost in next_states:
            updatedCost = gVal[successor]+cost
            #update optimalState
            if updatedCost<minimum:
                minimum = updatedCost
                optimalState = successor
                nextAction = action
        # checking for walls/obstacles
        if problem.isThereWall(optimalState) == True:
            # if there is a wall adding it to knownWalls
            problem.addWall(optimalState)
            # update key modifier, to give the remaining nodes
            # more weight, so that this path will have lesser 
            # probability to be chosen
            km =  km + util.manhattanDistance(slast,start)
            # assigning start again as start will change as pacman discovers
            # new paths
            slast = start
            #now update the state value and queue
            updateVertex(optimalState)
            # recomputing the path with known walls
            computeShortestPath()
        else: #no walls continue in the path
            endpath.append(start)
            actions.append(nextAction)
            start = optimalState

    #while is skipping he goal, so adding it here
    endpath.append(goal)
    actions.append(nextAction)

    stop_time = datetime.now()
    #execution time, for more accurate representation
    elapsed_time = stop_time - start_time
    print("Execution Time: {} seconds".format(elapsed_time))
    print("Length of Path",len(actions))
    print("Number of Obstacles/ Walls",len(problem.knownWalls))
    print("Grid Size ", str(problem.height)+"x"+str(problem.width))
    
    #reused from astar lifelong
    problem.printPath(endpath)
    problem.drawObstacles()

    return actions

#lifelong planning astar
def lifelongPlanningAStarSearch(problem, heuristic):

    def calculateKey(state):
        g_rhs = min(problem.g[state], problem.rhs[state])
        return (g_rhs + heuristic(state, problem), g_rhs)

    def initialize():
        for state in problem.getStates():
            problem.g[state] = float('inf')
            problem.rhs[state] = float('inf')
        problem.rhs[problem.currentStartState] = 0
        problem.U.push(problem.currentStartState, calculateKey(problem.currentStartState))

    def updateVertex(u):
        if u != problem.currentStartState:
            pathCosts = [float('inf')]
            for successorState, _, cost in problem.getSuccessors(u):
                pathCosts.append(problem.g[successorState] + cost)
            problem.rhs[u] = min(pathCosts)
            problem.U.remove(u)

        if problem.g[u] != problem.rhs[u]:
            problem.U.push(u, calculateKey(u))

    def computeShortestPath():
        goal = problem.getGoalState()
        while (problem.U.topKey() < calculateKey(goal)) or (problem.rhs[goal] != problem.g[goal]):
            u = problem.U.pop()
            if problem.g[u] > problem.rhs[u]:
                problem.g[u] = problem.rhs[u]
                for successorState, _, _ in problem.getSuccessors(u):
                    updateVertex(successorState)
            else:
                problem.g[u] = float('inf')
                updateVertex(u)
                for successorState, _, _ in problem.getSuccessors(u):
                    updateVertex(successorState)

    def createPath():
        path = []
        state = (problem.getGoalState(), None)
        path.append(state)
        while state[0] != problem.currentStartState:
            tempMin = float('inf')
            for successor, direction, _ in problem.getSuccessors(state[0]):
                if tempMin > problem.g[successor]:
                    tempMin = problem.g[successor]
                    # Reverse the action direction, since we are going from goal to start.
                    state = (successor, Actions.reverseDirection(direction))
            path.append(state)
        # reverse the list since we calculated path from goal to start here
        return path[::-1]

    def pathPlanning():
        path = createPath()
        if len(path) == 1 and path[0][0] == problem.getGoalState():
            return True
        for i in range(len(path) - 1):
            currentState, currentDirection = path[i]
            nextState, _ = path[i+1]
            problem.finalPath.append((currentState, currentDirection))
            print(str(nextState))
            if problem.isObstacle(nextState):
                print("\nObstacle @ " + str(nextState))
                problem.addToObstaclesSet(nextState)
                updateVertex(nextState)
                problem.currentStartState = currentState
                return False
            elif nextState == problem.getGoalState():
                return True

    def main():
        # initialization
        problem.U = PriorityQueueLPA()
        problem.g = {}
        problem.rhs = {}
        problem.finalPath = []
        problem.currentStartState = problem.getStartState()
        stop = False
        print('Goal position: ', problem.getGoalState())
        print("Calculated path: ")
        print(problem.currentStartState)

        while (problem.currentStartState != problem.getGoalState()) and not stop:
            initialize()
            computeShortestPath()
            stop = pathPlanning()

        problem.finalPath.append((problem.getGoalState(), None))
        directions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentDirection = problem.finalPath[index]
            nextState, _ = problem.finalPath[index + 1]
            if currentState != nextState:
                directions.append(currentDirection)
                states.append(currentState)

        problem.drawObstacles()
        problem.printPath(states)
        print('Path Length: ', len(directions))
        print('Number of obstacles: ', len(problem.obstacles))
        return directions

    return main()

def dStarSearchOptimized(problem, heuristic=nullHeuristic):
    start_time = datetime.now()
    openset = util.PriorityQueue()
    rhs, gVal = {}, {}
    km = 0
    #setting startState and goalState
    start, goal = problem.getStartState(), problem.getGoalState()
    endpath = []
    actions = []

    #setting gVal and rhs values of nodes to infinity
    def initilize(allStates):#allStates = problem.getAllStates()
        for state in allStates:
            rhs[state] = gVal[state] = math.inf
        #goal node (from where the seach will start, will have rhs as 0)
        rhs[goal] = 0
        #pushing goal state in Openset priority queue(considered incosistent as this is where the search starts from)
        openset.push(goal,calculateKey(goal))

    #calculating the keys for everyn node in the new format
    #first with heuristic and key modifier, second with only distance
    def calculateKey(s):
        #using non local km to access km initialized in dstar
        nonlocal km
        #HEURISTIC: change the heuristic here for evaluations
        heuristic = util.manhattanDistance(s, start)
        direct = (min(gVal[s], rhs[s]) + heuristic + km, min(gVal[s], rhs[s]))
        return direct

    #check for inconsistencies and update gVal and rhs for the state
    def updateVertex(next_state):
        if not(next_state == goal):
            temp = math.inf
            next_states = problem.getSuccessors(next_state,walls = False)
            for successor,direction, cost in next_states:
                new_gval = gVal[successor] + cost
                temp = min(temp, new_gval)
            rhs[next_state] = temp
        #removing node from openset queue as it is consistent and discovered
        openset.remove(next_state)
        if gVal[next_state] != rhs[next_state]:#checking inconsistency
            #pushing node back openset queue if node inconsistent, calculate key again
            openset.push(next_state, calculateKey(next_state))

    #Initial shortest path from goal to start, without any knowledge
    #KnownWalls = []
    def computeShortestPath():
        # print(rhs[start], gVal[start],openset.peek()[0], calculateKey(start),)     
        #early termination, doesn't wait till rhs[start] becomes gVal[start]
        while openset.peek()[0] < calculateKey(start) or rhs[start] > gVal[start]:
            kold = openset.peek()[0]
            u = openset.pop()
            if kold < calculateKey(u): #correct estimate push it
                openset.push(u,calculateKey(u))
            elif gVal[u] > rhs[u]: # overestimate fix it
                gVal[u] = rhs[u]
                next_states = problem.getSuccessors(u, walls = False)
                for successor, _,_ in next_states:
                    updateVertex(successor)
            else:  # underestimate recalculate and propagate to next states as well
                gVal[u] = math.inf
                updateVertex(u)
                next_states = problem.getSuccessors(u, walls = False)
                #propagation of changes
                for successor, _,_ in next_states:
                    updateVertex(successor)

    slast = start
    nextAction = None
    initilize(problem.getAllStates())
    computeShortestPath()
    print("Start",start)
    print("Goal", goal)
    # IMPORTANT:
    # This is where the code slightly differs from the paper
    # We are not using chaging the path or wall structure as
    # Pacman is moving, instead we are blinding pacman, by 
    # restricting him to only immediate actions and not letting
    # pacman see walls. 
    while start != goal:
        minimum = math.inf
        # maintain optimal state/ state with min cost
        optimalState = None
        # walls is set to falls for algorithm implementation
        # getSuccessors will return all actions including walls
        next_states = problem.getSuccessors(start,walls = False)
        for successor, action ,cost in next_states:
            updatedCost = gVal[successor]+cost
            #update optimalState
            if updatedCost<minimum:
                minimum = updatedCost
                optimalState = successor
                nextAction = action
        # checking for walls/obstacles
        if problem.isThereWall(optimalState) == True:
            # if there is a wall adding it to knownWalls
            problem.addWall(optimalState)
            # update key modifier, to give the remaining nodes
            # more weight, so that this path will have lesser 
            # probability to be chosen
            km =  km + util.manhattanDistance(slast,start)
            # assigning start again as start will change as pacman discovers
            # new paths
            slast = start
            #now update the state value and queue
            updateVertex(optimalState)
            # recomputing the path with known walls
            computeShortestPath()
        else: #no walls continue in the path
            endpath.append(start)
            actions.append(nextAction)
            start = optimalState

    #while is skipping he goal, so adding it here
    endpath.append(goal)
    actions.append(nextAction)

    stop_time = datetime.now()
    #execution time, for more accurate representation
    elapsed_time = stop_time - start_time
    print("Execution Time: {} seconds".format(elapsed_time))
    print("Length of Path",len(actions))
    print("Number of Obstacles/ Walls",len(problem.knownWalls))
    print("Grid Size ", str(problem.height)+"x"+str(problem.width))
    
    #reused from astar lifelong
    problem.printPath(endpath)
    problem.drawObstacles()

    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#dstar
dstar = dStarSearch

#dstarOptimized
dstaropt = dStarSearchOptimized

#lifelong planning astar
lastar = lifelongPlanningAStarSearch