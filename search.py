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
from util import PriorityQueueLAS

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

#new
def getActions(path):
    actions = []
    x,y = path[0]
    for i in range(1,len(path)):
        xi,yi = path[i]
        #X val compare
        if x > xi:
            actions.append(Directions.WEST)
        elif x<xi:
            actions.append(Directions.EAST)
        #y val compare
        if y<yi:
            actions.append(Directions.NORTH)
        elif y>yi:
            actions.append(Directions.SOUTH)
        x = xi
        y = yi
    return actions

def dStarSearch(problem, heuristic=nullHeuristic):
    start_time = datetime.now()
    # nodes = []
    openset = util.PriorityQueue()
    rhs, gVal = {}, {}
    km = 0
    #initiating start and goal state
    start, goal = problem.getStartState(), problem.getGoalState()
    endpath = []

    #initializing all nodes with gval and rhs value to infinity and final node with rhs value zero
    def initilize(allStates):#allStates = problem.getAllStates()
        for state in allStates:
            rhs[state] = gVal[state] = math.inf
        rhs[goal] = 0
        #as goal state is inconsistent, pushing goal state in Openset priority queue
        openset.push(goal,calculateKey(goal))

    #assigning kmod to zero when agent is on start node
    #removing this as this is causing looping for certain visited nodes
    def km0():
        nonlocal km
        km=0

    #updating the key as per gval, rhs and mahhatten distance
    def calculateKey(s):
        #we are using manhattan distance as heuristic
        nonlocal km
        distance_travelled = util.manhattanDistance(s, start)
        direct = (min(gVal[s], rhs[s]) + distance_travelled+ km, min(gVal[s], rhs[s]))
        # km0()
        return direct

    #updating the gval and rhs value at each vertex with iterations
    def updateVertex(next_state):
        if not(next_state == goal):
            temp = math.inf
            next_states = problem.getSuccessors(next_state,walls = False)
            for successor,direction, cost in next_states:
                new_gval = gVal[successor] + cost
                temp = min(temp, new_gval)
            rhs[next_state] = temp
        #removing consistent node from openset queue
        openset.remove(next_state)
        if gVal[next_state] != rhs[next_state]:
            #pushing inconsistent node in openset queue
            openset.push(next_state, calculateKey(next_state))

    #finding the shortest path
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
            else:  # underestimate recalculate next states as well
                gVal[u] = math.inf
                updateVertex(u)
                next_states = problem.getSuccessors(u, walls = False)
                #propagation of changes
                for successor, _,_ in next_states:
                    updateVertex(successor)

    slast = start
    initilize(problem.getAllStates())
    computeShortestPath()
    print("Start",start)
    print("Goal", goal)
    fluctuatingpath = []
    while start != goal:
        minimum = math.inf
        minimumState = None
        fluctuatingpath.append(start)
        next_states = problem.getSuccessors(start,walls = False)
        for successor, action ,cost in next_states:
            updatedCost = gVal[successor]+cost
            if updatedCost<minimum:
                minimum = updatedCost
                minimumState = successor
        #checking for walls/obstacles at state or not
        if problem.isThereWall(minimumState) == True:
            # if there is a wall adding it to knownWalls
            # recomputing the path with known walls
            problem.addWall(minimumState)
            km =  km + util.manhattanDistance(slast,start)
            slast = start
            updateVertex(minimumState)
            fluctuatingpath=[]
            computeShortestPath()
        else:
            endpath.append(start)
            start = minimumState
            fluctuatingpath.append(start)

    endpath.append(goal)
    actions = getActions(endpath)
    stop_time = datetime.now()
    #execution time
    elapsed_time = stop_time - start_time
    print("Execution Time: {} seconds".format(elapsed_time))
    print("Length of Path",len(actions))
    print("Number of Obstacles/ Walls",len(problem.knownWalls))
    print("Grid Size ", str(problem.height)+"x"+str(problem.width))
    problem.printPath(endpath)
    problem.drawObstacles()
    return actions

#lifelong planning astar
def lifeLongAStarSearch(problem, heuristic):

    # function directly implemented from the paper
    def calculateKey(state):
        g_rhs = min(problem.g[state], problem.rhs[state])
        return (g_rhs + heuristic(state, problem), g_rhs)

    # function directly implemented from the paper
    def initialize():
        for state in problem.getStates():
            problem.rhs[state] = float('inf')
            problem.g[state] = float('inf')
        problem.rhs[problem.dynamicStartState] = 0
        problem.U.insert(problem.dynamicStartState, calculateKey(problem.dynamicStartState))

    # function directly implemented from the paper
    def updateVertex(u):
        if u != problem.dynamicStartState:
            prevKeys = [float('inf')]
            for successor, _, cost in problem.getSuccessors(u):
                prevKeys.append(problem.g[successor] + cost)
            problem.rhs[u] = min(prevKeys)

            problem.U.remove(u)

            if problem.g[u] != problem.rhs[u]:
                problem.U.insert(u, calculateKey(u))

    # function directly implemented from the paper
    def computeShortestPath():
        goal = problem.getGoalState()
        while (problem.U.topKey() < calculateKey(goal)) or (problem.rhs[goal] != problem.g[goal]):
            u = problem.U.pop()
            if problem.g[u] > problem.rhs[u]:
                problem.g[u] = problem.rhs[u]
                # the successor function produces a tuple of state, action, cost values
                for successor, _, _ in problem.getSuccessors(u):
                    updateVertex(successor)
            else:
                problem.g[u] = float('inf')
                updateVertex(u)
                for successor, _, _ in problem.getSuccessors(u):
                    updateVertex(successor)

    # After computing the shortest path the g values are updated.
    # From goal to start we will follow the least g value among
    # the successors and get the shortest path.
    def shortestPath():
        path = []
        state = (problem.getGoalState(), None)
        path.append(state)
        while state[0] != problem.dynamicStartState:
            minimum = float('inf')
            for successor, action, _ in problem.getSuccessors(state[0]):
                if minimum > problem.g[successor]:
                    minimum = problem.g[successor]
                    # since we are going in reverse direction, we need to reverse the actions.
                    state = (successor, Actions.reverseDirection(action))
            path.append(state)
        # reversing the direction path from start to goal
        return path[::-1]

    def planning():
        path = shortestPath()
        if len(path) == 1 and path[0][0] == problem.getGoalState():
            return True
        for index in range(len(path) - 1):
            currentState, currentAction = path[index]
            nextState, _ = path[index + 1]
            problem.finalPath.append((currentState, currentAction))
            print("--> " + str(nextState))
            if problem.isObstacle(nextState):
                print("\nObstacle @ " + str(nextState))
                print("Replanning...")
                problem.insertObstacle(nextState)
                updateVertex(nextState)
                problem.dynamicStartState = currentState
                return False
            elif nextState == problem.getGoalState():
                return True

    def main():
        # initializing
        problem.U = PriorityQueueLAS()
        problem.g = {}
        problem.rhs = {}
        problem.finalPath = []
        problem.dynamicStartState = problem.getStartState()
        # initialize()
        stop = False
        print('The goal position is', problem.getGoalState())
        print("The path is: ")
        print(problem.dynamicStartState)
        while (problem.dynamicStartState != problem.getGoalState()) and not stop:
            initialize()
            computeShortestPath()
            stop = planning()
        problem.finalPath.append((problem.getGoalState(), None))
        print("\nDone Planning")
        actions = []
        states = []
        for index in range(len(problem.finalPath[:-1])):
            currentState, currentAction = problem.finalPath[index]
            nextState, _ = problem.finalPath[index + 1]
            if currentState != nextState:
                actions.append(currentAction)
                states.append(currentState)
        problem.drawObstacles()
        problem.printPath(states)
        print('Path Length: ', len(actions))
        print('Size of the Layout: ', str(problem.height) + 'x' + str(problem.width))
        print('Number of obstacles: ', len(problem.obstacles))
        return actions

    return main()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#dstar
dstar = dStarSearch

#lifelong planning astar
lastar = lifeLongAStarSearch