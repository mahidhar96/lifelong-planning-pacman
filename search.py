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
import math

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
    directions = []

    #assigning kmod to zero when agent is on start node
    def km0():
        nonlocal km
        km=0

    #updating the key as per gval, rhs and mahhatten distance
    def calculateKey(s):
        #we are using manhatten distance as heuristic
        nonlocal km
        distance_travelled = util.manhattanDistance(s, start)
        direct = (min(gVal[s], rhs[s]) + distance_travelled+ km, min(gVal[s], rhs[s]))
        km0()
        return direct

    #ititializing all nodes with gval and rhs value to infinity and final node with rhs value zero
    def initilize(allStates):
        for state in allStates:
            rhs[state] = gVal[state] = math.inf
        rhs[goal] = 0
        #as goal state is inconsistent, pushing goal state in Openset priority queue
        openset.push(goal,calculateKey(goal))

    #updating the gval and rhs value at each vertex with iterations
    def updateVertex(next_state):
        if not(next_state == goal):
            temp = math.inf
            next_states = problem.getSuccessors(next_state)
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
        while openset.peek()[0] < calculateKey(start) or rhs[start] != gVal[start]:
            kold = openset.peek()[0]
            u = openset.pop()
            if kold < calculateKey(u):
                openset.push(u,calculateKey(u))
            elif gVal[u] > rhs[u]:
                gVal[u] = rhs[u]
                next_states = problem.getSuccessors(u)
                # print("next_states",next_states)
                for successor, _,_ in next_states:
                    updateVertex(successor)
            else:  # underestimate
                gVal[u] = math.inf
                next_states = problem.getSuccessors(u)
                next_states.append((u, 1))
                for successor, _,_ in next_states:
                    updateVertex(successor)

    slast = start
    initilize(problem.getAllStates())
    computeShortestPath()
    # nodes.append(problem._expanded)
    fluctuatingpath = []
    #performing all operations in sequence
    while start != goal:
        minimum = math.inf
        minimumState = None

        fluctuatingpath.append(start)
        next_states = problem.getSuccessors(start)
        for successor, direction ,cost in next_states:
            temp = gVal[successor]+cost
            if temp<minimum:
                minimum = temp
                minimumState = successor
        #checking wall at state or not
        if problem.isThereWall(minimumState) == True:
            problem.addWall(minimumState)
            km =  km + util.manhattanDistance(slast,start)
            slast = start
            updateVertex(minimumState)
            fluctuatingpath=[]
            computeShortestPath()
            # nodes.append(problem._expanded)
        else:
            endpath.append(start)
            start = minimumState
            fluctuatingpath.append(start)

    # print(nodes)
    endpath.append(goal)
    actions = getActions(endpath)
    stop_time = datetime.now()
    #execution time
    elapsed_time = stop_time - start_time
    print("Execution Time: {} seconds".format(elapsed_time))
    # print(actions)
    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#dstar
dstar = dStarSearch
