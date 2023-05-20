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
from util import Stack, Queue, PriorityQueue, Counter, PriorityQueueWithFunction

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

def depthFirstSearch(problem: SearchProblem):
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
    # Since this is DFS, we use a stack to explore neighbors in a first-come, first-serve manner,
    # similar to a pre-order traversal
    frontier = Stack()

    # States will be denoted by a tuple (coordinate; (x,y), actionPath: String Array)
    start_state = (problem.getStartState(), [])

    # Push initial state to the frontier and create an empty array to keep track of future visited states
    frontier.push(start_state)
    reached = []

    # While we still have nodes to explore
    while not frontier.isEmpty():
        # Extract next node to visit
        next_state, actions = frontier.pop()
        # If this current node is the goal state we are searching for return the path it has taken to get there
        if problem.isGoalState(next_state):
            return actions

        # If it's not a goal state, checked if we have seen this node before
        if next_state not in reached:
            # If we haven't seen it, mark it so
            reached.append(next_state)
            # Look at each neighbor, and add it to the frontier to visit it next
            # while also keeping track of the path it has taken to get to that node
            for neighbor, next_action, next_cost in problem.getSuccessors(next_state):
                # Successors have the format of a triple (neighborCoord: (x,y), nextAction: String, nextCost: Integer)
                # We create our state by creating a tuple of the following format:
                # (neighborCoord: (x,y), [prevActions] + [nextAction]: Concatenation of Arrays)
                frontier.push((neighbor, actions + [next_action]))

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Since this is BFS, we use a queue to explore neighbors by layers, similar to a level-order traversal
    frontier = Queue()

    # States will be denoted by a tuple (coordinate; (x,y), actionPath: String Array)
    start_state = (problem.getStartState(), [])

    # Push initial state to the frontier and create an empty array to keep track of future visited states
    frontier.push(start_state)
    reached = []

    # While we still have nodes to explore
    while not frontier.isEmpty():
        # Extract next node to visit
        next_state, actions = frontier.pop()

        # If this current node is the goal state we are searching for return the path it has taken to get there
        if problem.isGoalState(next_state):
            return actions

        # If it's not a goal state, checked if we have seen this node before
        if next_state not in reached:
            # If we haven't seen it, mark it so
            reached.append(next_state)
            # Look at each neighbor, and add it to the frontier to visit it next
            # while also keeping track of the path it has taken to get to that node
            for neighbor, next_action, next_cost in problem.getSuccessors(next_state):
                # Successors have the format of a triple (neighborCoord: (x,y), nextAction: String, nextCost: Integer)
                # We create our state by creating a tuple of the following format:
                # (neighborCoord: (x,y), prevActions + [nextAction]: Concatenation of Arrays)
                frontier.push((neighbor, actions + [next_action]))

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    """
    Initialize Data Structures to use for Search Algorithm:
        1) Frontier: A Priority Queue that will use the edge weights (or cost) as a priority, said PQ is implemented
                     with a min heap, so the minimum edge weight/cost will be the highest priority.
        2) costs: A modified version of a dictionary that defaults any new entry to 0. Said structure will be used 
                  to keep track of edge costs as the algorithm performs and will modify edge costs per visited node.
        3) reached: An array used to keep track of states that have already been visited such that the algorithm
                    does not visit them again, this expanding a much less amount of states.
    """
    frontier = PriorityQueue()
    costs = Counter()
    start_state = (problem.getStartState(), [])
    frontier.push(start_state, 0)
    reached = []

    # While the frontier still has states to expand, keep expanding
    while not frontier.isEmpty():

        # Inspect the next state with minimum cost from the frontier, since this is a PQ implemented with a min heap
        next_state, actions = frontier.pop()

        # If this state is the goal state, we're done. Return the actions it took to get to that goal state.
        if problem.isGoalState(next_state):
            return actions

        # If we have not seen this state yet, we have to expand it and mark it as visited
        if next_state not in reached:
            reached.append(next_state)

            # For every neighboring state, add the cost to the counter and sum up the cost from the previous state to
            # its total value, then add it to the frontier with its new cost and actions it took
            for neighbor, next_action, next_cost in problem.getSuccessors(next_state):
                costs[neighbor] = costs[next_state] + next_cost
                frontier.push((neighbor, actions + [next_action]), costs[neighbor])



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    """
    Initialize Data Structures to use for Search Algorithm:
        1) Frontier: A Priority Queue that will use the edge weights (or cost) as a priority, said PQ is implemented
                     with a min heap, so the minimum edge weight/cost will be the highest priority.
        2) costs: A modified version of a dictionary that defaults any new entry to 0. Said structure will be used 
                  to keep track of edge costs as the algorithm performs and will modify edge costs per visited node.
                  For this version, we will be adding costs based on what the heuristic function tell us that the cost
                  is, not the distance from the starting position like in uniform cost search.
        3) reached: An array used to keep track of states that have already been visited such that the algorithm
                    does not visit them again, this expanding a much less amount of states.
    """
    frontier = PriorityQueue()
    costs = Counter()
    reached = []
    start_state = (problem.getStartState(), [])

    # We will always modify costs based on what the heuristic function tells us what
    # the cost for that state is based on the problem given
    costs[str(start_state[0])] += heuristic(start_state[0], problem)
    frontier.push(start_state, costs[str(start_state[0])])

    # While the frontier still has states to expand, keep expanding
    while not frontier.isEmpty():

        # Inspect the next state with minimum cost from the frontier, since this is a PQ implemented with a min heap
        next_state, actions = frontier.pop()

        # If this state is the goal state, we're done. Return the actions it took to get to that goal state.
        if problem.isGoalState(next_state):
            return actions

        # If we have not seen this state yet, we have to expand it and mark it as visited
        if next_state not in reached:
            reached.append(next_state)

            # For every neighboring state, add the cost to the counter and sum up the cost from the previous state to
            # its total value, then add it to the frontier with its new cost and actions it took
            for neighbor, next_action, next_cost in problem.getSuccessors(next_state):
                total_actions = actions + [next_action]

                # The costs in A* search will be calculated as the costs of all the previous total actions taken and
                # whatever the heuristic estimates the cost will be from that neighboring state to the goal.
                costs[str(neighbor)] = problem.getCostOfActions(total_actions) + heuristic(neighbor, problem)
                frontier.push((neighbor, total_actions), costs[str(neighbor)])


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
