import random
from typing import Generic, TypeVar

State = TypeVar('State')
Action = TypeVar('Action')

class Monte_Carlo_Tree(Generic[State]):
    class Node(Generic[State, Action]):
        def __init__(self, state: State, action: Action=None, parent=None):
            self.score = 0
            self.state = state
            self.action = action
            self.parent = parent
            self.children = []
        
        def update(self, result):
            self.score += result
        
        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            return str(self.state) + str(self.action) + str(self.score)

    def __init__(self, init_state: State, player: object, simulate_step: lambda state, player, state_constraints: (Node, int), init_action: Action=None):
        self.root = Monte_Carlo_Tree.Node(init_state, init_action)
        self.player = player
        self.simulate_step = simulate_step

    def update(self, state: State):
        """ Updates tree to current state. If the state already exist, simply change root else create new node. """

        for child in self.root.children:
            if child.state == state:
                self.root = child
                break
        else:
            self.root = Monte_Carlo_Tree.Node(state)

    def search(self, node: Node=None):
        """ Searches tree using Monte Carlo Tree Search. """

        if node == None:
            node = self.root

        for _ in range(100):
            next_node, next_state = self.expand(self.select(node)) 

            node.children.append(next_node)
            next_node.parent = node

            if next_state == 0:
                last_node, last_state = self.simulate(next_node)
                self.backpropagate(last_node, last_state)
    
    def current_state(self) -> State:
        """ Returns the current state of the tree. """
        return self.root.state

    def best_action(self) -> Action:
        """ Returns the next best action from the root node and updates the root node. """

        self.search()
        node = self.best_node()

        self.root = node

        return node.action

    def best_node(self, node: Node=None) -> Node:
        """ Returns the best child node relative to current node. """

        if node == None:
            node = self.root

        if len(node.children) == 0:
            return node
        else:
            return max(node.children, key=lambda child: child.score)

    def select(self, node: Node=None) -> Node:
        """ Selects the next best node to expand on the tree. """

        if node == None:
            node = self.root

        if len(node.children) == 0:
            return node
        
        return self.select(node.children[0])
        
    def expand(self, node: Node=None) -> (Node, int):
        """ Expands a node considering current node child constraints. """

        if node == None:
            node = self.root

        next_state = self.simulate_step(node.state, self.player, [child.state for child in node.children])

        return next_state

    def simulate(self, node: Node=None) -> (Node, int):
        """ Simulates current node using simulate_step until completion. Returns final node. """

        if node == None:
            node = self.root
        
        next_node, state = self.simulate_step(node.state, self.player)

        next_node.parent = node

        node.children.append(next_node)

        if state == 0:
            return self.simulate(next_node)
        else:
            return (next_node, state)
    
    def backpropagate(self, node: Node, result = None):
        """ Backpropagates the result from the current node to the parent nodes.  """

        if node != None:
            if result == None:
                result = node.score
            else:
                node.update(result)

            self.backpropagate(node.parent, result)