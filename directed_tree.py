# Nodes can be compared to one another by comparing their data.
class Node:
    def __init__(self, parent=None, data=None):
        self.parent = parent
        self.data = data
    
    def is_root(self):
        return self.parent is None
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.data == other.data
    
    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.data < other.data
    
    def __gt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.data > other.data
    
    def __le__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.data <= other.data
    
    def __ge__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.data >= other.data
    
    def __hash__(self):
        return hash(id(self))

    def __copy__(self):
        return Node(self.parent, self.data)