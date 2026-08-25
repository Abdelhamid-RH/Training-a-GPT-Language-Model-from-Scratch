import math

class Value:
    def __init__(self, data, children = (), op = '', label=''):
        self.data = data
        self._children = children
        self._op = op
        self.label = label
        self.grad = 0
        self._backward = lambda: None


    def __repr__(self):
        return f"Value({self.label} = {self.data})"

    #displaying object information
    def disp(self):
        if len(self._children) > 1:
            print(f"\n++++++++++++++\n{self.label} = {self.data}  \nchildren = {self._children[0].label, self._children[1].label}  \nop = {self._op} \ngrad = {self.grad}" )
        elif len(self._children) > 0:  
            print(f"\n++++++++++++++\n{self.label} = {self.data}  \nchildren = {self._children[0].label}  \nop = {self._op} \ngrad = {self.grad}" )
        else:
            print(f"\n++++++++++++++\n{self.label} = {self.data} \nchildren = non \nop = {self._op} \ngrad = {self.grad}")

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
            out = Value(self.data * other.data, (self, other), '*')
            def _backward():
                self.grad +=  out.grad * other.data
                other.grad += out.grad * self.data
            out._backward = _backward
            return out

    def tanh(self):
        x = self.data
        out = Value((math.exp(x) - math.exp(-x)) / (math.exp(x) +  math.exp(-x)), (self,), "tanh")
        def _backward():
            self.grad += (1 - out.data**2) * out.grad 
        out._backward = _backward
        return out

    
             
    def backward(self):
        srt = [] # values sorted in a topological order
        visited = set()

        def tpsort(val):
            if val not in visited:
                visited.add(val)
                for child in val._children:
                    tpsort(child)
                srt.append(val)

        tpsort(self)

        self.grad = 1
        for node in reversed(srt):
            node._backward()




    

         

    
