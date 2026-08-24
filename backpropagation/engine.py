class Value:
    def __init__(self, data, children = (), op = ''):
        self.data = data
        self._children = children
        self._op = op
        self.label = ""
        self.grad = 0
        self.isL = 1

    def __repr__(self):
        return f"Value({self.label} = {self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        self.isL = 0
        other.isL = 0
        return out

    def __mul__(self, other):
            out = Value(self.data * other.data, (self, other), '*')
            self.isL = 0
            other.isL = 0
            return out

    #displaying object information
    def disp(self):
         if len(self._children) > 1:
            print(f"\n++++++++++++++\n{self.label} = {self.data}  \nchildren = {self._children[0].label, self._children[1].label}  \nop = {self._op} \ngrad = {self.grad}" )
         else:
            print(f"\n++++++++++++++\n{self.label} = {self.data} \nchildren = non \nop = {self._op} \ngrad = {self.grad}")


         

    
