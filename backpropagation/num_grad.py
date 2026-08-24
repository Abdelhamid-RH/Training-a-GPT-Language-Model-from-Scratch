"""
This is my version of autograd I made it using differentiation by adding a small error h = 0.001
recursively instead of analytical derivatives. While it correctly approximates the gradient which is 
the objective of backpropagation it won't scale well to a large neural network without memory overhead 
and floating-point issues I will be mainly using it to test and debug micrograd
it also assumes that variables are not repeated in one function and only supports addition and multiplication
"""
class Value:
    def __init__(self, data = 0, children = (), op = ''):
        self.data = data
        self._children = children
        self._op = op
        self.label = ""
        self.grad = 0
        self.isf = 1 # 1 if the current object value is the function f 0 if not
        

    def __repr__(self):
        return f"Value({self.label} = {self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        self.isf = 0
        other.isf = 0
        return out

    def __mul__(self, other):
            out = Value(self.data * other.data, (self, other), '*')
            self.isf = 0
            other.isf = 0
            return out

    #displaying object information
    def disp(self):
         if len(self._children) > 1:
            print(f"\n++++++++++++++\n{self.label} = {self.data}  \nchildren = {self._children[0].label, self._children[1].label}  \nop = {self._op} \ngrad = {self.grad}" )
         else:
            print(f"\n++++++++++++++\n{self.label} = {self.data} \nchildren = non \nop = {self._op} \ngrad = {self.grad}")

    # calculating the partial derivitive of a value with repsect to each of its variable and store each partial derivitive in its coresponding variable value object
    def __cal_local_partials(self):
        h = 0.001
        if self.isf == 1:
            self.grad = 1
        if len(self._children) < 1:
            return
        else: 
            if self._op == "+":
                self._children[0].grad = (((self._children[0].data + h) + self._children[1].data) - self.data) / h
                self._children[1].grad = (((self._children[0].data) + self._children[1].data  + h) - self.data) / h
            elif self._op == "*":
                self._children[0].grad = (((self._children[0].data + h) * self._children[1].data) - self.data) / h
                self._children[1].grad = (((self._children[0].data) * (self._children[1].data + h)) - self.data) / h

        self._children[0].__cal_local_partials()
        self._children[1].__cal_local_partials()

    # applying the chain rule
    def __cal_global_partials(self):
        if self.isf == 1:
            self.__cal_local_partials()
        else:
            pass

        if len(self._children) < 1:
            return
        else:
            self._children[0].grad = self._children[0].grad * self.grad
            self._children[1].grad = self._children[1].grad * self.grad

        self._children[0].__cal_global_partials()
        self._children[1].__cal_global_partials()

    
    # collectting parial derivatives in one vector which is the gradient 
    def __collect_gradient(self):
        if self._op == "":
            self.__gradient.append({self.label : round(self.grad,3)})
        else:
            self._children[0].__collect_gradient()
            self._children[1].__collect_gradient()

    __gradient = []
    def backward(self):
        Value.__gradient = []
        self.__cal_global_partials()
        self.__collect_gradient()
        return self.__gradient



#testing

#variables
x = Value(5)
y = Value(1/2)
z = Value(3)

#function
f = x * (y + z)

x.label, y.label, z.label = "x", "y", "z"



print(f"Gradient of f: {f.backward()}")

