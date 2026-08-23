class Value:
    def __init__(self, data, prevs = (), op = '', label = ""):
        self.data = data
        self._prevs = set(prevs)
        self._op = op
        self.label = label

    def __repr__(self):
        return f"Value({self.label} = {self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
            out = Value(self.data * other.data, (self, other), '*')
            return out


#testing
a = Value(2, label="a")
b = Value(3,label="b")


d = a + b
d.label = "d"

t = c
print(f"{t.label} = {t.data}  \nchildren = {t._prevs}  \nop = {t._op}" )
