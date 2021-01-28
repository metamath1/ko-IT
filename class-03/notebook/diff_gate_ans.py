import numpy as np

class Multiply :
    """
    https://github.com/WegraLee/deep-learning-from-scratch
    곱하기의 순전파, 역전파를 구현한 클래스
    """
    def __init__(self, no=-1):
        self.node_no = no
        
        self.x = None
        self.y = None
        self.v = 0.0
        self.dx = 0.0
        self.dy = 0.0
        
    def __repr__(self):
        return "Mutiply"
        
    def forward(self, x, y):
        ##################################################
        # WRITE YOUR CODE HERE
        self.x = x
        self.y = y
        self.v = x * y
        
        return self.v

    def backward(self, dout):
        ##################################################
        # WRITE YOUR CODE HERE
        self.dx = self.y * dout
        self.dy = self.x * dout
        
        return self.dx, self.dy
        

class Add :
    """
    https://github.com/WegraLee/deep-learning-from-scratch
    더하기의 순전파, 역전파를 구현한 클래스
    """
    def __init__(self, no=-1):
        self.node_no = no
        
        self.v = 0.0
        self.dx = 0.0
        self.dy = 0.0
        
    def __repr__(self):
        return "Add1"
    
    def forward(self, x, y):
        self.v = x + y
        
        return self.v

    def backward(self, dout):
        self.dx = dout * 1
        self.dy = dout * 1

        return self.dx, self.dy

class Addn :
    """
    입력이 n개인 더하기 게이트의 순전파, 역전파를 구현한 클래스
    포워드시 입력은 list로 받는다.
    """
    def __init__(self, no=-1):
        self.node_no = no
        
        self.v = 0.0
        self.n = 0
        self.dx = None
        
    def __repr__(self):
        return "Addn"
    
    def forward(self, x):
        """
        x : list
        """
        self.n = len(x)
        self.v = np.asarray(x).sum()
        
        return self.v

    def backward(self, dout):
        self.dx = np.ones(self.n)*dout
        
        return self.dx
    
class Dot :
    """
    내적의 순전파, 역전파를 구현한 클래스
    """
    def __init__(self, no=-1):
        self.node_no = no
        
        self.v = 0.0
    
    def __repr__(self):
        return "Dot"
    
    def forward(self, x, y) :
        self.x = x
        self.y = y    
        
        self.v = 0.0
        
        for xi, yi in zip(x,y):
            self.v += xi*yi
            
        return self.v
    
    def backward(self, dout) :
        self.dx = dout * self.y  # x와 y를 바꾼다.
        self.dy = dout * self.x

        return self.dx, self.dy

    
class Exp :
    """
    지수함수의 순전파, 역전파를 구현한 클래스
    """
    def __init__(self):
        self.x = None
    
    def __repr__(self):
        return "Exp"
    
    def forward(self, x):
        self.x = x 
        
        return np.exp(x)
    
    def backward(self, dout):
        return np.exp(self.x)*dout
    
class Inverse :
    """
    역수 함수의 순전파, 역전파를 구현한 클래스
    """
    def __init__(self):
        self.x = None
    
    def __repr__(self):
        return "Inverse"
    
    def forward(self, x) :
        self.x = x
        
        return 1./x
    
    def backward(self, dout):
        return -(1 / self.x**2)*dout
    
class Ln :
    """
    로그 함수의 순전파, 역전파를 구현한 클래스 
    """
    def __init__(self):
        self.x = None
        
    def __repr__(self):
        return "Ln"
        
    def forward(self, x):
        self.x = x
        
        return np.log(x)

    def backward(self, dout):
        dx = dout * (1./self.x)

        return dx
    
    
#######################################################
# logistic function derivative by using graph method
class Logistic :
    """
    로지스틱 함수의 순전파, 역전파를 구현한 클래스 
    """
    def __init__(self):
        self.a = Multiply()
        self.b = Exp()
        self.c = Add()
        self.d = Inverse()
    
    def __repr__(self):
        return "Logistic"
    
    def forward(self, x):
        #forward, function value
        a_fwd = self.a.forward(x, -1)
        b_fwd = self.b.forward(a_fwd)
        c_fwd = self.c.forward(b_fwd, 1)
        d_fwd = self.d.forward(c_fwd)
        # d_fwd = d.forward(c.forward(b.forward(a.forward(x, -1)), 1))
        # print("a_fwd:{:f}, b_fwd:{:f}, c_fwd:{:f}, d_fwd:{:f}".format(a_fwd, b_fwd, c_fwd, d_fwd))
        return d_fwd
    
    def backward(self, dout):
        #backward, derivative
        d_bwd = self.d.backward(1)
        c_bwd = self.c.backward(d_bwd)
        b_bwd = self.b.backward(c_bwd[0])
        a_bwd = self.a.backward(b_bwd)
        # a_bwd = a.backward(b.backward(c.backward(d.backward(1))[0]))
        # print("a_bwd:{}, b_bwd:{:f}, c_bwd:{}, d_bwd:{:f}".format(a_bwd, b_bwd, c_bwd, d_bwd))
        return a_bwd[0]*dout
    
    
