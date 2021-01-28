import math
import numpy as np

class Matrix :
    def __init__(self, data, precision=3):
        """
        data      : 숫자 자료를 담고 있는 리스트, 
                    중첩된 2차원 리스트 또는 그냥 1차원 리스트
        precision : 행렬이 출력될 때 요소 숫자를 소수점 몇째자리까지 찍을지 지정 
        """
        self.data = data
        
        ############################################################
        # 생성자에서 프린트 포맷을 초기화 한다.
        # 기존 코드를 그대로 재사용 한다.
        #
        # 소수점 아래 자리수, +1은 점(.) 자리수 때문
        digits2   = precision + 1
        
        # 리스트에서 가장 큰 숫자를 찾는다.
        # 리스트가 2차원 또는 1차원 모두 가능하기 때문에 체크해서 처리
        # 3번 중첩된 리스트를 넘겨 받는 다면 코드는 제대로 동작하지 않음.
        # [array([72., 47., 58.]), array([22., 19., 15.]), array([38., 21., 32.])] 와 같은 경우도 처리
        if isinstance(self.data[0], list) or isinstance(self.data[0], np.ndarray) :
            max_value = max([ max(row) for row in self.data ])
        else :
            max_value = max(self.data)
            
        if max_value < np.finfo(np.float32).eps :
            digits1 = 1
        else :            
            # 가장 큰 숫자의 정수부분의 자리수 계산
            # +2 부분 : 로그로 자리수 계산할때 +1,  log10(10) = 1 인데 2가 되어야하므로 +1
            #           부호자리수 +1 
            digits1 = int(math.log10(abs(int(max_value)))) + 2
            
        self.print_format = "{:+"+str(digits1+digits2)+"."+str(precision)+"f}"
        ############################################################
        
    def __repr__(self) :
        s = ""
        for row in self.data :
            s += '[' + ','.join( list(map(lambda e: self.print_format.format(e), row))  ) + ']\n'

        return s

        
    def __add__(self, B):
        """
        넘겨 받은 객체 B와 나 자신을 더해서 되돌린다.
        __add__ 함수에 내용을 적으면 a+b식으로 함수를 호출할 수 있다.
        즉, a+b , a.__add__(b) 는 같은 표현이다.
        """
        m = len(self.data)
        n = len(self.data[0])
        r = len(B.data)
        p = len(B.data[0])

        # 두 행렬의 크기를 비교해서 크기가 다르면 에러
        if m == r and n == p :
            C = [ [0.0 for j in range(n)] for i in range(m) ]

            # 루프를 돌면서 하나하나 더한다.
            for i in range(len(self.data)) :
                for j in range(len(self.data[i])):
                    C[i][j] = self.data[i][j]+B.data[i][j]

            return Matrix(C)
        else :
            print("Can not add two matrices")
    
    def __mul__(self, B):
        m  = len(self.data)
        n1 = len(self.data[0])
        n2 = len(B.data)
        p  = len(B.data[0])

        if n1 == n2 :
            C = [ [0.0 for j in range(p)] for i in range(m) ]
            for i in range(m) :
                for j in range(p) :
                    for k in range(n1) :
                        C[i][j] += self.data[i][k]*B.data[k][j]

            return Matrix(C)
        else :
            print("Can not multiply two matrices")
        
    def __getitem__(self, item):
        """
        item에 지정한 번호로 인덱싱을 하게 한다.
        A[1] 이면 item은 1, A[2,2]이면 item은 (2,2)가 넘어온다.
        __add__와 마찬가지로 A[1], A.__getitem__(1)은 같은 표현이다.
        이 함수의 구현 목적은 1부터 시작되는 인덱스를 흉내내기 위한 것이다.
        """
        if type(item) is int :
            item = item - 1
            if type(self.data[item]) is list :
                return Matrix(self.data[item])
            else :
                return self.data[item]
        else :
            return self.data[item[0] - 1][item[1] - 1]
    
    
    def col_combination(self, B) :
        """
        실험 1
        행렬의 곱을 열들의 선형조합으로 구현해보세요.
        numpy를 사용
        """
        m = len(self.data)
        n1 = len(self.data[0])
        n2 = len(B.data)
        p  = len(B.data[0])
        
        if n1 == n2 :
            R = np.zeros((m, p))        # 결과를 담을 배열
            C = np.asarray(self.data)   # 앞 행렬
            D = np.asarray(B.data)      # 뒤 행렬
            
            ###################################
            # WRITE YOUR CODE HERE
            # numpy를 사용하여 열의 선형조합을 구현하여 Matrix 형태로 되돌린다.
            for k in range(p):
                R[:,k] = (C*D[:,k]).sum(axis=1) #D[:,k]가 1차원 어레이로 추출되어 C*D[:,k]가능
            
            return Matrix(list(R))
            
        else :
            print("Can not multiply two matrices")

            
    """
    아래 메서드는 숙제로 제공된 함수
    각각 함수를 완성하여 03-07-hw.ipynb에서 결과를 검증하세요.
    """
    def norm(self, L) :
        """
        L  : 1 또는 2를 넘겨 받아 L1, L2 노름을 계산할 때 씀
        """
        # 행렬인가 벡터인가를 판단한다.
        m = len(self.data)
        n = len(self.data[0])
        
        if m == 1 or n == 1 : #벡터라면
            # (1,n)행렬 또는 (m,1)행렬이므로 2차원 행렬을 1차원으로 편다.
            d = [e for row in self.data for e in row]
            
            # norm을 리턴
            norm = 0.0
            
            if L == 1 :
                ###################################
                # WRITE YOUR CODE HERE
                # 여기 L1 노름을 계산하는 코드를 적으세요.
                # 루트 계산은 math.sqrt함수를 사용하세요.
                # 벡터의 모든 요소는 리스트 d에 들어있음.
                # norm += 0.0을 적당히 알맞게 고치세요.
                for i in range(m*n) :
                    norm += math.sqrt(d[i]*d[i])
                print("L1 norm be calculated.")
                
            elif L == 2 :
                ###################################
                # WRITE YOUR CODE HERE
                # 여기 L2 노름을 계산하는 코드를 적으세요.
                # 루트 계산은 math.sqrt함수를 사용하세요.
                # 벡터의 모든 요소는 리스트 d에 들어있음.
                # norm += 0.0을 적당히 알맞게 고치세요.
                for i in range(m*n) :
                    norm += d[i]*d[i]
                norm = math.sqrt(norm)
                print("L2 norm be calculated.")
                
            else :
                print("Only L1 and L2 norm can be calculated.")
            
            return norm
        
        else : # 행렬이라면
            # 행렬의 노름을 계산한다.
            print("I did not learn about matrix norm.")
    
    
    def outer_vectorwise(self, B):
        """
        행렬의 곱을 열과 행의 외적의 합으로 구현한다. 
        """
        m = len(self.data)
        n1 = len(self.data[0])
        n2 = len(B.data)
        p  = len(B.data[0])
        
        if n1 == n2 :
            R = np.zeros((m, p))        # 결과를 담을 배열
            for i in range(n1):
                ###################################
                # WRITE YOUR CODE HERE
                # 앞 행렬에서 열을 뽑고, 뒤 행렬에서 행을 뽑아
                # 행렬곱을 한 다음 결과를 누적하여 되돌린다.
                row = []
                col = []
                #앞행렬에서 열뽑기
                for j in range(m):
                    row.append(self.data[j][i])
                    
                #뒤행렬에서 행뽑기
                for j in range(p):
                    col.append(B.data[i][j])
                    
                # row, col을 외적한다.
                R_ = np.zeros((m, p))        # 외적 결과를 담을 배열
                for s in range(m):
                    for r in range(p):
                        R_[s][r] = row[s]*col[r]
                R += R_
                
            return Matrix(list(R))
        else :
            print("Can not multiply two matrices")
            
        
        