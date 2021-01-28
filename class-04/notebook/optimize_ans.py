import numpy as np

def simple_grad_descent(x, f, df, eta, sols, stop=0.0001, max_iter=50) :
    """
    This function is simple 1D gradient descent
    x        : Initial point
    f        : Objective function
    df       : Derivative 
    eta      : step size
    sols     : List for update history
    stop     : stop criterion
    max_iter : Maximum number of iterations to prevent infinite loops
    """
    c = 100         # 임의 설정된 초기 기울기
    i = 0         # 반복 번수
    
    while np.abs(c) > stop and i < max_iter:
        ##################################
        # WRITE YOUR CODE HERE
        # 1. c에 미분계수를 계산하여 c에 할당하고
        # 2. x를 업데이트 하세요.
        c = df(x)
        x = x - eta*c

        sols.append((x, f(x)))
        print("|c|={:f}, f(x)={:f}, x={:f}".format(np.abs(c), f(x), x))
        i+=1
    
    
def f_alpha(alpha, fun, x, s, args=()) :
    """
    This is a one-dimensional version of the objective function
    given by the parameter alpha
    
    alpha : 1D independent variable
    fun   : Original objective function
    x     : Start point
    s     : 1D search direction
    args  : Tuple extra arguments passed to the objective function
    """
    x_new = x + alpha * s
    
    return fun(x_new, *args)

def gss(fun, dfun, x, s, args=(), delta=1.0e-2, tol=1e-15):
    '''
    Line search function by golden section search
    https://en.wikipedia.org/wiki/Golden-section_search and [arora]
    
    fun   : Original objective function
    dfun  : Objective function gradient which is not used
    x     : Start point
    s     : 1D search directin
    args  : Tuple extra arguments passed to the objective function
    delta : Init. guess interval determining initial interval of uncertainty
    tol   : stop criterion
    '''
    gr = (np.sqrt(5) + 1) / 2
    
    ########################################################################################
    # ESTABLISH INITIAL DELTA
    # 초기 delta를 잡는다.
    # alpah = 0에서 값과 delta에서의 함수값을 계산하고 delta에서의 값이 크다면 delta를 줄인다.
    ########################################################################################
    AL = 0.
    FL = f_alpha(AL, fun, x, s, args)
    AA = delta
    FA = f_alpha(AA, fun, x, s, args)
    while  FL < FA :
        delta = 0.1*delta
        AA = delta
        FA = f_alpha(AA, fun, x, s, args)
    ########################################################################################
    
    ########################################################################################
    # ESTABLISH INITIAL INTERVAL OF UNCERTAINTY
    # delta를 사용하여 초기 불확정 구간을 설정한다.
    # 결정된 구간을 [AL, AU] 로 두고 황금분할 탐색을 시작한다.
    ########################################################################################
    j = 1
    AU = AA + delta * (gr**j)
    FU = f_alpha(AU, fun, x, s, args)
    while FA > FU :
        AL = AA
        AA = AU
        FL = FA
        FA = FU
        
        j += 1
        AU = AA + delta * (gr**j)
        FU = f_alpha(AU, fun, x, s, args)

    AB = AL + (AU - AL) / gr
    FB = f_alpha(AB, fun, x, s, args)
    
    while abs(AA - AB) > tol:
        if f_alpha(AA, fun, x, s, args) < f_alpha(AB, fun, x, s, args):
            AU = AB
        else:
            AL = AA

        # we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        AA = AU - (AU - AL) / gr
        AB = AL + (AU - AL) / gr

    return ( (AU + AL) / 2, )


def numer_grad(x, fun, args, dtype='float64'):
    """
    Find the first derivative of a function at a point x.
    
    x     : The point at which derivative is found.
    fun   : Input function.
    args  : Tuple extra arguments passed to fun
    dtype : Data type of gradient, must be set to 'longdouble' if numerically unstable 
    """
    g = np.zeros(x.shape[0]).astype(dtype)
    
    for i in range(x.shape[0]) :
        dx1 = x.copy()
        dx2 = x.copy()
        
        # h 결정 대충 변수의 1%정도
        # https://en.wikipedia.org/wiki/Numerical_differentiation
        h = np.sqrt(np.finfo(np.float64).eps) if x[i] == 0.0 else np.sqrt(np.finfo(np.float64).eps) * x[i]
        # xph = w[i] + h
        # h = xph - w[i]
        
        ##########################################################
        # 식으로는 동일하나 아래쪽으로 더 많이 구현됨
        # dx1[i] += h/2.
        # dx2[i] -= h/2.
        # g[i] = ( fun(dx1,*args) - fun(dx2,*args)) / h
        
        dx1[i] += h
        dx2[i] -= h
        g[i] = ( fun(dx1, *args) - fun(dx2, *args) ) / (2*h)
        
    return g


def minimize(fun, x0, args=(), method='CGFR', jac=None, linesrch=gss, lineargs={},
             max_iter=2500, eps=1.0e-7, strict=True, callback=None, 
             verbose=True, verbose_step=10, debug=False) :
    """
    Minimize the given function, fun.
    
    fun          : The objective function to be minimized. fun(x, *args) -> float
    x0           : Initial guess.
    args         : Tuple extra arguments passed to the objective function
    method       : 'CGFR'    : Fletcher-Reeves conjugate gradient
                   'CGPR'    : Polak and Ribiere
                   Otherwise : steepest descent
    jac          : Method for computing the gradient vector
    linesrch     : Line search(1D optimize) function, default : golden section search(gss)
    lineargs     : Dict. extra arguments passed to the line search function
    max-iter     : Maximum number of iterations to perform.
    eps          : Stop criterion
    strict       : True : Check if the objective function is always decreasing
    verbose      : Print process info.
    verbose_step : Set how often to print process info.
    debug        : Print debug info.
    """
    
    methods = {'CGFR':'Conjugate gradient Fletcher-Reeves', 'CGPR':'Conjugate gradient Polak and Ribiere'}
    
    #################################################################
    # 0. 변수 초기화
    #################################################################
    x = x0   # 초기 시작점
    d = np.zeros(x.shape[0]) # 강하 방향    
    beta = 0.0 # 공액 경사도법에서 쓰는 beta
    
    ################################################################
    # 정보 출력
    #################################################################
    if verbose == True :
        print("################################################################")
        print("# START OPTIMIZATION")
        print("################################################################")
        print("INIT POINT : {}, dtype : {}".format(x, x.dtype))
        print("METHOD     : {}".format(methods[method]  if method in ["CGFR","CGPR"] else "Steepest Descent"))
        print("##############")
        print("# START ITER.")
        print("##############")

    for i in range(max_iter): # while True :
        ###################################################
        # 1. 그래디언트 계산 시작
        ###################################################
        if jac == None :
            c = numer_grad(x, fun, args)
        else :
            c = jac(x, *args)

        if debug == True :
            print("{}th gradient {}".format(i, c))
        ###################################################
        # 1. 그래디언트 계산 끝
        ###################################################
        
        ###################################################
        # 2. 정지 기준 체크 시작
        ###################################################
        if np.linalg.norm(c) < eps :
            if verbose==True :
                print('Stop criterion break')
                print( "Iter:{:5d}, |c|:{:11.7f}, alpha:{:.7f}, Cost:{:11.7f}, d:{}, x:{}".format(i+1, np.linalg.norm(c), 
                                                                                       alpha, fun(x, *args), d, x) )
            break
        ###################################################
        # 2. 정지 기준 체크 끝
        ###################################################
        
        ###################################################
        # 3. 강하 방향 계산 시작
        ###################################################
        if i > 0 and method == "CGFR" :   
            ###############################################
            # Fletcher-Reeves (original method)
            # WRITE YOU CODE HERE, slide eq(2)
            # current gradient: c, previous gradient: c_old
            # Using np.linalg.norm() for vector norm. 
            pass
            beta = (np.linalg.norm(c) / np.linalg.norm(c_old))**2
            # WRITE YOU CODE HERE, slide eq(3)
            pass
            d = -c + beta*d_old
            ###############################################
        elif i > 0 and method == "CGPR" :   
            ###############################################
            # Polak and Ribiere
            beta = (np.dot(c, c-c_old)) / np.linalg.norm(c_old)**2
            d = -c + beta*d_old
            ###############################################
        else :
            ###############################################
            # Steepest descent method 
            # WRITE YOU CODE HERE, slide eq(1)
            pass
            d = -c 
            ###############################################

        if debug == True :
            print("{}th descent direction {}".format(i+1, d))
        ###################################################
        # 3. 강하 방향 계산 끝
        ###################################################
                
        ###################################################    
        # 4. 스탭사이즈 결정 시작
        ###################################################
        if linesrch != None :
            alpha = linesrch(fun, jac, x, d, args=args, **lineargs)[0]
        else :
            alpha = lineargs["lr"]
        
        if debug == True :
            print("{}th alpha {}".format(i+1, alpha))
        ###################################################    
        # 4. 스탭사이즈 결정 끝
        ###################################################
        
        ###################################################
        # 5. 변수 업데이트 시작
        ###################################################
        cost_old = fun(x, *args)
        
        if debug == True :
            print("{}th before update x {}".format(i+1, x))
        
        if alpha == None :
            print("line search failed")
            break
        else :
            ###############################################
            # WRITE YOU CODE HERE, slide eq(4) 
            pass
            x = x + alpha * d
            ###############################################

        if debug == True :
            print("{}th after update x {}".format(i+1, x))
        ###################################################
        # 5. 변수 업데이트 끝
        ###################################################
        
        cost_new = fun(x, *args)
        c_old = c.copy()
        d_old = d.copy()

        # Check for increasing the objective function.
        # 강하조건을 만족시키지 않을 수 있는 경우 이 조건을 끌 수 있게 해야 한다.
        if strict == True and cost_new > cost_old :
            if verbose == True :
                print("Numerical unstable break : iter:{:5d}, Cost_old:{:.7f}, Cost_new:{:.7f}".format(i+1, 
                                                                                                       cost_old, cost_new))
            break

        # print information.
        if verbose==True and i % verbose_step == 0 :
            print( "Iter:{:5d}, |c|:{:11.7f}, alpha:{:.7f}, Cost:{:11.7f}, d:{}, x:{}".format(i+1, np.linalg.norm(c), 
                                                                                       alpha, fun(x, *args), d, x) )

        if callback :
            callback(x)    
    else :    
        if verbose==True :
            print('max-iter break')
            print( "Iter:{:5d}, |c|:{:11.7f}, alpha:{:.7f}, Cost:{:11.7f}, d:{}, x:{}".format(i+1, np.linalg.norm(c), 
                                                                                       alpha, fun(x, *args), d, x) )
        
    return x