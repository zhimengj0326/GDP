import numpy as np

def kernelList(restrictiveU):
    if restrictiveU:
        return ['uniform', 'triangle', 'cosinus', 'epanechnikov1', 'epanechnikov2', 'epanechnikov3']
    else:
        return ['gaussian', 'cauchy', 'picard']

def kernel(kernelString):
    if kernelString == 'gaussian':
        return gaussianKernel
    elif kernelString == 'cauchy':
        return cauchyKernel
    elif kernelString == 'picard':
        return picardKernel
    elif kernelString == 'uniform':
        return uniformKernel
    elif kernelString == 'triangle':
        return triangleKernel
    elif kernelString == 'cosinus':
        return cosKernel
    elif kernelString == 'epanechnikov1':
        return epanichnikov1
    elif kernelString == 'epanechnikov2':
        return epanichnikov1
    elif kernelString == 'epanechnikov3':
        return epanichnikov1
    else:
        raise NameError('Kernel function not found! Use a valid kernel function.')
    
def gaussianKernel(u, derivative):
    K = 1 / np.sqrt(2 * np.pi) * np.exp(-np.power(u, 2) / 2)
    if derivative == 0:
        return K
    else:
        dK = -u * K
        
        if derivative == 1:
            return K, dK
        else:
            ddK = (np.power(u, 2) - 1) * K
            
            return K, dK, ddK
        
def cauchyKernel(u, derivative):
    K = 1 / np.sqrt(np.pi * (1+np.power(u, 2)))
    if derivative == 0:
        return K
    else:
        dK = -2*u /(1+np.power(u, 2)) * K
        
        if derivative == 1:
            return K, dK
        else:
            ddK = (-2*np.power(1+np.power(u, 2),2) + 8*np.power(u, 2)) / \
                  (np.pi * np.power(1+np.power(u, 2),3))
            
            return K, dK, ddK
        
def picardKernel(u, derivative):
    K = 1 / 2 * np.exp(-np.abs(u))
    if derivative == 0:
        return K
    else:
        dK = -np.sign(u) * K
        
        if derivative == 1:
            return K, dK
        else:
            ddK = K
            
            return K, dK, ddK
        
def uniformKernel(u, derivative):
    indeces_u = np.abs(u) > 1
    
    K = 1 / 2 * np.ones(u.shape)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = np.zeros(u.shape)
        
        if derivative == 1:
            return K, dK
        else:
            ddK = np.zeros(u.shape)
            
            return K, dK, ddK
        
def triangleKernel(u, derivative):
    indeces_u = np.abs(u) > 1
    
    K = 1 - np.abs(u)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = -np.sign(u)
        dK[indeces_u] = 0
        
        if derivative == 1:
            return K, dK
        else:
            ddK = np.zeros(u.shape)
            
            return K, dK, ddK
        
def cosKernel(u, derivative):
    indeces_u = np.abs(u) > 1
    
    K = np.pi/4 * np.cos(np.pi/2*u)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = -np.power(np.pi,2)/8 * np.sin(np.pi/2*u)
        dK[indeces_u] = 0
        
        if derivative == 1:
            return K, dK
        else:
            ddK = -np.power(np.pi,3)/16 * np.cos(np.pi/2*u)
            ddK[indeces_u] = 0
            return K, dK, ddK
        
def epanechnikovKernel(u, derivative, p):
    indeces_u = np.abs(u) > 1
    
    if p == 1:
        Cp = 3/4
    elif p == 2:
        Cp = 15/16
    elif p == 3:
        Cp = 35/32
    else:
        raise ValueError('Wrong p! Use 1, 2 or 3.')
    
    K = Cp * np.power(1-np.power(u,2),p)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = -2*p*Cp*u*np.power(1-np.power(u,2),p-1)
        dK[indeces_u] = 0
        
        if derivative == 1:
            return K, dK
        else:
            ddK = 2*p*Cp*(2*np.power(u,2)*np.power(1-np.power(u,2),p-2)\
                           -np.power(1-np.power(u,2),p-1))
            ddK[indeces_u] = 0
            return K, dK, ddK
        
def epanichnikov1(u, derivative):
    return epanechnikovKernel(u, derivative, p = 1)

def epanichnikov2(u, derivative):
    return epanechnikovKernel(u, derivative, p = 2)

def epanichnikov3(u, derivative):
    return epanechnikovKernel(u, derivative, p = 3)

def nadarayaWatsonEstomator(u_feature, y_feature, kernelstring, h, scaleKernel=True, \
                            derivative=0):
    m_u = u_feature.shape
    
    u = u_feature / h

    kernelFunction = kernel(kernelstring)
    
    if derivative == 0:
        K = kernelFunction(u, derivative)
    elif derivative == 1:
        K, dK = kernelFunction(u, derivative)
    else:
        K, dK, ddK = kernelFunction(u, derivative)

    # print(f'a={a}')
    # print(f'b={b}')
    # print(f'K={K}')
    # print(f'm_u={m_u}')
    # print(f'u_feature={u_feature}')
    
    if scaleKernel:
        Kh = K / h

        a = np.dot(Kh, y_feature)
        b = np.sum(Kh, axis=1)
    else:
        a = np.dot(K, y_feature)
        b = np.sum(K, axis=1)
    
    # print(f'b={b}')
    # print(f'a={a}')
    # print(f'Kh={Kh}')
    # m = a / b
    m = np.divide(a, b)
    
    if np.any(np.isnan(m)):
        if scaleKernel:
            m[np.isnan(m)] = np.sum(y_feature / h) / np.sum(1 / h)
        else:
            m[np.isnan(m)] = np.sum(y_feature)
            
    if derivative == 0:
        return m, b / K.shape[1]
    else:        
        if scaleKernel:
            dKh = -(K + u * dK) / np.power(h, 2)
            da = dKh * y_feature
            db = dKh
        else:
            dK = -u / h * dK
            da = dK * y_feature
            db = dK
            
        if h.size == 1:
            da = np.sum(da, axis=1)
            db = np.sum(db, axis=1)
        else:
            a = a.reshape((m_u[0], 1))
            b = b.reshape((m_u[0], 1))
        
        dm = da / b - a * db / np.power(b, 2)
        
        if np.any(np.isnan(dm)):
            dm[np.isnan(dm)] = 0
            
        if np.any(np.isinf(dm)):
            dm[np.isinf(dm)] = 0
        
        if derivative == 1:
            return m, dm
        else:
            if scaleKernel:
                ddKh = -(u * dK + K) / np.power(h, 3) - u / h * (dKh + u * ddK)
                dda = ddKh * y_feature
                ddb = ddKh
            else:
                ddK = u / np.power(h, 2) * (2 * dK + u * ddK)
                dda = ddK * y_feature
                ddb = ddK
                
            if h.size == 1:
                dda = np.sum(dda, axis=1)
                ddb = np.sum(ddb, axis=1)
                
            ddm = (dda * b - 2 * da * db - a * ddb) / np.power(b, 2) + \
                  (a * np.power(db, 2) / np.power(b, 3))
            
            if np.any(np.isnan(ddm)):
                if scaleKernel:
                    ddm[np.isnan(ddm)] = -np.sum(y_feature / h) / np.sum(1 / h)
                else:
                    ddm[np.isnan(ddm)] = -np.sum(y_feature)
            
            return m, dm, ddm