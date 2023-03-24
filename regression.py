"""Calculus Corporation"""
import numpy as np
import matplotlib.pyplot as plt

def scale_down(x):
    u=np.mean(x)
    sigma=np.std(x)
    x=(x-u)/sigma
    return x
    
        
def J(x_train,y_train,w,b,lamb=0):
    m=len(x_train)
    y_pre=(w*x_train)+b
    l=(y_pre-y_train)**2+(lamb*w**2)
    res=np.sum(l)/(2*m)
    return(res)   

def J_gen(x_train,y_train,w,b,lamb=0):
    m=len(x_train)
    s=0
    for i in range(m):
        s+=((np.dot(w,x_train[i])+b)-y_train[i])**2+lamb*np.sum(w**2)
    return(s/(2*m))

def R2score(x_train,y_train,w,b):
    m=x_train.shape[0]
    y_mean=np.mean(y_train)
    l=list()
    for i in range(m):
        l.append(np.dot(x_train[i],w)+b)
    y_pre=np.array(l)
    ssres=(y_pre-y_train)**2
    sstot=(y_train-y_mean)**2
    R2=1-(ssres/sstot)
    return(R2)

def progress(x_train,y_train,w,b):
    n=len(w)
    plt.grid(True)
    plt.scatter(x_train,y_train,marker='x',c='red',label='Data points')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    y_pre=np.zeros(len(x_train))+b
    for i in range(n):
        y_pre+=w[i]*(x_train**(i+1))
    plt.plot(x_train,y_pre,c='blue',label='Prediction of model')
    plt.legend()
    plt.show()
    
def linear(x_train,y_train,alpha=0.01,k=513,lamb=0):
    w=10
    b=2
    c=0
    m=len(x_train)
    while True:
        derivative_jwrtw=(1/m)*np.sum(((w*x_train)+b-y_train)*x_train)+(lamb*w)/m
        derivative_jwrtb=(1/m)*np.sum((w*x_train)+b-y_train)
        #derivative_jwrtw=(J(x_train,y_train,(w+h),b)-j)/h
        #derivative_jwrtb=(J(x_train,y_train,w,(b+h))-j)/h
        wt=w-alpha*derivative_jwrtw
        bt=b-alpha*derivative_jwrtb
        w=wt
        b=bt
        if c%50==0:
            progress(x_train,y_train,[w],b)
            print("w--->",w)
            print("b-->",b,"\n")
            print("Cost Function(J)---> ",J(x_train,y_train,w,b))
            print("count-->",c,"\n")
        if c==k:
            break
        c+=1
    return (w,b)
          
def linear_gen(x_train,y_train,alpha=0.01,k=513,lamb=0):
    b=0
    xtemp=np.transpose(x_train)
    w=np.zeros(len(xtemp))
    c=0
    m,n=x_train.shape
    while c!=(k+1):        
        l=list()
        for x in x_train:
            l.append(np.dot(w,x)+b)
        y_pre=np.array(l)
        l1=list()
        for i in range(n):
            l1.append(np.sum(xtemp[i]*(y_pre-y_train)))
        derivative_jwrtw=(1/m)*np.array(l1)+(lamb/m)
        derivative_jwrtb=(1/m)*np.sum(y_pre-y_train)
        w=w-alpha*derivative_jwrtw
        b=b-alpha*derivative_jwrtb
        j=J_gen(x_train,y_train,w,b)
        if c%50==0:
            print("-"*30)
            print("w--->",w)
            print("b-->",b,"\n")
            print("J----->",j)
            print("count-->",c,"\n")
            print("-"*30)
        c+=1
    return (w,b)

def check_alpha(x_train,y_train,alpha=0.01,k=513,lamb=0):
    xtemp=np.transpose(x_train)
    w=np.zeros(len(xtemp))
    b=1
    c=0
    m=len(x_train)
    n=len(xtemp)
    ld=list()
    lc=list()
    while c<=k:
        j=J_gen(x_train,y_train,w,b)
        lc.append(j)
        l=list()
        for x in x_train:
            l.append(np.dot(w,x)+b)
        y_pre=np.array(l)
        l1=list()
        for i in range(n):
            l1.append(np.sum(xtemp[i]*(y_pre-y_train)))
        derivative_jwrtw=(1/m)*np.array(l1)+(lamb/m)
        derivative_jwrtb=(1/m)*np.sum(y_pre-y_train)
        w=w-alpha*derivative_jwrtw
        b=b-alpha*derivative_jwrtb
        ld.append(derivative_jwrtb)
        c+=1
    print(ld)
    plt.grid(True)
    plt.plot(range(k+1),lc,c='red')
    plt.xlabel('No. of Iterations')
    plt.ylabel('Cost Function(J)')
    plt.show()

def transform(x_train,order,combination=[]):
    n=x_train.shape[1]  #number of input columns
    if type(order)==int:
        order=[order]*n  #list order contains of the maximum order for each feature
    c=len(order)
    if(c>n):
        print("Fatal ERROR!")
        return 1
    if(c<n):
        for i in range(n-c):
            order.append(1)
    xtemp=list(np.transpose(x_train))    
    i=0
    j=0
    while i<c:
        pmax=order[i]
        if type(pmax)==int:
            if pmax>1:
                u=1
                for power in range(2,pmax+1):
                    xp=xtemp[j]**power
                    xtemp.insert(j+u,xp)
                    u+=1
                j+=u
            elif pmax<1:
                xp=xtemp[j]**pmax
                xtemp.insert(j+1,xp)
                j+=2
            else:
                j+=1
        elif type(pmax)==list:
            u=1
            for power in pmax:
                xp=xtemp[j]**power
                xtemp.insert(j+u,xp)
                u+=1
            j+=u
                        
        i+=1
    xtfin=list(xtemp)
    xtemp=list(np.transpose(x_train))
    if type(combination==list):
        combination=list(set(combination))
        l=np.array([1]*x_train.shape[0])
        for x in combination:
            for i in x:
                l=l*xtemp[i-1]
            xtfin.append(l)
    else:
        print("Invalid combination!!!")
    xtfin=np.array(xtfin)
    x_train=np.transpose(xtfin)  #modification of input dataset complete
    return(x_train)

def polynomial(x_train,y_train,order=5,alpha=0.01,k=513):
    x_train=transform(x_train,order)
    xtemp=np.transpose(x_train)
    for i in range(len(xtemp)):
        if((max(xtemp[i])>=5)or(min(xtemp[i]<=-5))):
            xtemp[i]=scale_down(xtemp[i])
    x_train=np.transpose(xtemp)
    return(linear_gen(x_train,y_train,alpha,k))
    
                
            
        
        
        
        
    
