import numpy as np
import matplotlib.pyplot as plt

e=2.72

def log(x,n=1e5):
    s=0
    i=1
    while True:
        s+=(((x-1)**i)*((-1)**(i+1)))/i
        if (i==n):
            break
        
        i+=1
    return(s)

def scale_down(x):
    #u=np.mean(x)
    u=min(x)
    d=max(x)-min(x)
    #d=np.std(x)
    x=(x-u)/d
    return x
    
def sigmoid(z):
    res=1/(1+np.exp(-z))
    return(res)

def softmax(x,theta,domain):
    d=len(domain)
    s=0
    z=np.zeros(d)
    n=theta.shape[1]
    for j in range(d):
        v=np.exp(np.dot(theta[j][0:n-1],x)+theta[j,n-1])
        s+=v
        z[j]=v
    z=z/s
    return(z)       

def J(x_train,y_train,w,b):
    m=len(x_train)
    y_pre=sigmoid((w*x_train)+b)
    l=-y_train*np.log(y_pre)-(1-y_train)*np.log(1-y_pre)
    return(np.sum(l/m))

def J_gen(x_train,y_train,w,b):
    m=len(x_train)
    s=0
    for i in range(m):
        y_pre=sigmoid(np.dot(w,x_train[i])+b)
        s+=-y_train[i]*np.log(y_pre)-(1-y_train[i])*np.log(1-y_pre)
    return(s/m)

def J_softmax(x_train,y_train,theta):
    m,n=x_train.shape
    s=0
    domain=np.array(Domain(y_train))
    for i in range(m):
        l=(1*(domain==y_train[i]))*np.log(softmax(x_train[i],theta,domain))
        s+=np.sum(l)
        
    J=-s
    return(J)

def progress(x_train,y_train,w,b):
    n=len(w)
    plt.grid(True)
    xt1=[]
    xt0=[]
    for i in range(len(x_train)):
        if y_train[i]==1:
            xt1.append(x_train[i])
        elif y_train[i]==0:
            xt0.append(x_train[i])
    plt.scatter(xt1,[1]*len(xt1),marker='x',c='red',label='True')
    plt.scatter(xt0,[0]*len(xt0),marker='o',label='False')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    y_pre=np.zeros(len(x_train))+b
    for i in range(n):
        y_pre+=w[i]*(x_train**(i+1))
    y_pre=sigmoid(y_pre)
    plt.scatter(x_train,y_pre,c='green',marker='.',label='Prediction of model')
    plt.legend()
    plt.show()
    
def linear(x_train,y_train,alpha=0.01,k=513):
    w=0
    b=0
    c=0
    m=len(x_train)
    while True:
        derivative_jwrtw=(1/m)*np.sum((sigmoid((w*x_train)+b)-y_train)*x_train)
        derivative_jwrtb=(1/m)*np.sum(sigmoid((w*x_train)+b)-y_train)
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
          
def linear_gen(x_train,y_train,alpha=0.01,k=513):
    b=0
    xtemp=np.transpose(x_train)
    w=np.zeros(len(xtemp))
    c=0
    m,n=x_train.shape
    j=J_gen(x_train,y_train,w,b)            
    while(c!=(k+1)):        
        l=list()
        for x in x_train:
            l.append(sigmoid(np.dot(w,x)+b))
        y_pre=np.array(l)
        l1=list()
        for i in range(n):
            l1.append(np.sum(xtemp[i]*(y_pre-y_train)))
        derivative_jwrtw=(1/m)*np.array(l1)
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

def Domain(y_train):
    dom=list()
    for x in y_train:
        if x not in dom:
            dom.append(x)
    dom.sort()
    return(dom)

def max_index(l):
    ind=0
    for i in range(len(l)):
        if l[i]>l[ind]:
            ind=i
    return(ind)

def linear_softmax(x_train,y_train,alpha=0.01,k=513):
    domain=Domain(y_train)
    d=len(domain)
    m,n=x_train.shape
    theta=np.zeros([d,n+1])
    b=0
    w=np.zeros(n)
    fun=np.append(w,b)
    for j in range(d):
        theta[j]=np.array(fun)
    J=J_softmax(x_train,y_train,theta)
    c=0
    xtemp=np.transpose(x_train)
    while(c!=k):
        J=J_softmax(x_train,y_train,theta)
        derivative_jwrtw=np.zeros([d,n])
        derivative_jwrtb=np.zeros(d)
        for j in range(n):
            vj=0
            for i in range(m): 
                vj+=x_train[i][j]*((softmax(x_train[i],theta,domain))-(1*(domain==y_train[i])))
            for ki in range(d):
                derivative_jwrtw[ki][j]+=vj[ki]
        for i in range(m):
            derivative_jwrtb+=(softmax(x_train[i],theta,domain))-(1*(domain==y_train[i]))
          
        for i in range(d):
            theta[i][0:n]=theta[i][0:n]-alpha*derivative_jwrtw[i]
            theta[i][n]=theta[i][n]-alpha*derivative_jwrtb[i]
        
        if c%50==0:
            print("-"*30)
            print("J----->",J)
            print("count-->",c,"\n")
            print("-"*30)
        c+=1
    return(theta)
       
def predict(x_train,w,b,threshold=0.5):
    l=[]
    m=x_train.shape[0]
    for i in range(m):
        rex=(np.dot(w,x_train[i])+b)
        rex=sigmoid(rex)
        if rex>=threshold:
            l.append(1)
        else:
            l.append(0)
    y_pre=np.array(l)
    return(y_pre)

def predict_softmax(x_train,theta,domain):
    m,n=x_train.shape
    y_pre=np.zeros(m)
    for i in range(m):
        p=softmax(x_train[i],theta,domain)
        y_pre[i]=domain[max_index(p)]
    return(y_pre)        

    
def check_alpha(x_train,y_train,alpha=0.01,k=513):
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
            l.append(sigmoid(np.dot(w,x)+b))
        y_pre=np.array(l)
        l1=list()
        for i in range(n):
            l1.append(np.sum(xtemp[i]*(y_pre-y_train)))
        derivative_jwrtw=(1/m)*np.array(l1)
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

def check_alpha_softmax(x_train,y_train,alpha=0.01,k=513):
    ld=list()
    lc=list()
    domain=Domain(y_train)
    d=len(domain)
    m,n=x_train.shape
    theta=np.zeros([d,n+1])
    b=0
    w=np.zeros(n)
    fun=np.append(w,b)
    for j in range(d):
        theta[j]=np.array(fun)
    J=J_softmax(x_train,y_train,theta)
    c=0
    while c<=k:
        J=J_softmax(x_train,y_train,theta)
        lc.append(J)
        derivative_jwrtw=np.zeros([d,n])
        derivative_jwrtb=np.zeros(d)
        for j in range(n):
            vj=0
            for i in range(m): 
                vj+=x_train[i][j]*((softmax(x_train[i],theta,domain))-(1*(domain==y_train[i])))
            for ki in range(d):
                derivative_jwrtw[ki][j]+=vj[ki]
        for i in range(m):
            derivative_jwrtb+=(softmax(x_train[i],theta,domain))-(1*(domain==y_train[i]))
          
        for i in range(d):
            theta[i][0:n]=theta[i][0:n]-alpha*derivative_jwrtw[i]
            theta[i][n]=theta[i][n]-alpha*derivative_jwrtb[i]
        ld.append(derivative_jwrtb)
        c+=1
    print(np.array(ld))
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
        if((max(xtemp[i])>1)):
            xtemp[i]=scale_down(xtemp[i])
    x_train=np.transpose(xtemp)
    return(linear_gen(x_train,y_train,alpha,k))
            
        
        
        
        
        
        
        
        
        
        
        
        
