import numpy as np
import matplotlib.pyplot as plt

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

def softmax(x,theta):
    d,n=theta.shape
    s=0
    z=np.zeros(d)
    for j in range(d):
        v=np.exp(np.dot(theta[j][0:n-1],x)+theta[j,n-1])
        s+=v
        z[j]=v
    z=z/s
    return(z)

def evaluate_softmax(x,wb):
    pass

def cross_entropy(ypprei,yptraini):
    s=0
    l=np.log(ypprei)
    s-=np.dot(l,yptraini)
    return(s)

def J_softmax(yppre,yptrain):
    s=0
    m=yppre.shape[0]
    for i in range(m):
        s+=cross_entropy(yppre[i],yptrain[i])
    return(s)

def ReLU(z):
    l=np.array(z)
    for i in range(l.shape[0]):
        l[i]=max(z[i],0)
    return(l)        

def derivative_sigmoid(z):
    res=np.exp(-z)/((1+np.exp(-z))**2)
    return(res)

def derivative_ReLU(z):
    res=np.array(z)
    for i in range(res.shape[0]):
        if res[i]>=0:
            res[i]=1
        else:
            res[i]=0
    return(res)
           

def evaluate_ReLU(x,wb):
    m,n=wb.shape
    w=wb[:,0:n-1]
    b=wb[:,-1]
    res=np.matmul(w,x)+b
    return(ReLU(res))

def evaluate_sigmoid(x,wb):
    m,n=wb.shape
    wbtmp=np.transpose(wb)
    w=np.transpose(wbtmp[0:n-1])
    b=np.array(wbtmp[n-1])
    res=np.matmul(w,x)+b
    return(sigmoid(res))
    
def J(yptrain,yppre):
    m=yptrain.shape[0]
    err=(1/(2*m))*(yppre-yptrain)**2
    return(np.sum(err))

def max_index(l):
    ind=0
    for i in range(len(l)):
        if l[i]>l[ind]:
            ind=i
    return(ind)


def Domain(y_train):
    dom=list()
    for x in y_train:
        if x not in dom:
            dom.append(x)
    dom.sort()
    return(dom)
    
    
def indexof(ele,l):
    for i in range(len(l)):
        if l[i]==ele:
            return(i)
    else:
        return(-1)

def classify(x_train,y_train,alpha=0.01,N=41,k=2,dim_hidden=[16,16]):
    m,n=x_train.shape
    epsiloncol=list()
    costcol=list()
    domain=Domain(y_train)
    d=len(domain)
    yptrain=np.zeros([m,d])
    for row in range(m):
        yptrain[row][indexof(y_train[row],domain)]=1
        
    #Preparation of the collections of w and b vectors, 
    #and their derivatives
    #stored in the form of two-dimensional matrices
    wbcol=list()
    derivatives_col=list()
    pdim=n
    for c in range(k):
        dim=dim_hidden[c]
        wb=np.zeros([dim,pdim+1])
        pdim=dim
        wbcol.append(wb)
        derivatives_col.append(wb)
    wbcol.append(np.zeros([d,dim+1]))
    derivatives_col.append(np.zeros([d,dim+1]))
    
    
    yppre=np.zeros([m,len(domain)])
    epsilon = 0
    while epsilon<=N:
        layers=list()
        pdim=n
        xc_prev=x_train
        for c in range(k):
            dim=dim_hidden[c]
            wb=wbcol[c]
            xc=np.zeros([m,dim])
            for i in range(m):
                xc[i]=evaluate_ReLU(xc_prev[i],wb)
            layers.append(xc)
            xc_prev=xc
        yppre=np.zeros([m,d])
        for i in range(m):
            yppre[i]=softmax(xc[i],wbcol[-1])
            
                
        pdim=dim_hidden[-1]
        for i in range(m):
            wb=wbcol[-1]
            pdim=dim_hidden[-1]
            s=0
            deda=np.zeros(pdim)
            dedz=np.zeros(d)
            for j in range(d):
                dedz[j]=yppre[i][j]-yptrain[i][j]
            for ki in range(pdim):
                s=0
                for j in range(d):
                    s+=dedz[j]*wb[j][ki]
                    derivatives_col[-1][j][ki]+=dedz[j]*layers[-1][i][ki]
                deda[ki]=s
                
            for j in range(d):
                s+=dedz[j]*wb[j][ki]
                derivatives_col[-1][j][-1]+=dedz[j]
            
            c=k-1
            while c>=0:
                wb=wbcol[c]
                dim=dim_hidden[c]
                if c==0:
                    pdim=n
                    player=x_train
                else:
                    pdim=dim_hidden[c-1]
                    player=layers[c-1]
                s=0
                dedz=deda*derivative_ReLU(layers[c][i])
                deda=np.zeros(pdim)
                
                for ki in range(pdim):
                    s=0
                    for j in range(dim):
                        s+=dedz[j]*wb[j][ki]
                        derivatives_col[c][j][ki]+=dedz[j]*player[i][ki]
                        
                    deda[ki]=s
                    
                for j in range(dim):
                    derivatives_col[c][j][-1]+=dedz[j]
                c-=1
                
        for c in range(k+1):
            wbcol[c]=wbcol[c]-(alpha*derivatives_col[c])
        
        cost=J_softmax(yppre,yptrain)
        if epsilon%50==0:
            print("-----------------"*5)
            print(f"epsilon---> {epsilon}\nCost Function(J)---> {cost}")
            print("-----------------"*5)
        epsiloncol.append(epsilon)
        costcol.append(cost)
        epsilon+=1
    plt.grid(True)
    plt.plot(epsiloncol,costcol,c='red')
    plt.xlabel('No. of Iterations')
    plt.ylabel('Cost Function(J)')
    plt.show()
    return(wbcol)

def predict(x_train,wbcol):
    m,n=x_train.shape
    k=len(wbcol)
    xc_prev=x_train
    for c in range(k):
        wb=wbcol[c]
        dim=wb.shape[0]
        xc=np.zeros([m,dim])
        for i in range(m):
            xc[i]=evaluate_ReLU(xc_prev[i],wb)
        xc_prev=xc
    yppre=np.zeros([m,wbcol[-1].shape[0]])
    for i in range(m):
        yppre[i]=softmax(xc[i],wbcol[-1])
    return(yppre)
    
            
                
            
            
                
            
            
            
                
                
                
                
            
                
        
        
        
        
            
    
    
    
    
    
    
    
    
    
            
        
    
        
                
                    
                            
                        
                        
                        
                        
                        
                    
                    
                    
            
                
        
        
               
                
    
        
            
