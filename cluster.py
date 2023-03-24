import numpy as np
import matplotlib.pyplot as plt

def kmeans(x_train,k):
    m,n=x_train.shape
    res=np.zeros([k,n])
    res[0]=np.array([1,2])
    res[1]=np.array([[5,4]])
    return(res)
    

def min_index(l):
    ind=0
    for i in range(len(l)):
        if l[i]<l[ind]:
            ind=i
    return(ind)

def progress(x_train,label,k):
    m=x_train.shape[0]
    xtemp=np.transpose(x_train)
    x=xtemp[0]
    y=xtemp[1]
    plt.grid(True)
    color_list=['blue','red','green','yellow','orange','brown']
    l=list()
    for i in range(k):
        l.append([[],[]])
    for i in range(m):
        for j in range(k):
            if label[i]==j:
                l[j][0].append(xtemp[0][i])
                l[j][1].append(xtemp[1][i])
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    c=0
    for i in range(k):
        plt.scatter(l[i][0],l[i][1],marker='x',c=color_list[i],label=('cluster'+str(i+1)))
    plt.legend()
    plt.show()
        


def classify_kmeans(x_train,k=2):
    m,n=x_train.shape
    label=np.zeros(m)
    label-=1
    c=0
    x_means=kmeans(x_train,k) #x_means is a matrix of dimension(k,n)
    distmp=np.zeros(k)
    distmp+=100
    while c!=1:
        domain=np.arange(k)
        dist=np.zeros(k)
        for i in range(m):
            for j in range(k):
                dist[j]=np.sum((x_means[j]-x_train[i])**2)
            if((distmp[0]-dist[0])<0.1):
                c=1
            distmp=np.array(dist)
            label[i]=min_index(dist)
        cluster_collection=list()
        cluster_centroids=np.zeros([k,n])
        for i in range(k):
            cluster=list()
            for j in range(m):
                if i==label[j]:
                    cluster.append(x_train[j])
                    cluster_centroids[i]+=x_train[j]
            cluster_centroids[i]/=len(cluster)    
            cluster_collection.append(cluster)
        x_means=np.array(cluster_centroids)
        print(f"-------------\ndist--->{dist}\n--------------\n")
        if n==2:
            progress(x_train,label,k)
            
        
    
    
    
    
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    