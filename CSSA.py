import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cec2017.functions import *
from cec2019comp100digit import cec2019comp100digit

'''优化函数'''
def fun(x):
    # bench = cec2019comp100digit
    # bench.init(4, 10) # Init function 3
    # return bench.eval(x)
    o = f4(x)
    return o

'''优化函数'''
# y = x^2, 用户可以自己定义其他函数
def fun1(X):
    output = sum(np.square(X))
    return output
#y5=sum(ix^4)+random[0,1),搜索范围[-1.28,1.28]
# def fun(X):
#     output1 = 0
#     for i in range(pop):
#         output1 += (i+1)*(X[i])**4
#         output = output1+random.random()
#     return output
# y4 , 搜索范围[-100,100]
# def fun(X):
#     output = max(abs(X))
#     return output
#y5 ,搜索范围[-1.28,1.28]
# def fun(X):
#     output1 = 0
#     for i in range(pop):
#         output1 += (i+1)*(X[i])**4
#         output = output1+random.random()
#     return output
# y6 = sum abs(x)+, 用户可以自己定义其他函数[-500,500]
# def fun(X):
#     output = sum(-X*np.sin(np.sqrt(np.abs(X))))
#     return output
#低维函数y10
# def fun(x):
#     output = 4*np.square(x[0])-2.14*(x[0])**4+(1/3)*(x[0])**6+(x[0])*(x[1])-4*np.square(x[1])+4*(x[1])**4
    # return output
# a14 = [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
#     [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]
# def fun14(x):
#     output = 0
#     for i in range(25):
#         output += ((i+1)+(x[0]-a14[0][i])**(6)+(x[1]-a14[1][i])**(6))**(-1)
#     o = (0.002+output)**(-1)
#     return o

''' Tent映射 '''
def Tent(Max_iter):
    x = np.zeros([Max_iter,1])
    x[0] = random.random() #初始点
    a = 0.7 #参数a的值
    for i in range(Max_iter - 1):
        if x[i]<a:
            x[i+1]=x[i]/a
        if x[i]>=a:
            x[i+1]=(1-x[i])/(1-a)
    
    return x
    

''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        TentValue = Tent(dim)
        for j in range(dim):
            X[i, j] = TentValue[j]*(ub[j] - lb[j]) + lb[j]
            if X[i,j]>ub[j]:
                X[i, j] = ub[j]
            if X[i,j]<lb[j]:
                X[i, j] = lb[j]
    
    return X,lb,ub
            
'''边界检查函数'''
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X
    
    
'''计算适应度函数'''
def CaculateFitness(X,fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index


'''根据适应度对位置进行排序'''
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

'''麻雀发现者更新'''
def PDUpdate(X,PDNumber,ST,Max_iter):
    X_new  = X
    R2 = random.random()
    for j in range(PDNumber):
        if R2<ST:
            X_new[j,:] = X[j,:]*np.exp(-j/(random.random()*Max_iter))
        else:
            X_new[j,:] = X[j,:] + np.random.randn()*np.ones([1,dim])
    return X_new
        
'''麻雀加入者更新'''            
def JDUpdate(X,PDNumber,pop,dim):
    X_new = X
    for j in range(PDNumber+1,pop):
         if j>(pop - PDNumber)/2 + PDNumber:
             X_new[j,:]= np.random.randn()*np.exp((X[-1,:] - X[j,:])/j**2)
         else:
             #产生-1，1的随机数
             A = np.ones([dim,1])
             for a in range(dim):
                 if(random.random()>0.5):
                     A[a]=-1       
             AA = np.dot(A, (np.dot(A.T,A)))
             X_new[j,:]= X[1,:] + np.abs(X[j,:] - X[1,:])*AA.T
           
    return X_new                    
            
'''危险更新'''   
def SDUpdate(X,pop,SDNumber,fitness,BestF):
    X_new = X
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[0:SDNumber]
    for j in range(SDNumber):
        if fitness[SDchooseIndex[j]]>BestF:
            X_new[SDchooseIndex[j],:] = X[1,:] + np.random.randn()*np.abs(X[SDchooseIndex[j],:] - X[1,:])
        elif fitness[SDchooseIndex[j]] == BestF:
            K = 2*random.random() - 1
            X_new[SDchooseIndex[j],:] = X[SDchooseIndex[j],:] + K*(np.abs( X[SDchooseIndex[j],:] - X[-1,:])/(fitness[SDchooseIndex[j]] - fitness[-1] + 10E-8))
    return X_new
              
    

'''混沌麻雀搜索算法'''
def CSSA(pop,dim,lb,ub,Max_iter,fun):
    ST = 0.6 #预警值
    PD = 0.2 #发现者的比列，剩下的是加入者
    SD = 0.2 #意识到有危险麻雀的比重
    PDNumber = int(pop*PD) #发现者数量
    SDNumber = int(pop*SD) #意识到有危险麻雀数量
    X,lb,ub = initial(pop, dim, ub, lb) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0,:]
    GbestScore = fitness[0]
    Curve = np.zeros([MaxIter,1])
    for i in range(MaxIter):
        
        BestF = fitness[0]
        
        X = PDUpdate(X,PDNumber,ST,Max_iter)#发现者更新
        
        X = JDUpdate(X,PDNumber,pop,dim) #加入者更新
        
        X = SDUpdate(X,pop,SDNumber,fitness,BestF) #危险更新
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测
        
        fitness = CaculateFitness(X,fun) #计算适应度值
        #step6 高斯变异
        avgF = np.mean(fitness)
        for j in range(pop):
            if fitness[j]<avgF:
                Temp = X[j,:]*(1 + np.random.randn()) #高斯变异
                for a in range(dim):
                    if(Temp[a]>ub[a]):
                        Temp[a] = ub[a]
                    if(Temp[a]<lb[a]):
                        Temp[a] = lb[a]
                fitnew = fun(Temp)
                if fitnew<fitness[j]:
                    X[j,:] = Temp
                    fitness[j] = fitnew
            else:
                
                TentValue = Tent(dim) #tent 扰动
                Temp = X[j,:]*(1 + TentValue.T)
                for a in range(dim):
                    if(Temp[0,a]>ub[a]):
                        Temp[0,a] = ub[a]
                    if(Temp[0,a]<lb[a]):
                        Temp[0,a] = lb[a]
                fitnew = fun(Temp[0,:])
                if fitnew<fitness[j]:
                    X[j,:] = Temp[0,:]
                    fitness[j] = fitnew
                
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        if(fitness[0]<=GbestScore): #更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0,:] = X[0,:]
        Curve[i] = GbestScore
    
    return GbestScore,GbestPositon,Curve
    
# def f4(x):
#     bench = cec2019comp100digit
#     bench.init(4, 10)  # Init function 2
#     return bench.eval(x)
#
# dim = 10 # 维度
# lb = -100 * np.ones([dim, 1])  # 下边界
# ub = 100 * np.ones([dim, 1])  # 上边界
# f = f4


'''主函数 '''
#设置参数
pop = 30 #种群数量
MaxIter = 500 #最大迭代次数
dim = 30 #维度
lb = -100*np.ones([dim, 1]) #下边界
ub = 100*np.ones([dim, 1])#上边界

GbestScore,GbestPositon,Curve = CSSA(pop,dim,lb,ub,MaxIter,fun1)
print('最优适应度值：',GbestScore)
print('最优解：',GbestPositon)
print(Curve)

#绘制适应度曲线
plt.figure(1)
plt.plot(Curve,'r-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('CSSA',fontsize='large')


#绘制搜索空间
# fig = plt.figure(2)
# ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# Z = X**2 + Y**2
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()









