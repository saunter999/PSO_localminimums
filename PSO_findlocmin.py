#!/usr/bin/env python
from scipy import *
from pylab import *
from numpy import inf,random
from numpy import linalg as LA



class PSO:
        #-----------PSO parameters and objects---------#
	def __init__(self,pN,dim,max_iter):
	    self.w=1.0
	    self.wdamp=0.99
	    self.c1=2.0
	    self.c2=0.0
	    self.pN=pN
	    self.lb=-9.
	    self.ub=9.
	    self.dim=dim
	    self.max_iter=max_iter
	    self.X=zeros((self.pN,self.dim))     ## particle's positions and velocity 
	    self.V=zeros((self.pN,self.dim))
	    self.pbest=zeros((self.pN,self.dim)) ##particle's personal best position and the swarm's global best position 
	    self.gbest=zeros(self.dim)
	    self.p_fit=zeros(self.pN)            ## particle's personal best fit and the swarm's global best fit
	    self.g_fit=inf
	
        #-----------target function---------#
	def function(self,X): ##X is local
	    return X*sin(X)+X*cos(2*X)  

        #-----------PSO objects initialization---------#
	def init_Population(self):
	    for i in range(self.pN):
	        self.X[i]=self.lb*ones(self.dim)+(self.ub-self.lb)*random.rand(self.dim)
		self.V[i]=random.rand(self.dim)
		self.pbest[i]=self.X[i]
		tmp=self.function(self.X[i])
		self.p_fit[i]=tmp
		if tmp<self.g_fit:
		   self.g_fit=tmp
		   self.gbest=self.X[i]
	    

        #-------------PSO iteration-------------#
	def iterator(self):
	     locmin=[]
	     xls=[]
	     pfitls=[]
	     fitness=[]
	     for i in range(self.max_iter):
	         self.w=self.w*self.wdamp
	 	 fitness.append(self.g_fit)
		 pfitls.append(self.p_fit[0])
		 tmp=self.pbest[0][0]
		 xls.append(tmp)
		 for j in range(self.pN):
		     tmp=self.function(self.X[j])
		     if tmp<self.p_fit[j]:  ##updating personal best
		        self.pbest[j]=self.X[j]
			self.p_fit[j]=tmp
#		        if tmp<self.g_fit:  ##updating global best
#			   self.gbest=self.X[j]
#			   self.g_fit=tmp
		     self.V[j]=self.w*self.V[j]+self.c1*random.rand(self.dim)*(self.pbest[j]-self.X[j])+self.c2*random.rand(self.dim)*(self.gbest-self.X[j])
		     ##velocity normalization##
		     vnom=LA.norm(self.V[j])
		     if vnom>1.0:
		     	 self.V[j]/=vnom
		     self.X[j]+=self.V[j]
		     ##boundary regularization
		     for k in range(self.dim):
		         if self.X[j][k]<self.lb: self.X[j][k]=self.lb
		         if self.X[j][k]>self.ub: self.X[j][k]=self.ub
	     for i in range(self.pN):
	         locmin.append(self.pbest[i][0])
	     return fitness,pfitls,xls,locmin


def function(x):
	return x*sin(x)+x*cos(2*x)

def plotting(mls):
	xs=linspace(-9,9,100)
	ys=array([function(x) for x in xs ])
	plot(xs,ys)
	grid()
	for xm in mls:
	    axvline(x=xm,c='r')
	savefig("plotf.png")
    


if __name__=="__main__":
	my_pso=PSO(pN=30,dim=1,max_iter=120)
	my_pso.init_Population()
	fitness,pfitls,xls,locmins=my_pso.iterator()
	figure(2)
	title('personal_best y vs iteration')
	plot(range(my_pso.max_iter),pfitls,'*-')
	figure(3)
	title('personal_best x_0 vs iteration')
	plot(range(my_pso.max_iter),xls,'*-')
	figure(4)
	plotting(locmins)
	show()
