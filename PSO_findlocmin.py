#!/usr/bin/env python
from scipy import *
from pylab import *
from numpy import random
from numpy import linalg as LA

"""
1)c1 is the key parameter in the method.
A good c1 should be comparable to distances between two adjacent local minimums. 
2)wdamp=1.0 is very good for for the case of function with a single global minimum; otherwise take wdamp<1.0 
"""


class PSO:
        #-----------PSO parameters and objects---------#
	def __init__(self,pN,dim,max_iter):
	    self.w=1.0
	    self.wdamp=0.99
	    self.c1=2.0
	    self.pN=pN
	    self.lb=-12.
	    self.ub=12.
	    self.dim=dim
	    self.max_iter=max_iter
	    self.X=zeros((self.pN,self.dim))     ## particle's positions and velocity 
	    self.V=zeros((self.pN,self.dim))
	    self.pbest=zeros((self.pN,self.dim)) ##particle's personal best position  
	    self.p_fit=zeros(self.pN)            ## particle's personal best fit
	
	def print_info(self):
	    print "w=",self.w
	    print "wdamp=",self.wdamp
	    print "pN=",self.pN
	    print "dim=",self.dim
	    print "max_iter=",self.max_iter
	    print "lb=",self.lb
	    print "ub=",self.ub

        #-----------target function---------#
	def function(self,X): ##X is local
	    return X*sin(X)+X*cos(2*X)  
#	    return (X**2-5*X)*sin(3.1*X) 
# 	    return cos(14.5 * X - 0.3) + (X + 0.2) * X
# 	    return X**2-4*X+3
# 	    return (X**2-5*X)*sin(X) 
#	    return LA.norm(X)**2

        #-----------PSO objects initialization---------#
	def init_Population(self):
	    for i in range(self.pN):
	        self.X[i]=self.lb*ones(self.dim)+(self.ub-self.lb)*random.rand(self.dim)
		self.V[i]=random.rand(self.dim)
		self.pbest[i]=self.X[i]
		self.p_fit[i]=self.function(self.X[i])
	    

        #-------------PSO iteration-------------#
	def iterator(self):
	     xlocmin=[]
	     p0fitls=[]
	     x0ls=[]
	     for i in range(self.max_iter):
	         self.w*=self.wdamp
		 p0fitls.append(self.p_fit[0])
		 tmp=self.pbest[0][0]
		 x0ls.append(tmp)
		 for j in range(self.pN):
		     tmp=self.function(self.X[j])
		     if tmp<self.p_fit[j]:  ##updating personal best
		        self.pbest[j]=self.X[j]
			self.p_fit[j]=tmp
		     # Adding sign alternating term for the inertia term for the case of function with a single global minimum
		     if random.rand()<0.5: 
		        sgn=1.0
		     else:
		        sgn=-1.0
		     self.V[j]=sgn*self.w*self.V[j]+self.c1*random.rand(self.dim)*(self.pbest[j]-self.X[j])
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
	         xlocmin.append(self.pbest[i][0])
	     return p0fitls,x0ls,xlocmin



    


if __name__=="__main__":
	my_pso=PSO(pN=30,dim=1,max_iter=1000)
	my_pso.print_info()
	my_pso.init_Population()
	p0fitls,x0ls,xlocmins=my_pso.iterator()

#	print xlocmins
	figure(1)
	title('personal_best y_0 vs iteration')
	plot(range(my_pso.max_iter),p0fitls,'o-',markersize=2)
	figure(2)
	title('personal_best x_0 vs iteration')
	plot(range(my_pso.max_iter),x0ls,'o-',markersize=2)

	xs=linspace(my_pso.lb,my_pso.ub,200)
	ys=array([my_pso.function(x) for x in xs ])
	if my_pso.dim==1:
	  figure(0)
	  plot(xs,ys,'o-',markersize=2)
	  for xm in xlocmins:
	    axvline(x=xm,c='r')
	  savefig("PSO_Alllocalmins.png")
	show()
