from __future__ import division

# import matplotlib.pyplot as plt
import scipy

import scipy.misc as plt
import numpy as np
import itertools
from numpy import linalg as LA
import pandas
import time

## here I simply group by natural order
def grouper(n,iterable):
	args = [iter(iterable)]*n
	return([e for e in t if e!=None] for t in itertools.izip_longest(*args))

def soft_thresholding(x,tau):
	tmp = abs(x) - tau
	ans = np.sign(x)*(tmp >= 0) * tmp
	return ans

def group_st(x,tau):
	tmp = 1 - tau/LA.norm(x)
	ans = (tmp >=0)*tmp*x
	return ans

def data_generation(n,p,rho,gamma1,gamm2,seed):
	# generate X
	tmp = np.arange(p)
	tmp = abs(tmp[:,None] - tmp)
	np.random.seed(seed)
	cov_matrix = np.exp(tmp*np.log(rho))
	L = LA.cholesky(cov_matrix)
	X = np.array([np.random.normal(0,1,p) for i in xrange(1,n+1)])
	X = np.dot(X,L.transpose())

	# generate beta
	beta = np.zeros(p)
	groups = list(grouper(10,xrange(p)))
	selected_groups = np.random.choice(1000,gamma1,replace=False)
	selected_features = np.array([np.random.choice(groups[i],gamma2,replace=False) for i in selected_groups],dtype='int64').reshape(gamma1*gamma2,1)
	num_features = gamma1*gamma2
	beta[selected_features] = np.sign(np.random.uniform(-1,1,num_features))*np.random.uniform(0.5,10,num_features)

	# generate Y
	Y = np.dot(X,beta) + np.random.normal(0,0.01,n)

	return X,Y,groups,beta

def epsilon_norm(x,alpha,R):
	epsilon_norm = 0
	j_0 = 0
	if alpha==0 and R == 0:
		epsilon_norm = 0
	elif alpha ==0 and R != 0:
		epsilon_norm = LA.norm(x)/R
	elif R ==0:
		epsilon_norm = LA.norm(x,np.inf)/alpha
	else:
		tmp = alpha*LA.norm(x,np.inf)/(alpha+R)
		tmp_ni = [i for i in abs(x) if i > tmp]
		tmp_ni = np.sort(tmp_ni)[::-1]
		n_i = tmp_ni.size
		thresh_value = R ** 2/(alpha ** 2)
		if n_i >= 2:
			S_old = 0
			S2_old = 0
			a_old = 0
			for k in xrange(n_i-1):
				S_new = S_old + tmp_ni[k]
				S2_new = S2_old + tmp_ni[k] ** 2
				a_new = S2_new/(tmp_ni[k+1] ** 2) - 2*S_new/tmp_ni[k+1] + k+1
				if  thresh_value>= a_old and thresh_value < a_new:
					j_0 = k+1
					break
				S_old = S_new
				S2_old = S2_new
				a_old = a_new
			if j_0 == 0:
				j_0 = k+2  # because python start from 0
				S_new = S_old + tmp_ni[j_0-1]
				S2_new = S2_old + tmp_ni[j_0-1]**2
			tmp = alpha ** 2 * j_0 - R **2
			if tmp ==0:
				epsilon_norm = S2_new/(2*alpha*S_new)
			else:
				epsilon_norm = (alpha*S_new - np.sqrt(alpha**2 * S_new ** 2 - S2_new*tmp))/tmp

	return epsilon_norm

# test for above function
def epsilon_norm_testing(tmp_x,groups,epsilon_g,i):
	x = tmp_x[groups[i]]
	alpha = 1-epsilon_g[i]
	R = epsilon_g[i]
	v = epsilon_norm(x,alpha,R)
	return np.sum(np.square(soft_thresholding(x,v*alpha))) - (v*R)**2


def dual_norm(w,groups,tmp_X,num_groups,tau,epsilon_g):
	ans = np.max([epsilon_norm(tmp_X[groups[i]],
					     1-epsilon_g[i],epsilon_g[i]) for i in xrange(num_groups)])/(tau + (1-tau)*w[0])
	return ans

def primal_eval(X,Y,beta,tau,w,lambda_t,groups,num_groups):
	group_norm = np.sum([w[i]*LA.norm(beta[groups[i]]) for i in xrange(num_groups)])
	index = groups.ravel()
	ans = 0.5*LA.norm(Y - np.dot(X,beta[index]))**2 + lambda_t * (tau*LA.norm(beta[index],1) + (1-tau)*group_norm)
	return ans

def dual_eval(Y,lambda_t,theta):
	ans = 0.5*LA.norm(Y)**2 - 0.5*lambda_t**2*LA.norm(theta - Y/lambda_t)**2
	return ans

def active_groups(R,tau,w,groups,theta,num_groups,X):
	T_g = np.zeros(num_groups)
	active_g = np.zeros(num_groups)
	X_l2_norm = [LA.norm(X[:,i],2) for i in groups]
	for i in xrange(num_groups):
		X_g = X[:,groups[i]]
		tmp_X = np.dot(np.transpose(X_g),theta)
		tmp = LA.norm(tmp_X,np.inf)
		if tmp > tau:
			T_g[i] = LA.norm(soft_thresholding(tmp_X,tau)) + R*X_l2_norm[i]
		else:
			tmp_T_g = tmp + R*X_l2_norm[i] - tau
			T_g[i] = tmp_T_g * (tmp_T_g>=0)
	active_g = (T_g>=(1-tau)*w[0])*1
	return active_g

def active_features(theta,R,tau,X_g):
	num_features = X_g.shape[1]
	active_features = np.zeros(num_features)
	tmp = np.abs(np.dot(np.transpose(X_g),theta)) + R*np.sum(np.abs(X_g)**2,axis=0)**(0.5)
	active_features = (tmp>=tau)*1

	return active_features

def dst3_center(w,groups,X,Y,num_groups,tau,epsilon_g,lambda_max,lambda_t):
	tmp_X = np.dot(X.transpose(),Y)
	g_star = np.argmax([epsilon_norm(tmp_X[groups[i]],1-epsilon_g[i],epsilon_g[i]) for i in xrange(num_groups)]/(tau + (1-tau)*w[0]))
	tmp = np.dot(np.transpose(X[:,groups[g_star]]),Y/lambda_max)
	tmp_epsilon_norm = epsilon_norm(tmp,1-epsilon_g[g_star],epsilon_g[g_star])
	ksi_star = soft_thresholding(tmp,(1-epsilon_g[g_star])*tmp_epsilon_norm)
	eta = np.dot(X[:,groups[g_star]],ksi_star)
	eta = eta/(epsilon_g[g_star]*LA.norm(ksi_star) + (1-epsilon_g[g_star])*LA.norm(ksi_star,1))
	center = Y/lambda_t - eta*(np.dot(eta,Y)/lambda_t - (tau + (1-tau)*w[g_star]))/(LA.norm(eta) ** 2)
	return center

def gap_safe_rule(X,Y,p,epsilon,K,f_ce,lambda_sequence,lambda_max,tau,groups,screening=False,method=1):
	## method 1 represents gap safe;
	## method 2 represents dynamic safe region
	## method 3 represents dst3
	## method 4 represents gap sequential(I don't know how to select the radius for this one)
	## method 5 represents static safe region
	## above methods all under circumstances when screening is True

	num_groups = len(groups)
	L_gradient = np.zeros(num_groups)
	w = np.ones(num_groups) * np.sqrt(10)
	L_gradient = [LA.norm(X[:,groups[i]],2)**2 for i in xrange(num_groups)]
	X_T = np.transpose(X)
	epsilon_g = (1-tau)*w[0]/(tau + (1-tau)*w[0]) * np.ones(num_groups)
	# tmp_X = np.dot(X_T,Y)
	# lambda0 = dual_norm(w,groups,tmp_X,num_groups,tau,epsilon_g)
	T = len(lambda_sequence)
	beta = np.zeros((T+1,p))
	active_variables = np.zeros((T,11))

	for t in xrange(T):
		lambda_t = lambda_sequence[t]
		alpha_g = [lambda_t/i for i in L_gradient]
		beta_new = beta[t]
		resid = np.dot(X,beta_new) - Y
		f_gradient = np.dot(X_T,resid)
		if method == 2 or method == 5:
			center = Y/lambda_t
		if method == 5:
			R = LA.norm(Y/lambda_max - Y/lambda_t)

		step = 0
		if screening == True:
			a_groups = groups
			num_a_groups = len(a_groups)
			old_a_groups_label = np.arange(num_a_groups)
		for k in xrange(K):
			if k % f_ce == 0:
				print k
				rho_k = -resid
				tmp_X = np.dot(X_T,rho_k)
				if screening == False:
					tmp_lambda = dual_norm(w,groups,tmp_X,num_groups,tau,epsilon_g)
				else:
					tmp_lambda = dual_norm(w[old_a_groups_label],a_groups,tmp_X,num_a_groups,tau,epsilon_g[old_a_groups_label])
				theta_k = rho_k/max(tmp_lambda,lambda_t)
				if screening == False:
					tmp_difference = primal_eval(X,Y,beta_new,tau,w,lambda_t,groups,num_groups) - dual_eval(Y,lambda_t,theta_k)
				else:
					index = a_groups.ravel()
					tmp_difference = primal_eval(X[:,index],Y,beta_new,tau,w,lambda_t,a_groups,num_a_groups) - dual_eval(Y,lambda_t,theta_k)
				print tmp_difference,"\n"

				if  method == 1:
					center = theta_k
					R = np.sqrt(2*tmp_difference)/lambda_t
				elif method == 2:
					R = LA.norm(theta_k - Y/lambda_t)
				elif method == 3:
					center = dst3_center(w,groups,X,Y,num_groups,tau,epsilon_g,lambda_max,lambda_t)
					R = np.sqrt(LA.norm(Y/lambda_t - theta_k)**2 - LA.norm(Y/lambda_t - center)**2)
				elif method ==4:
					if k==0:
						center = theta_k
					R_seq = primal_eval(X[:,index],Y,beta_new,tau,w,lambda_t,a_groups,num_a_groups) - dual_eval(Y,lambda_t,center)
					R = np.sqrt(2*R_seq)/lambda_t
					print "radius for gap safe seq is ", R
					print "radius for gap safe is ", np.sqrt(2*tmp_difference)/lambda_t
					print "distance between center is ", LA.norm(center - theta_k,2)**2

				if tmp_difference < epsilon:
					if screening == True:
						non_a_groups = np.setdiff1d(xrange(num_groups),old_a_groups_label)
						if len(non_a_groups)>0:
							for i in non_a_groups:
								beta_new[groups[i]] = 0
						for (i,j) in enumerate(old_a_groups_label):
							beta_new[groups[j][active_f[i]==0]] = 0
					active_variables[t,step:-1] = active_variables[t,step-1]
					beta[t+1] = beta_new
					break
				if screening == True:
					active_g = active_groups(R,tau,w,a_groups,center,num_a_groups,X)
					new_a_groups_label = old_a_groups_label[active_g!=0]
					non_a_groups = np.setdiff1d(old_a_groups_label,new_a_groups_label)
					old_a_groups_label = new_a_groups_label
					a_groups = a_groups[active_g!=0]
					num_a_groups = len(a_groups)
					print float(num_a_groups)/num_groups
					active_f = [active_features(center,R,tau,X[:,i]) for i in a_groups]
					if np.log2(k+1) >= step:
						active_variables[t,step] = sum([sum(i) for i in active_f])
						print "currently there are ", active_variables[t,step], "variables left"
						step = step+1
					if num_a_groups==0:
						active_variables[t,step:-1] = active_variables[t,step-1]
						beta[t+1] = 0
						break

			if screening == False:
				for i in xrange(num_groups):
					if i!=0:
						resid = resid + np.dot(X[:,groups[i-1]],tmp)
					f_gradient[groups[i]] = np.dot(X_T[groups[i],:],resid)
					tmp = beta_new[groups[i]] - f_gradient[groups[i]]/L_gradient[i]
					tmp = soft_thresholding(tmp,tau*alpha_g[i])
					tmp = group_st(tmp,(1-tau)*w[i]*alpha_g[i]) - beta_new[groups[i]]
					beta_new[groups[i]] = tmp + beta_new[groups[i]]
					if i==num_groups-1:
						resid = resid + np.dot(X[:,groups[i]],tmp)
			else:
				for (i,j) in enumerate(old_a_groups_label):
					if i!=0:
						last_a_group = old_a_groups_label[i-1]
						resid = resid + np.dot(X[:,groups[last_a_group][active_f[i-1]!=0]],tmp)
					index = groups[j][active_f[i]!=0]
					f_gradient[index] = np.dot(X_T[index,:],resid)
					tmp = beta_new[index] - f_gradient[index]/L_gradient[j]
					tmp = soft_thresholding(tmp,tau*alpha_g[j])
					tmp = group_st(tmp,(1-tau)*w[j]*alpha_g[j]) - beta_new[index]
					beta_new[index] = tmp + beta_new[index]
					if i == num_a_groups-1:
						resid = resid + np.dot(X[:,index],tmp)

			if k == K-1:
				non_a_groups = np.setdiff1d(xrange(num_groups),old_a_groups_label)
				if len(non_a_groups)>0:
					for i in non_a_groups:
						beta_new[groups[i]] = 0
				for (i,j) in enumerate(old_a_groups_label):
					beta_new[groups[j][active_f[i]==0]] = 0
				beta[t+1] = beta_new

	return beta,active_variables


if __name__ == '__main__':
	n = 100
	p = 10000
	rho = 0.5
	gamma1 = 10
	gamma2 = 4
	tau = 0.2
	seed = 2

    # I save the data to local file since data generation takes a relative long time
	# X,Y,groups,beta = data_generation(n,p,rho,gamma1,gamma2,seed)
	# np.savetxt("X.txt",X)
	# np.savetxt("Y.txt",Y)
	# np.savetxt("groups.txt",groups)
	# np.savetxt("beta.txt",beta)

	X = pandas.read_table("X.txt",sep=" ",header=None)
	X = X.as_matrix()
	Y = np.loadtxt("Y.txt")
	groups = np.loadtxt("groups.txt")
	groups = groups.astype('int')
	beta_true = np.loadtxt("beta.txt")

	epsilon = np.array([1e-4,1e-6,1e-8])
	K = 1024
	f_ce = 10
	w = np.array([np.sqrt(10)])
	num_groups = 1000
	epsilon_g = (1-tau)*w[0]/(tau + (1-tau)*w[0]) * np.ones(num_groups)
	tmp_X = np.dot(np.transpose(X),Y)
	lambda_max = dual_norm(w,groups,tmp_X,num_groups,tau,epsilon_g)
	T = 100
	lambda_sequence = lambda_max*10**(-3.*np.arange(1,T)/(T-1))


    ## experiments 1
    # gap_safe
	# beta,active_variables1 = gap_safe_rule(X,Y,p,epsilon[2],K,f_ce,lambda_sequence[0:73:6],lambda_max,tau,groups,True,1)
	# np.savetxt('gap_safe_screen.txt',active_variables1)
    # # dynamic safe
	# beta,active_variables2 = gap_safe_rule(X,Y,p,epsilon[2],K,f_ce,lambda_sequence[0:73:6],lambda_max,tau,groups,True,2)
	# np.savetxt('dynamic_safe.txt',active_variables2)
    # # dst3
	# beta,active_variables3 = gap_safe_rule(X,Y,p,epsilon[2],K,f_ce,lambda_sequence[0:73:6],lambda_max,tau,groups,True,3)
	# np.savetxt('dst3.txt',active_variables3)
    # # static_safe
	# beta,active_variables5 = gap_safe_rule(X,Y,p,epsilon[2],K,f_ce,lambda_sequence[0:73:6],lambda_max,tau,groups,True,5)
	# np.savetxt('static_safe.txt',active_variables5)
	#
	#
	## experiments 2(I didn't run this experiment because I don't know how to choose radius for gap sequential
	## gap_safe
	# beta,active_variables6 = gap_safe_rule(X,Y,p,epsilon[2],K,f_ce,lambda_sequence[0:99],lambda_max,tau,groups,True,1)
	# np.savetxt('gap_safe_long.txt',active_variables6)
    ## gap_sequential
	# beta,active_variables7 = gap_safe_rule(X,Y,p,epsilon[2],K,f_ce,lambda_sequence[0:99],lambda_max,tau,groups,True,4)
	beta,active_variables7 = gap_safe_rule(X,Y,p,epsilon[2],K,f_ce,np.array([lambda_sequence[50]]),lambda_max,tau,groups,True,1)
	# np.savetxt('gap_sequential_long.txt',active_variables7)
	# np.savetxt('gap_seq_long.txt',active_variables7)

	#
	# ## experiments 3(Also I didn't use the result of gap sequential, I discard the data when plotting in R)
	# time_passed = np.zeros((3,6))
	# for i in xrange(3):
	# 	for j in xrange(6):
	# 		if j==0:
	# 			start_time = time.time()
	# 			beta,active_variables = gap_safe_rule(X,Y,p,epsilon[i],K,f_ce,lambda_sequence[0:10],lambda_max,tau,groups)
	# 			time_passed[i,j] = time.time() - start_time
	# 		else:
	# 			start_time = time.time()
	# 			beta,active_variables = gap_safe_rule(X,Y,p,epsilon[i],K,f_ce,lambda_sequence[0:10],lambda_max,tau,groups,True,j)
	# 			time_passed[i,j] = time.time() - start_time
	# np.savetxt("time_passed.txt",time_passed)


	# print [sum(i!=0) for i in beta]
