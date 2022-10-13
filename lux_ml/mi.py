"""
	Mutual Information Estimation 

	This set of functions are used for calculating mutual information (MI)
	in various contexts using the (KSG) estimator by Kraskov et. al.,
	(https://arxiv.org/abs/cond-mat/0305641). Some of this code is copied from 
	NPEET, (https://github.com/gregversteeg/NPEET), such as the basic mi 
	function, the avgdigamma function and the joint_space function,
	(originally called zip2).
	
	Main Authors:
		Nicholas Carrara [nmcarrara@ucdavis.edu]
		University of California at Davis
		Davis, CA 95616
		Jesse Ernst		 [jae@albany.edu]
		University at Albany
		Albany, NY 12224
"""
#-----------------------------------------------------------------------------
#	Required packages
#-----------------------------------------------------------------------------
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.spatial as ss
import numpy.random as nr
from math import log
import random
import itertools
from scipy.special import digamma, gamma
from math import log2
from typing import List
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi
#-----------------------------------------------------------------------------
def mi(
	x: List[List[float]], 
	y: List[List[float]], 
	z: List[List[float]] = None,
	k: int = 1, 
	base: float = 2
) -> float:
	"""
	Mutual information of two sets of continuous variables, x and y.
	x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
	if x is a one-dimensional scalar and we have four samples.
	"""
	assert len(x) == len(y), "Arrays should have same length"
	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
	x, y = np.asarray(x), np.asarray(y)
	x = add_noise(x)
	y = add_noise(y)
	points = [x, y]
	if z is not None:
		points.append(z)
	points = np.hstack(points)
	# Find nearest neighbors in joint space, p=inf means max-norm
	tree = ss.cKDTree(points)
	dvec = query_neighbors(tree, points, k)
	if z is None:
		a, b, c, d = avgdigamma2(x, dvec), avgdigamma2(y, dvec), digamma(k), digamma(len(x))
	else:
		xz = np.c_[x, z]
		yz = np.c_[y, z]
		a, b, c, d = avgdigamma2(xz, dvec), avgdigamma2(yz, dvec), avgdigamma(z, dvec), digamma(k)
	return (-a - b + c + d) / log(base)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_binary
#-----------------------------------------------------------------------------
def mi_binary(
	signal: List[List[float]], 
	background: List[List[float]], 
	k: int = 1, 
	base: float = 2
) -> float:
	"""
	Mutual information between a set of continuous variables x, and a binary
	category y.  signal should be a vector of vectors, e.g. 
	signal = [[1.3],[2.4],...], as well as background, e.g. 
	background = [[1.0],[-1.0],[1.0]...].
	"""
	if (len(signal) == 0) or (len(background) == 0):
		return 0.0
	assert len(signal[0]) == len(background[0]), "Lists should have same  \
	                                              number of variables!"
# 	print("Number of signal events:     %s" % len(signal))
# 	print("Number of background events: %s" % len(background))
	N = len(signal) + len(background)
	smallest = np.nextafter(0,1)  

	signal, background = add_shake(signal, background)
	sig_tree = ss.cKDTree(signal)
	back_tree = ss.cKDTree(background)

	sig_dvec = [max(sig_tree.query(point,k+1,p=float('inf'))[0][k],smallest)
	            for point in signal]
	back_dvec = [max(back_tree.query(point,k+1,p=float('inf'))[0][k],smallest)
	            for point in background]

	psi_s = avgdigamma(signal, back_tree, sig_dvec, k)/N
	psi_b  = avgdigamma(background, sig_tree, back_dvec, k)/N
	psi_theta = avgdigamma_disc(len(signal),len(background))
	psi_k = digamma(k)
	psi_N = digamma(N)
	mi = (-psi_s - psi_b - psi_theta + psi_k + psi_N)

	if (mi) <= 0:
		return 0
	return (mi) / log(base)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_multi
#-----------------------------------------------------------------------------
def mi_multi(
	multi: List[List[List[float]]], 
	k: int = 1, 
	base: float = 2
) -> float:
	"""
	Mutual information between a set of continuous variables x, and a multi-class
	category y.  multi should be a vector of vectors of vectors, e.g. 
	multi = [[[1.3],[2.4],...]].
	"""
	# check that there are at least two non-empty categories
	n_theta = len(multi)
	N = 0
	num_non_empty = 0
	for n in range(n_theta):
		if len(multi[n]) != 0:
			num_non_empty += 1
		N += len(multi[n])
	if (num_non_empty <= 1):
		return 0.0

	multi_nums = [len(multi[n]) for n in range(len(multi))]
	
	smallest = np.nextafter(0,1)  

	multi = add_shake_multi(multi)
	trees = [ss.cKDTree(multi[n]) for n in range(len(multi))]

	dvecs = [[max(trees[n].query(point,k+1,p=float('inf'))[0][k],smallest)
	            for point in multi[n]] for n in range(len(multi))]

	psi_x = sum([avgdigamma_multi(multi[n],trees,dvecs[n],k,n)/N for n in range(len(multi))])
	psi_theta = avgdigamma_disc_multi(multi_nums)
	psi_k = digamma(k)
	psi_N = digamma(N)
	mi = (-psi_x - psi_theta + psi_k + psi_N)

	if (mi) <= 0:
		return 0
	return (mi) / log(base)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_binary_discrete
#-----------------------------------------------------------------------------
def mi_binary_discrete(
	signal: List[List[float]], 
	background: List[List[float]], 
	disc_list: List[int] = [], 
	k: int = 1, 
	base: float = 2, 
) -> float:
	""" 
	Mutual information between a set of mixed variables x, and a binary
	category y.  X should be a vector of vectors, e.g. x = [[1.3],[2.4],...]
	as well as y, e.g. y = [[1.0],[-1.0],[1.0]...].  Disc_list should contain
	the column numbers of X that are the discrete variables, e.g. 
	disc_list = [1,2,4...].
	"""
	assert len(signal[0]) == len(background[0]), "Lists should have same  \
	                                              number of variables!"
	assert k <= len(signal) - 1, "Set k smaller than num. samples - 1"

	if disc_list == []:
		return mi_binary(signal,background,k,base)
	n_s = len(signal)
	n_b = len(background)
	N = n_s + n_b
	avg_mi = 0
	disc_mi = 0

	unique_disc = []

	s_i = []		
	b_i = []
	for i in range(len(signal)):
		temp_disc = [signal[i][j] for j in range(len(signal[0]))
			         if j in disc_list]
		if temp_disc not in unique_disc:
			unique_disc.append(temp_disc)
			s_i.append([i])
			b_i.append([])
		else:
			for j in range(len(unique_disc)):
				if temp_disc == unique_disc[j]:
					s_i[j].append(i)
	for i in range(len(background)):
		temp_disc = [background[i][j] for j in range(len(background[0]))
			         if j in disc_list]
		if temp_disc not in unique_disc:
			unique_disc.append(temp_disc)
			b_i.append([i])
			s_i.append([])
		else:
			for j in range(len(unique_disc)):
				if temp_disc == unique_disc[j]:
					b_i[j].append(i)	

	n_s_i = [len(s_i[j]) for j in range(len(s_i))]
	n_b_i = [len(b_i[j]) for j in range(len(b_i))]
	n_i = [n_s_i[j] + n_b_i[j] for j in range(len(s_i))]	

	disc_mi_s = 0
	disc_mi_b = 0
	for i in range(len(n_s_i)):
		if (n_s_i[i] != 0):
			p_s_i = (float(n_s_i[i])*N)/(n_s*(n_i[i]))
			disc_mi_s += (n_s_i[i]/float(N))*log2(p_s_i)
	for i in range(len(n_b_i)):
		if (n_b_i[i] != 0):
			p_b_i = (float(n_b_i[i])*N)/(n_b*(n_i[i]))
			disc_mi_b += (n_b_i[i]/float(N))*log2(p_b_i)
	disc_mi = disc_mi_s + disc_mi_b

	mi = 0.0
	for l in range(len(unique_disc)):
		temp_sig = [signal[s_i[l][i]] for i in range(len(s_i[l]))]
		temp_back = [background[b_i[l][i]] for i in range(len(b_i[l]))]
		mi += mi_binary(temp_sig, temp_back, k, base) * (n_i[l]/sum(n_i))
	return (disc_mi + mi)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_binary_weights
#-----------------------------------------------------------------------------
def mi_binary_weights(
	signal: List[List[float]], 
	background: List[List[float]], 
	sig_weights: List[float],
	back_weights: List[float], 
	k: int = 1, 
	base: float = 2
) -> float:
	""" 
	Mutual information between a set of mixed variables x, and a binary
	category y with weights.  X should be a vector of vectors, e.g.
	x = [[1.3],[4.5],...] as well as y, e.g. y = [[1.0],[-1.0],[1.0]...].
	"""
	assert len(signal[0]) == len(background[0]), "Lists should have same  \
	                                              number of variables!"
	assert k <= len(signal) - 1, "Set k smaller than num. samples - 1"

	sig_max, back_max = max(sig_weights), max(back_weights)
	sig_weights = [sig_weights[i]/sig_max for i in range(len(sig_weights))]
	back_weights = [back_weights[i]/back_max for i in range(len(back_weights))]
	fraction = sum(sig_weights)/sum(back_weights)

	if fraction >= 1.0:
		weight_s = 1.0/fraction
		weight_b = 1.0
	else:
		weight_s = 1.0
		weight_b = fraction

	new_signal = []
	new_background = []
	for i in range(len(signal)):
		if weight_s * sig_weights[i] >= np.random.uniform(0,1,1)[0]:
			new_signal.append(signal[i])
	for i in range(len(background)):
		if weight_b * back_weights[i] >= np.random.uniform(0,1,1)[0]:
			new_background.append(background[i])

	return mi_binary(new_signal, new_background, k, base), len(new_signal), len(new_background)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	mi_binary_discrete_weights
#-----------------------------------------------------------------------------
def mi_binary_discrete_weights(
	signal: List[List[float]], 
	background: List[List[float]], 
	disc_list: List[int], 
	sig_weights: List[float], 
	back_weights: List[float], 
	k: int = 1, 
	base: float = 2
) -> float:
	""" 
	Mutual information between a set of mixed variables x, and a binary
	category y with weights.  Signal should be a vector of vectors, e.g.
	signal = [[1.3],[4.5],...] as well as background, e.g. 
	background = [[1.0],[-1.0],[1.0]...].  Disc_list should contain the column 
	numbers of signal and background that are the discrete variables, e.g. 
	disc_list = [1,2,4...].
	"""
	assert len(signal[0]) == len(background[0]), "Lists should have same  \
	                                              number of variables!"
	assert k <= len(signal) - 1, "Set k smaller than num. samples - 1"

	sig_max, back_max = max(sig_weights), max(back_weights)
	sig_weights = [sig_weights[i]/sig_max for i in range(len(sig_weights))]
	back_weights = [back_weights[i]/back_max for i in range(len(back_weights))]
	fraction = sum(sig_weights)/sum(back_weights)

	if fraction >= 1.0:
		weight_s = 1.0/fraction
		weight_b = 1.0
	else:
		weight_s = 1.0
		weight_b = fraction

	new_signal = []
	new_background = []
	for i in range(len(signal)):
		if weight_s * sig_weights[i] >= np.random.uniform(0,1,1)[0]:
			new_signal.append(signal[i])
	for i in range(len(background)):
		if weight_b * back_weights[i] >= np.random.uniform(0,1,1)[0]:
			new_background.append(background[i])

	return mi_binary_discrete(new_signal, new_background, disc_list, k, base)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	joint_space
#-----------------------------------------------------------------------------
def joint_space(*args):
    """
	zip2(x, y) takes the lists of vectors and makes it a list of vectors 
	in a joint space. E.g. zip2([[1], [2], [3]], [[4], [5], [6]])
	= [[1, 4], [2, 5], [3, 6]].
	"""
    return [sum(sublist, []) for sublist in zip(*args)]
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	avgdigamma
#-----------------------------------------------------------------------------
def avgdigamma(
	points: List[List[float]], 
	tree: ss.cKDTree,
	dvec: List[float], 
	k: int
) -> float:
	"""
	This part finds number of neighbors in some radius in the marginal space
	returns expectation value of <psi(nx)>.
	"""
	smallest = np.nextafter(0,1)
	N = len(points)
	avg = 0.
	for i in range(N):
		dist = dvec[i]
		num_points = len(tree.query_ball_point(points[i], 
							dist - smallest, 
							p=float('inf')))
		if num_points > 0:
			avg += digamma(k + num_points)
		else:
			avg += digamma(k)
	return avg
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	avgdigamma_multi
#-----------------------------------------------------------------------------
def avgdigamma_multi(
	points: List[List[float]], 
	trees: List[ss.cKDTree],
	dvec: List[float], 
	k: int,
	n: int
) -> float:
	"""
	This part finds number of neighbors in some radius in the marginal space
	returns expectation value of <psi(nx)>.
	"""
	smallest = np.nextafter(0,1)
	N = len(points)
	avg = 0.
	for i in range(N):
		dist = dvec[i]
		num_points = 0
		for l in range(len(trees)):
			if l != n:
				num_points += len(trees[l].query_ball_point(points[i], 
									dist - smallest, 
									p=float('inf')))
		if num_points > 0:
			avg += digamma(k + num_points)
		else:
			avg += digamma(k)
	return avg
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	avgdigamma_disc
#-----------------------------------------------------------------------------
def avgdigamma_disc(
	num_sig: int, 
	num_back: int
) -> float:
	"""
	This is used to evaluate the <digamma(y)> when y is a binary
	category variable. 
	"""
	N = num_sig + num_back
	term_1 = 0
	term_2 = 0
	if num_sig > 0:
		term_1 = (num_sig)/N * digamma(num_sig)
	if num_back > 0:
		term_2 = (num_back)/N * digamma(num_back)
	avg = term_1 + term_2 
	return avg
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	avgdigamma_disc_multi
#-----------------------------------------------------------------------------
def avgdigamma_disc_multi(
	multi: List[int]
) -> float:
	"""
	This is used to evaluate the <digamma(y)> when y is a multi-class
	category variable. 
	"""
	N = sum(multi)
	avg = 0
	for n in range(len(multi)):
		if multi[n] > 0:
			avg += (multi[n])/N * digamma(multi[n])
	return avg
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	add_shake
#-----------------------------------------------------------------------------
def add_shake(
	signal: List[List[float]], 
	background: List[List[float]]
) -> (List[List[float]], List[List[float]]):
	"""
	For each variable, make the shaking amplitude some orders of mag smaller
	than mean value of the variable over all signal and background events.  
	However, don't let it be smaller than some absolute smallest value.  The 
	former prevents shaking from being so small relative to the variable that 
	the ckdtree has precision problems.  The latter prevents having no shaking 
	amplitude in cases where the variables are all very close to zero, 
	or identically zero.
	"""
	for i_var in range(len(signal[0])):
		i_var_mean = np.mean([row[i_var] for row in signal] + 
								[row[i_var] for row in background])
		intens = max(i_var_mean*1e-7, 1e-7)
		for row in range(len(signal)): 
			signal[row][i_var] += intens * (nr.rand() - 0.5)
		for row in range(len(background)): 
			background[row][i_var] += intens * (nr.rand() - 0.5)
	return signal, background
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	add_shake_multi
#-----------------------------------------------------------------------------
def add_shake_multi(
	multi: List[List[List[float]]], 
) -> (List[List[List[float]]]):
	"""
	For each variable, make the shaking amplitude some orders of mag smaller
	than mean value of the variable over all multi and background events.  
	However, don't let it be smaller than some absolute smallest value.  The 
	former prevents shaking from being so small relative to the variable that 
	the ckdtree has precision problems.  The latter prevents having no shaking 
	amplitude in cases where the variables are all very close to zero, 
	or identically zero.
	"""
	for i_var in range(len(multi[0][0])):
		temp_vals = []
		for n in range(len(multi)):
			for row in multi[n]:
				temp_vals.append(row[i_var])
		i_var_mean = np.mean(temp_vals)
		intens = max(i_var_mean*1e-7, 1e-7)
		for n in range(len(multi)):
			for row in range(len(multi[n])): 
				multi[n][row][i_var] += intens * (nr.rand() - 0.5)
	return multi
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	prob_error
#-----------------------------------------------------------------------------
def prob_error(
	signal: List[List[float]], 
	background: List[List[float]], 
	k: int = 1, 
) -> float:
	"""
	Computes the probability of error for a given sample.
	"""
	if (len(signal) == 0) or (len(background) == 0):
		return 0.0
	assert len(signal[0]) == len(background[0]), "Lists should have same  \
	                                              number of variables!"
	N = len(signal) + len(background)
	smallest = np.nextafter(0,1)  

	signal, background = add_shake(signal, background)
	sig_tree = ss.cKDTree(signal)
	back_tree = ss.cKDTree(background)

	sig_dvec = [max(sig_tree.query(point,k+1,p=float('2'))[0][k],smallest)
	            for point in signal]
	back_dvec = [max(back_tree.query(point,k+1,p=float('2'))[0][k],smallest)
	            for point in background]

	error = find_ratios(signal, back_tree, sig_dvec, k) / N
	error += find_ratios(background, sig_tree, back_dvec, k) / N

	return error
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	find_ratios
#-----------------------------------------------------------------------------
def find_ratios(
	points: List[List[float]], 
	tree: ss.cKDTree,
	dvec: List[float], 
	k: int
) -> float:
	"""
	This part finds number of neighbors in some radius in the marginal space
	returns the smaller of either k/n or #found/n.
	"""
	N = len(points)
	avg = 0.
	for i in range(N):
		dist = dvec[i]
		num_points = len(tree.query_ball_point(points[i], 
							dist, 
							p=float('2')))
		if num_points > k:
			avg += k/(k+num_points-1)
		else:
			avg += num_points/(k+num_points-1)
	return avg
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	fano
#-----------------------------------------------------------------------------
def fano_plot(
	signal: List[List[float]], 
	background: List[List[float]], 
	k: int = 1, 
) -> float:
	"""
	Plots the H(theta|x) versus Fano curve.
	"""
	n_s = len(signal)
	n_b = len(background)

	mis = []
	errors = []
	num_sig_events = len(signal)/10
	num_back_events = len(background)/10
	for j in range(10):
		temp_sig = [signal[i] for i in range(int(j*num_sig_events),int((j+1)*num_sig_events))]
		temp_back = [background[i] for i in range(int(j*num_back_events),int((j+1)*num_back_events))]
		mis.append(mi_binary(temp_sig,temp_back,k))
		errors.append(prob_error(temp_sig,temp_back,k=10))
	mi = np.mean(mis)
	mi_std = np.std(mis)/np.sqrt(10)
	error = np.mean(errors)
	error_std = np.std(errors)/np.sqrt(10)

	p_s = n_s/(n_s + n_b)
	p_b = 1 - p_s
	entropy_theta = -p_s*log2(p_s) - p_b*log2(p_b)
	cond_ent = entropy_theta - mi

	error_curve = [0.005*i for i in range(101)]
	fano_curve = [0.0]
	for i in range(1,len(error_curve)):
		fano_curve.append((-error_curve[i]*log2(error_curve[i])-(1-error_curve[i])*log2(1-error_curve[i])))
	bayes_curve = [0.0,1.0]
	bayes_error = [0.0,0.5]

	fig, axs = plt.subplots(figsize=(10,10))
	axs.scatter(cond_ent,error,color='r')
	axs.errorbar(cond_ent,error,yerr=error_std,xerr=mi_std,color='r',capsize=2)
	axs.plot(fano_curve,error_curve,color='k',label="Fano")
	axs.plot(bayes_curve,bayes_error,color='g',linestyle='--',label="Bayes")
	axs.set_xlabel(r"$H(\theta|\mathbf{X})$")
	axs.set_ylabel(r"$P(E)$")
	plt.title(r"$P(E)$ vs. $H(\theta|\mathbf{X})$")
	plt.legend()
	plt.show()
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#	fano
#-----------------------------------------------------------------------------
def fano(
	signal: List[List[float]], 
	background: List[List[float]], 
	k: int = 1, 
) -> float:
	"""
	Computes the fano bound for a given sample.
	"""
	if (len(signal) == 0) or (len(background) == 0):
		return 0.0
	assert len(signal[0]) == len(background[0]), "Lists should have same  \
	                                              number of variables!"
	N = len(signal) + len(background)
	smallest = np.nextafter(0,1)  

	signal, background = add_shake(signal, background)
	sig_tree = ss.cKDTree(signal)
	back_tree = ss.cKDTree(background)

	sig_dvec = [max(sig_tree.query(point,k+1,p=float('2'))[0][k],smallest)
	            for point in signal]
	back_dvec = [max(back_tree.query(point,k+1,p=float('2'))[0][k],smallest)
	            for point in background]

	error = find_ratios(signal, back_tree, sig_dvec, k) / N
	error += find_ratios(background, sig_tree, back_dvec, k) / N

	return -error * log2(error) - (1-error)*log2(1-error)
#-----------------------------------------------------------------------------

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]
	
def avgdigamma2(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    n_elements = len(points)
    tree = ss.cKDTree(points)
    avg = 0.
    dvec = dvec - 1e-15
    for point, dist in zip(points, dvec):
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(point, dist, p=float('inf')))
        if num_points > 0:
        	avg += digamma(num_points) / n_elements
    return avg	

if __name__ == "__main__":
	print("none")
	N = 10000

	# one Gaussian variable separated by delta_mi = 2 for the third variable.  
	# all others are uniform noise
	x_1 = [[np.random.normal(6,1,1)[0]] for i in range(N)]

	x_2 = [[np.random.normal(0,1,1)[0]] for i in range(N)]
	
	x_3 = [[np.random.normal(-6,1,1)[0]] for i in range(N)]

	x_4 = [[np.random.normal(-20,1,1)[0]] for i in range(N)]

	x = [x_1,x_2,x_3,x_4]
	print(mi_multi(x))
	print(mi_binary(x_1,x_2))
