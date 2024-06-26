# References: pseudocode and calculations from Rokbani et al. https://core.ac.uk/download/pdf/81582583.pdf
#             reference Python implementation from Agarwal et al. https://www.kaggle.com/code/himanshuagarwal4498/firefly-algorithm/notebook

import math
import random
import itertools as it
from math import cos, sin, pi

import numpy as np
import matplotlib.pyplot as plt

# profiling
import pstats
import cProfile

num_angles = 6

class Firefly():
    position = []
    intensity = 0
    ghost = np.zeros(num_angles) # ghost of best firefly to follow
    Tf = None

    def __init__(self, position):
        # Random distr of fireflies, 6 angles between -pi to pi
        self.position = position
        self.intensity = 0
        self.compute_fkine()

    def euclid_dist(self, targetTf):
        d_sum = 0
        for i in range(3):
            d_sum = d_sum + math.pow((targetTf[i,3] - self.Tf[i,3]), 2)
        return math.sqrt(d_sum)

    def angle_dist(self, targetTf):
        RMSE_sum = 0
        for i in range(3):
            for j in range(3):
                RMSE_sum = RMSE_sum + math.pow((targetTf[i,j] - self.Tf[i,j]), 2)
        return math.sqrt(RMSE_sum / 9)

    def compute_I(self, targetTf, gamma):
        d = self.euclid_dist(targetTf)
        RMSE = self.angle_dist(targetTf)

        self.intensity = 0.5 / (1 + gamma*d) + 0.5 / (1 + 300*gamma*RMSE)

    def compute_fkine(self):
        self.Tf = f_kine(self.position)

    def move(self, other, alpha, beta, gamma):
        diff = other.position - self.position
        d = np.linalg.norm(diff) # sqrt(sum(abs((self.__position - better_position))))
        # self.position = self.position + beta*diff + alpha*rand_angles(num_angles)
        self.position = self.position + beta*np.exp(-gamma*(d**2))*diff + alpha*rand_angles(num_angles)

    # def random_walk(self, alpha):
    #     self.position = self.position + rand_angles(num_angles)*alpha

def rand_angles(n):
    return (np.random.rand(n)-0.5)*pi

def firefly_IK(target_Tf, maxGenerations, n, debug=False, graph=False, alpha=0.05, beta=0.02, gamma=0.08):
    d_out = []
    angle_out = []
    alpha_inner = alpha

    # Generate initial population
    fireflies = []
    for _ in range(n):
        position = rand_angles(num_angles)
        fireflies.append(Firefly(position))
    
    t = 0

    fig = None
    line1 = None
    line2 = None
    if (debug):
        plt.ion() 
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax.set_xlim([-1000, 1000])
        ax.set_ylim([-1000, 1000])
        line1, = ax.plot([0], [0], 'o')

        ax2.set_xlim([-1000, 1000])
        ax2.set_ylim([-1000, 1000])
        line2, = ax2.plot([0], [0], 'o')

    best_ff = None
    
    for i in range(n):
        fireflies[i].compute_I(target_Tf, gamma)
    
    while (t < maxGenerations):
        for i in range(n):
            for j in range(n): # nlog(n) loop
                r = np.sum((fireflies[j].position - fireflies[i].position)**2)
                if (fireflies[i].intensity < fireflies[j].intensity*math.exp(-gamma*r)):
                    fireflies[i].move(fireflies[j], alpha_inner, beta, gamma)
            fireflies[i].compute_fkine()
            fireflies[i].compute_I(target_Tf, gamma)

        # Get current best firefly
        best_i = get_best(fireflies)
        if (best_ff is None or fireflies[best_i].intensity > best_ff.intensity):
            best_ff = fireflies[best_i]

        # Random walk the best firefly
        # fireflies[best_i].random_walk(alpha_inner)
        # fireflies[best_i].compute_fkine()
        # fireflies[best_i].compute_I(target_Tf, gamma)

        t = t + 1
        alpha_inner = alpha_new(alpha_inner, maxGenerations) # NOTE: May result in premature convergence
        
        # Misc.
        if (graph):
            d_out.append(fireflies[best_i].euclid_dist(target_Tf))
            angle_out.append(fireflies[best_i].angle_dist(target_Tf))
        if (debug and t % 4 == 0):
            print(fireflies[best_i].intensity)
            print(fireflies[best_i].position)
            x = [ff.Tf[0,3] for ff in fireflies]
            y = [ff.Tf[1,3] for ff in fireflies]
            z = [ff.Tf[2,3] for ff in fireflies]

            line1.set_xdata(x)
            line1.set_ydata(y)
            line2.set_xdata(x)
            line2.set_ydata(z)

            fig.canvas.draw() 
            fig.canvas.flush_events()

    # Return value
    if (graph):
        return [d_out, angle_out]
    else:
        return best_ff

def alpha_new(alpha, n):
    delta = 1 - (0.005/0.9)**(1/n)
    return (1-delta)*alpha

def get_best(fireflies):
    intensities = np.array([ff.intensity for ff in fireflies])
    return np.argmax(intensities)

def f_kine(angles):
    th1 = angles[0]
    th2 = angles[1]
    th3 = angles[2]
    th4 = angles[3]
    th5 = angles[4]
    th6 = angles[5]

    # From MATLAB script
    Tf = np.array([[sin(th6)*(cos(th4)*sin(th1) - sin(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) + cos(th6)*(sin(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - cos(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2))), cos(th6)*(cos(th4)*sin(th1) - sin(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - sin(th6)*(sin(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - cos(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2))), cos(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) + sin(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2)), 115*cos(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - 220*cos(th1)*sin(th2) + 115*sin(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2)) - 220*cos(th1)*cos(th2)*sin(th3) - 220*cos(th1)*cos(th3)*sin(th2)],
    [sin(th6)*(cos(th1)*cos(th4) + sin(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th6)*(sin(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2))), cos(th6)*(cos(th1)*cos(th4) + sin(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - sin(th6)*(sin(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2))), cos(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - sin(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2)), 220*sin(th1)*sin(th2) + 115*cos(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - 115*sin(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2)) + 220*cos(th2)*sin(th1)*sin(th3) + 220*cos(th3)*sin(th1)*sin(th2)],
    [cos(th6)*(cos(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*sin(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2))) + sin(th4)*sin(th6)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)),                                                                                                   cos(th6)*sin(th4)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*sin(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2))),                                                         - sin(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*cos(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)),                                                                                220*cos(th2) + 220*cos(th2)*cos(th3) - 220*sin(th2)*sin(th3) - 115*sin(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - 115*cos(th4)*cos(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)) + 155],
    [                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                     0,                                                                                                                                                                      0,                                                                                                                                                                                                                                                                        1]])
    return Tf

    # def DH2tform(th, d, A, al):
    #     T = np.array([
    #     [cos(th), -cos(al)*sin(th),  sin(al)*sin(th), A*cos(th)],
    #     [sin(th),  cos(al)*cos(th), -sin(al)*cos(th), A*sin(th)],
    #     [    0,          sin(al),          cos(al),         d],
    #     [    0,                0,                0,         1]
    #     ])
    #     return T
    
    # l1 = 155
    # l2 = 220
    # l3 = 45
    # l4 = 175
    # l5 = 115
    # T1 = DH2tform(-th1, l1, 0, pi/2)
    # T2 = DH2tform(th2, 0, 0, -pi/2)
    # T3 = DH2tform(0, l2, 0, pi/2)
    # T4 = DH2tform(th3, 0, 0, -pi/2)
    # T5 = DH2tform(-th4, l3+l4, 0, pi/2)
    # T6 = DH2tform(pi/2+th5, 0, 0, -pi/2)
    # T7 = DH2tform(th6, l5, 0, 0)

    # Tf_out = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7

    # return Tf_out

def finetune_FA_IK():

    maxGenerations = 50
    n = 10

    # Search space
    alpha_s = [0.1, 0.01, 0.001, 0.0001]
    beta_s = [0.1, 0.01, 0.001, 0.0001]
    gamma_s = [0.1, 0.01, 0.001, 0.0001]
    # alpha_s = [0.1]
    # beta_s = [0.1]
    # gamma_s = [0.0001, 0.00001, 0.000001, 0.0000001, 0]
    # alpha_s = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    # beta_s = [0.5]
    # gamma_s = [0.00001]

    search_space = list(it.product(alpha_s, beta_s, gamma_s))

    best_pair = [[],[]]
    best_d = None
    best_ang_d = None

    for abg in search_space:
        n = 20
        avg_d = 0
        avg_ang_d = 0

        for _ in range(n):
            alpha = abg[0]
            beta = abg[1]
            gamma = abg[2]

            target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

            sln = firefly_IK(target_Tf, maxGenerations, n, alpha=alpha, beta=beta, gamma=gamma)

            avg_d = avg_d + sln.euclid_dist(target_Tf)
            avg_ang_d = avg_ang_d + sln.angle_dist(target_Tf)

        avg_d = avg_d / n
        avg_ang_d = avg_ang_d / n
        print("-----------------")
        print(abg)
        print(avg_d)
        print(avg_ang_d)
        print("-----------------")
        if (best_d is None or avg_d < best_d):
            best_pair[0] = abg
            best_d = avg_d
        if (best_ang_d is None or avg_ang_d < best_ang_d):
            best_pair[1] = abg
            best_ang_d = avg_ang_d

    print("================================BEST")
    print("Distance, angle")
    print(best_pair)
    print(best_d)


def graph_task(maxGenerations):
    alpha = 0.05
    beta = 0.5
    gamma = 0.00001 # multiplier on intensity

    n = 20

    target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

    return firefly_IK(target_Tf, maxGenerations, n, graph=True, alpha=alpha, beta=beta, gamma=gamma)

def graph_FA_IK():

    maxGenerations = 500

    import multiprocessing as mp

    x = np.array(list(range(1, maxGenerations+1)))
    
    pool = mp.Pool()
    ans = pool.map(graph_task, np.ones(10)*maxGenerations)

    fig, ax = plt.subplots(2)
    for i in range(0,10):
        ax[0].plot(x, ans[i][0], '-')
        ax[1].plot(x, ans[i][1], '-')
        
    plt.show()
    
    pool.close()


def debug():
    alpha = 0.05
    beta = 0.5
    gamma = 0.00001 # multiplier on intensity

    maxGenerations = 500
    n = 20

    target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

    sln = firefly_IK(target_Tf, maxGenerations, n, debug=False, alpha=alpha, beta=beta, gamma=gamma)

    print("Solution to IK is: ")
    print(sln.position)

    print("Euclid distance is: ")
    print(sln.euclid_dist(target_Tf))
    print("Angle distance is: ")
    print(sln.angle_dist(target_Tf))

    print("Transform is: ")
    print(f_kine(sln.position))

    print("Target transform is: ")
    print(target_Tf)

def debug_profile():
    cProfile.run("debug()", 'restats')
    p = pstats.Stats('restats')
    p.sort_stats('cumulative').print_stats("firefly_algorithm.py", 10)

if __name__ == "__main__":
    # debug_profile()
    # debug()
    finetune_FA_IK()
    # graph_FA_IK()
    print("wait")