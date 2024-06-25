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

    def __init__(self, position):
        # Random distr of fireflies, 6 angles between -pi to pi
        self.position = position
        self.intensity = 0

    def euclid_dist(self, Tf, targetTf):
        d_sum = 0
        for i in range(3):
            d_sum = d_sum + math.pow((targetTf[i,3] - Tf[i,3]), 2)
        return math.sqrt(d_sum)

    def angle_dist(self, Tf, targetTf):
        RMSE_sum = 0
        for i in range(3):
            for j in range(3):
                RMSE_sum = RMSE_sum + math.pow((targetTf[i,j] - Tf[i,j]), 2)
        return math.sqrt(RMSE_sum / 9)

    def compute_I(self, Tf, targetTf, gamma):
        d = self.euclid_dist(Tf, targetTf)
        RMSE = self.angle_dist(Tf, targetTf)

        # self.intensity = 1 / (1 + gamma*d)
        self.intensity = 0.5 / (1 + gamma*d) + 0.5 / (1 + 100*gamma*RMSE)

    def move(self, other, alpha, beta, gamma):
        # """
        
        diff = other.position - self.position
        d = np.linalg.norm(diff) # sqrt(sum(abs((self.__position - better_position))))
        # self.position = self.position + beta*diff + alpha*rand_angles(num_angles)
        self.position = self.position + beta*np.exp(-gamma*(d**2))*diff + alpha*rand_angles(num_angles)
        
        """
        velocity = 0.1 # radians
        randomness_mag = 0.05
        # Best was v = 0.01, r = 0.005 for position only - much better than alpha and beta values
        
        rand = random.uniform(-randomness_mag, randomness_mag)

        # move towards the position at a set velocity (for each angle)
        diff = other.position - self.position
        diff_sign = np.sign(diff)
        diff_sign[abs(diff) <= velocity] = 0.1 # fine grain if distance is less than velocity
        
        self.position = self.position + np.ones(num_angles)*velocity*diff_sign + rand
        """

    def random_walk(self, alpha, fine_grain_alpha=0.1):
        self.position = self.position + rand_angles(num_angles)*alpha*fine_grain_alpha

def rand_angles(n):
    return (np.random.rand(n)-0.5)*pi

def firefly_IK(target_Tf, maxGenerations, n, debug=False, graph=False, alpha=0.05, beta=0.02, gamma=0.08):
    y_out = []

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

    best_ff = [0,0] # [intensity, ff_index]
    Tfs = np.empty(n, dtype=object)
    
    for i in range(n):
        Tfs[i] = f_kine(fireflies[i].position)
        fireflies[i].compute_I(Tfs[i], target_Tf, gamma)
    
    while (t < maxGenerations):
        for i in range(n):
            for j in range(n):
                r = np.sum((fireflies[j].position - fireflies[i].position)**2)
                if (fireflies[i].intensity < fireflies[j].intensity*math.exp(-gamma*r)):
                # if (fireflies[i].intensity < fireflies[j].intensity):
                    fireflies[i].move(fireflies[j], alpha, beta, gamma)
                    Tfs[i] = f_kine(fireflies[i].position)
                    fireflies[i].compute_I(Tfs[i], target_Tf, gamma)

        # Get current best firefly
        best_i = get_best(fireflies)
        if (fireflies[best_i].intensity > best_ff[0]):
            best_ff = [fireflies[best_i].intensity, best_i]

        # Random walk the best firefly
        # fireflies[best_i].random_walk(alpha)
        # Tfs[best_i] = f_kine(fireflies[best_i].position)
        # fireflies[best_i].compute_I(Tfs[best_i], target_Tf, gamma)

        t = t + 1
        
        # Misc.
        if (graph):
            y_out.append(fireflies[best_i].intensity)
        if (debug and t % 4 == 0):
            print(fireflies[best_i].intensity)
            print(fireflies[best_i].position)
            x = [Tf[0,3] for Tf in Tfs]
            y = [Tf[1,3] for Tf in Tfs]
            z = [Tf[2,3] for Tf in Tfs]

            line1.set_xdata(x)
            line1.set_ydata(y)
            line2.set_xdata(x)
            line2.set_ydata(z)

            fig.canvas.draw() 
            fig.canvas.flush_events()

    # Return value
    if (graph):
        return y_out
    else:
        return fireflies[best_ff[1]]

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
    # Tf = np.array([[sin(th6)*(cos(th4)*sin(th1) - sin(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) + cos(th6)*(sin(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - cos(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2))), cos(th6)*(cos(th4)*sin(th1) - sin(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - sin(th6)*(sin(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - cos(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2))), cos(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) + sin(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2)), 115*cos(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - 220*cos(th1)*sin(th2) + 115*sin(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2)) - 220*cos(th1)*cos(th2)*sin(th3) - 220*cos(th1)*cos(th3)*sin(th2)],
    # [sin(th6)*(cos(th1)*cos(th4) + sin(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th6)*(sin(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2))), cos(th6)*(cos(th1)*cos(th4) + sin(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - sin(th6)*(sin(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2))), cos(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - sin(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2)), 220*sin(th1)*sin(th2) + 115*cos(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - 115*sin(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2)) + 220*cos(th2)*sin(th1)*sin(th3) + 220*cos(th3)*sin(th1)*sin(th2)],
    # [cos(th6)*(cos(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*sin(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2))) + sin(th4)*sin(th6)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)),                                                                                                   cos(th6)*sin(th4)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*sin(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2))),                                                         - sin(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*cos(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)),                                                                                220*cos(th2) + 220*cos(th2)*cos(th3) - 220*sin(th2)*sin(th3) - 115*sin(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - 115*cos(th4)*cos(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)) + 155],
    # [                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                     0,                                                                                                                                                                      0,                                                                                                                                                                                                                                                                        1]])

    def DH2tform(th, d, A, al):
        T = np.array([
        [cos(th), -cos(al)*sin(th),  sin(al)*sin(th), A*cos(th)],
        [sin(th),  cos(al)*cos(th), -sin(al)*cos(th), A*sin(th)],
        [    0,          sin(al),          cos(al),         d],
        [    0,                0,                0,         1]
        ])
        return T
    
    l1 = 155
    l2 = 220
    l3 = 45
    l4 = 175
    l5 = 115
    T1 = DH2tform(-th1, l1, 0, pi/2)
    T2 = DH2tform(th2, 0, 0, -pi/2)
    T3 = DH2tform(0, l2, 0, pi/2)
    T4 = DH2tform(th3, 0, 0, -pi/2)
    T5 = DH2tform(-th4, l3+l4, 0, pi/2)
    T6 = DH2tform(pi/2+th5, 0, 0, -pi/2)
    T7 = DH2tform(th6, l5, 0, 0)

    Tf_out = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7

    return Tf_out

"""
def tform2quat(Tf):
        
    qw = 0
    qx = 0
    qy = 0
    qz = 0

    m = Tf[0:2, 0:2]

    # From https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    tr = m[0,0] + m[1,1] + m[2,2]

    if (tr > 0):
        S = sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (m[2,1] - m[1,2]) / S
        qy = (m[0,2] - m[2,0]) / S
        qz = (m[1,0] - m[0,1]) / S
    elif ((m[0,0] > m[1,1])&(m[0,0] > m[2,2])): 
        S = sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        qw = (m[2,1] - m[1,2]) / S
        qx = 0.25 * S
        qy = (m[0,1] + m[1,0]) / S
        qz = (m[0,2] + m[2,0]) / S
    elif (m[1,1] > m[2,2]):
        S = sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        qw = (m[0,2] - m[2,0]) / S
        qx = (m[0,1] + m[1,0]) / S
        qy = 0.25 * S
        qz = (m[1,2] + m[2,1]) / S
    else:
        S = sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        qw = (m[1,0] - m[0,1]) / S
        qx = (m[0,2] + m[2,0]) / S
        qy = (m[1,2] + m[2,1]) / S
        qz = 0.25 * S

    return [qw, qx, qy, qz]
"""

def finetune_FA_IK():
    # TODO: MULTITHREADING

    maxGenerations = 50
    n = 10

    # Search space
    # alpha_s = [0.1, 0.01, 0.001, 0.0001]
    # beta_s = [0.1, 0.01, 0.001, 0.0001]
    # gamma_s = [0.1, 0.01, 0.001, 0.0001]
    # alpha_s = [0.1]
    # beta_s = [0.1]
    # gamma_s = [0.0001, 0.00001, 0.000001, 0.0000001, 0]
    alpha_s = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    beta_s = [0.5]
    gamma_s = [0.00001]

    search_space = list(it.product(alpha_s, beta_s, gamma_s))

    best_pair = []
    best_d = None

    for abg in search_space:
        n = 20
        avg_d = 0

        for _ in range(n):
            alpha = abg[0]
            beta = abg[1]
            gamma = abg[2]

            target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

            sln = firefly_IK(target_Tf, maxGenerations, n, alpha=alpha, beta=beta, gamma=gamma)

            avg_d = avg_d + sln.euclid_dist(f_kine(sln.position), target_Tf)

        avg_d = avg_d / n
        print("-----------------")
        print(abg)
        print(avg_d)
        print("-----------------")
        if (best_d is None or avg_d < best_d):
            best_pair = abg
            best_d = avg_d

    print("================================BEST")
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
    
    for i in range(0,10):
        plt.plot(x, ans[i], '-')
        
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
    print(sln.euclid_dist(f_kine(sln.position), target_Tf))

    print("Transform is: ")
    print(f_kine(sln.position))

    print("Target transform is: ")
    print(target_Tf)

def debug_profile():
    cProfile.run("debug()", 'restats')
    p = pstats.Stats('restats')
    p.sort_stats('cumulative').print_stats("firefly_algorithm.py", 10)

if __name__ == "__main__":
    debug_profile()
    # debug()
    # finetune_FA_IK()
    # graph_FA_IK()
    print("wait")