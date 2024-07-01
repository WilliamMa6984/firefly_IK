# References: pseudocode and calculations from Nizar Rokbani https://core.ac.uk/download/pdf/81582583.pdf
#             reference Python implementation from Agarwal et al. https://www.kaggle.com/code/himanshuagarwal4498/firefly-algorithm/notebook
#             Further reference implementation from Xin-She Yang https://www.researchgate.net/publication/235979455_Nature-Inspired_Metaheuristic_Algorithms

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
    Tf = None

    def __init__(self, position):
        self.position = position
        self.intensity = 0
        self.compute_fkine()

    def euclid_dist(self, targetTf):
        diff = targetTf[0:3,3] - self.Tf[0:3,3]
        euc_d = np.linalg.norm(diff)
        return euc_d

    def angle_dist(self, targetTf):
        R = targetTf[0:3, 0:3] @ np.transpose(self.Tf[0:3, 0:3])
        trace = np.trace(R)

        try:
            theta = math.acos((trace - 1)/2.0)
            return theta
        except: # Edge case: trace returns 3.00...01 - causes acos(~) error
            trace_r = math.floor(trace * 10000) / 10000.0 # Round to nearest 4 decimals
            theta = math.acos((trace_r - 1)/2.0)
            return theta

    def compute_I(self, targetTf, gamma, preemptcond=None):
        angle_mult = 100 # From calc_angle_intensity_mult

        d = self.euclid_dist(targetTf)
        theta = self.angle_dist(targetTf)

        if (preemptcond is not None and d < preemptcond["dist_tol_mm"] and theta < preemptcond["angle_tol_rad"]):
            print("Preempt")
            self.intensity = 1

        self.intensity = 0.5 / (1 + gamma*d) + 0.5 / (1 + angle_mult*gamma*theta)

    def compute_fkine(self):
        self.Tf = f_kine(self.position)

    def move(self, other, alpha, beta, gamma):
        diff = other.position - self.position
        d = np.linalg.norm(diff) # sqrt(sum(abs((self.__position - better_position))))
        # self.position = self.position + beta*diff + alpha*rand_angles(num_angles)
        self.position = self.position + beta*np.exp(-gamma*(d**2))*diff + alpha*rand_angles(num_angles)

    def random_walk(self, alpha):
        self.position = self.position + rand_angles(num_angles)*alpha

def rand_angles(n):
    return (np.random.rand(n)-0.5)*pi*2

def firefly_IK(target_Tf, maxGenerations, n, debug=False, graph=False, alpha0=0.05, beta=0.02, gamma=0.08, preempt=False, preemptcond=None):
    d_out = []
    angle_out = []
    alpha = alpha0

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
                    fireflies[i].move(fireflies[j], alpha, beta, gamma)
                    fireflies[i].compute_fkine()
                    fireflies[i].compute_I(target_Tf, gamma, preemptcond)
                    if (preempt == True and fireflies[i].intensity == 1):
                        return fireflies[i]

        # Get current best firefly
        best_i = get_best(fireflies)
        if (best_ff is None or fireflies[best_i].intensity > best_ff.intensity):
            best_ff = fireflies[best_i]

        # Random walk the best firefly
        fireflies[best_i].random_walk(alpha)
        fireflies[best_i].compute_fkine()
        fireflies[best_i].compute_I(target_Tf, gamma, preemptcond)
        if (preempt == True and fireflies[best_i].intensity == 1):
            return fireflies[best_i]

        t = t + 1
        alpha = alpha_new(alpha, t, maxGenerations)
        
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

def alpha_new(alpha, t, maxGenerations): # NOTE: May result in premature convergence
    # x = maxGenerations / 3.0
    x = 100
    delta = 1 - 0.005**(x/t) # from Yang
    
    return delta*alpha

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

def solve_IK(target_Tf, arg):
    alpha = arg['alpha']
    beta = arg['beta']
    gamma = arg['gamma']

    maxGenerations = arg['maxGenerations']
    n = arg['n']

    dist_tol_mm = 1
    angle_tol_rad = 0.017 # 1 degree

    t = 0
    t_cutoff = 10

    while True:
        sln = firefly_IK(target_Tf, maxGenerations, n, alpha0=alpha, beta=beta, gamma=gamma,
                         preempt=True, preemptcond={"dist_tol_mm": 1, "angle_tol_rad": 0.017})

        # Check solution -> within tolerances
        if (sln.euclid_dist(target_Tf) < dist_tol_mm and sln.angle_dist(target_Tf) < angle_tol_rad):
            print("Completed in " + str(t) + " FA loops")

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

            return sln.position

        if (t > t_cutoff):
            print("Firefly timeout reached, breaking...")

            print("Target transform is: ")
            print(target_Tf)

            return None

        t = t + 1

def finetune_FA_IK():

    maxGenerations = 50
    n = 10

    # Search space
    # alpha_s = [0.1, 0.01, 0.001, 0.0001]
    # beta_s = [0.1, 0.01, 0.001, 0.0001]
    # gamma_s = [0.1, 0.01, 0.001, 0.0001]
    # alpha_s = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    # beta_s = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    # gamma_s = [0.0001, 0.00001, 0.000001, 0.0000001, 0]
    alpha_s = [0.05]
    beta_s = [0.5]
    gamma_s = [0.00001]
    # gamma1_s = [10, 50, 100, 500]

    search_space = list(it.product(alpha_s, beta_s, gamma_s))

    best_pair = [[],[],[]]
    best_d = None
    best_ang_d = None
    best_i = None

    for abg in search_space:
        n = 20
        avg_d = 0
        avg_ang_d = 0
        avg_i = 0

        for _ in range(n):
            alpha = abg[0]
            beta = abg[1]
            gamma = abg[2]
            # gamma1 = abg[3]

            target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

            sln = firefly_IK(target_Tf, maxGenerations, n, alpha0=alpha, beta=beta, gamma=gamma)

            avg_d = avg_d + sln.euclid_dist(target_Tf)
            avg_ang_d = avg_ang_d + sln.angle_dist(target_Tf)
            avg_i = avg_i + sln.intensity

        avg_d = avg_d / n
        avg_ang_d = avg_ang_d / n
        avg_i = avg_i / n
        print("-----------------")
        print(abg)
        print(avg_d)
        print(avg_ang_d)
        print(avg_i)
        print("-----------------")
        if (best_d is None or avg_d < best_d):
            best_pair[0] = abg
            best_d = avg_d
        if (best_ang_d is None or avg_ang_d < best_ang_d):
            best_pair[1] = abg
            best_ang_d = avg_ang_d
        if (best_i is None or avg_i < best_i):
            best_pair[2] = abg
            best_i = avg_i

    print("================================BEST")
    print("Distance, angle, intensity")
    print(best_pair)
    print(best_d)
    print(best_ang_d)
    print(best_i)

def graph_task(arg):
    alpha = arg['alpha']
    beta = arg['beta']
    gamma = arg['gamma']

    maxGenerations = arg['maxGenerations']
    n = arg['n']

    target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

    return firefly_IK(target_Tf, maxGenerations, n, graph=True, alpha0=alpha, beta=beta, gamma=gamma)

def graph_FA_IK(arg):
    maxGenerations = arg['maxGenerations']

    num_times = 30

    import multiprocessing as mp

    x = np.array(list(range(1, maxGenerations+1)))
    
    pool = mp.Pool()
    a = []
    [a.append(arg) for _ in range(num_times)]
    ans = pool.map(graph_task, a)

    fig, ax = plt.subplots(2)
    for i in range(0,num_times):
        ax[0].plot(x, ans[i][0], '-')
        ax[1].plot(x, ans[i][1], '-')
        
    plt.show()
    
    pool.close()

def debug(arg):
    alpha = arg['alpha']
    beta = arg['beta']
    gamma = arg['gamma']

    maxGenerations = arg['maxGenerations']
    n = arg['n']

    target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

    sln = firefly_IK(target_Tf, maxGenerations, n, debug=False, alpha0=alpha, beta=beta, gamma=gamma)

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

def debug_profile(arg):
    cProfile.run("debug(arg)", 'restats')
    p = pstats.Stats('restats')
    p.sort_stats('cumulative').print_stats("firefly_algorithm.py", 10)

def calc_angle_intensity_mult():
    n = 100
    euc_d_avg_outer = 0
    theta_avg_outer = 0
    for _ in range(n):
        target_angles = rand_angles(num_angles)
        target_Tf = f_kine(target_angles)

        n1 = 100
        euc_d_avg = 0
        theta_avg = 0
        for _ in range(n1):
            ff_angles = rand_angles(num_angles)
            ff = Firefly(ff_angles)
            euc_d_avg = euc_d_avg + ff.euclid_dist(target_Tf)
            theta_avg = theta_avg + ff.angle_dist(target_Tf)
        
        euc_d_avg_outer = euc_d_avg_outer + euc_d_avg / n1
        theta_avg_outer = theta_avg_outer + theta_avg / n1
        
    euc_d_avg_outer = euc_d_avg_outer / n
    theta_avg_outer = theta_avg_outer / n

    print("Euclidean distance vs theta")
    print(euc_d_avg_outer)
    print(theta_avg_outer)

    print("Angle Mult")
    print(euc_d_avg_outer / theta_avg_outer)

if __name__ == "__main__":
    arg = {
        'alpha': 0.05,
        'beta': 0.5,
        'gamma': 0.00001,
        'maxGenerations': 300,
        'n': 10
    }

    # calc_angle_intensity_mult()

    # debug_profile(arg)
    # debug(arg)
    # finetune_FA_IK()
    # graph_FA_IK(arg)


    from time import perf_counter
    start = perf_counter()


    # No solution:
    target_Tf = [[ 5.90672098e-01, -7.80534494e-01, -2.04627410e-01, -2.12052914e+01],
    [ 7.94408753e-01,  6.06979364e-01, -2.21536601e-02, -2.50164149e+00],
    [ 1.41496311e-01, -1.49472256e-01,  9.78589208e-01,  2.70310590e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

    # target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))
    solve_IK(target_Tf, arg)

    end = perf_counter()
    print("Completed FA-IK in " + str(end-start) + "s")

    print("wait")