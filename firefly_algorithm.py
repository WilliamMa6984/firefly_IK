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

    d_toggle = 1
    angle_toggle = 1

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
            trace_r = np.fix(trace) # Round to nearest whole, towards 0
            theta = math.acos((trace_r - 1)/2.0)
            return theta

    def compute_I(self, targetTf, gamma, preemptcond=None):
        d = self.euclid_dist(targetTf)
        theta = self.angle_dist(targetTf)
        """
        if (preemptcond is not None):
            if (d < preemptcond["dist_tol_mm"] and theta < preemptcond["angle_tol_rad"]):
                self.intensity = 1
                return
            
            if (d < preemptcond["dist_tol_mm"]):
                self.d_toggle = 0
            else:
                self.d_toggle = 1
            
            if (theta < preemptcond["angle_tol_rad"]):
                self.angle_toggle = 0
            else:
                self.angle_toggle = 1

        # ===================== USER TO MODIFY =====================
        # angle_mult = 100.0 # default
        
        # # Normalisation (range of 0-100)
        d_max = 710.0
        theta_max = pi
        d_min = 0
        theta_min = 0

        # if (preemptcond is not None):
        # d_min = preemptcond["dist_tol_mm"]
        # theta_min = preemptcond["angle_tol_rad"]

        d_norm = (d - d_min) / (d_max - d_min) * 1500
        theta_norm = (abs(theta) - theta_min) / (theta_max - theta_min) * 1500
        # ===================== END USER TO MODIFY =====================
        
        # self.intensity = 0.5 / (1 + gamma*d_norm) + 0.5 / (1 + gamma*theta_norm)
        self.intensity = 0.5 / (1 + self.d_toggle*gamma*d_norm) + 0.5 / (1 + self.angle_toggle*gamma*theta_norm)
        """

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

def firefly_IK(target_Tf, maxGenerations, n, debug=False, graph=False, alpha0=0.05, beta=0.02, gamma=0.08, preemptcond=None):
    d_out = []
    angle_out = []
    alpha = alpha0

    # Generate initial population
    fireflies = []
    for i in range(n):
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
                    if (preemptcond is not None and fireflies[i].intensity == 1):
                        print("Preempt t=" + str(t))

                        if (graph):
                            d_out.append(best_ff.euclid_dist(target_Tf))
                            angle_out.append(best_ff.angle_dist(target_Tf))
                            return [d_out, angle_out]
                        else:
                            return fireflies[i]

        # Get current best firefly
        best_i = get_best(fireflies)
        if (best_ff is None or fireflies[best_i].intensity > best_ff.intensity):
            best_ff = fireflies[best_i]

        # Random walk the best firefly
        # fireflies[best_i].random_walk(alpha)
        # fireflies[best_i].compute_fkine()
        # fireflies[best_i].compute_I(target_Tf, gamma, preemptcond)
        # if (preemptcond is not None and fireflies[best_i].intensity == 1):
        #     print("Preempt")
        #     if (graph):
        #         d_out.append(best_ff.euclid_dist(target_Tf))
        #         angle_out.append(best_ff.angle_dist(target_Tf))
        #         return [d_out, angle_out]
        #     else:
        #         return fireflies[best_i]

        t = t + 1
        alpha = alpha_new(alpha, t, maxGenerations)
        
        # Misc.
        if (graph):
            d_out.append(best_ff.euclid_dist(target_Tf))
            angle_out.append(best_ff.angle_dist(target_Tf))
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
    # x = 100
    # delta = 1 - 0.005**(x/t) # custom
    # delta = 0.985
    delta = (0.005/0.9)**(1.0/maxGenerations) # Yang
    
    return delta*alpha

def get_best(fireflies):
    intensities = np.array([ff.intensity for ff in fireflies])
    return np.argmax(intensities)

def f_kine(angles):
    # ===================== USER TO MODIFY =====================
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
    # ===================== END USER TO MODIFY =====================

    return Tf_out

def solve_IK(target_Tf, arg):
    alpha = arg['alpha']
    beta = arg['beta']
    gamma = arg['gamma']

    maxGenerations = arg['maxGenerations']
    n = arg['n']

    preemptcond = arg['preemptcond']

    t = 0
    t_cutoff = 10

    while True:
        sln = firefly_IK(target_Tf, maxGenerations, n, alpha0=alpha, beta=beta, gamma=gamma, preemptcond=preemptcond)

        # Check solution -> within tolerances
        if (sln.euclid_dist(target_Tf) < preemptcond['dist_tol_mm'] and sln.angle_dist(target_Tf) < preemptcond["angle_tol_rad"]):
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

def finetune_task(args):
    alpha = args[0]
    beta = args[1]
    gamma = args[2]
    gamma_i = args[3]
    
    maxGenerations = 400
    n_ff = 20
    
    preemptcond = {"dist_tol_mm": 0.1, "angle_tol_rad": 0.017}
    # preemptcond = None

    target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))
    
    # target_Tf = np.array([[ 5.90672098e-01, -7.80534494e-01, -2.04627410e-01, -2.12052914e+01],
    # [ 7.94408753e-01,  6.06979364e-01, -2.21536601e-02, -2.50164149e+00],
    # [ 1.41496311e-01, -1.49472256e-01,  9.78589208e-01,  2.70310590e+02],
    # [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    sln = firefly_IK(target_Tf, maxGenerations, n_ff, alpha0=alpha, beta=beta, gamma=gamma, gamma_i=gamma_i, preemptcond=preemptcond)

    # avg_d = avg_d + sln.euclid_dist(target_Tf)
    # avg_ang_d = avg_ang_d + sln.angle_dist(target_Tf)
    # avg_i = avg_i + sln.intensity
    return [sln.euclid_dist(target_Tf), sln.angle_dist(target_Tf), sln.intensity]

def finetune_FA_IK():
    # Search space
    # alpha_s = [0.1]
    # beta_s = [0.1]
    # gamma_s = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    # alpha_s = [0.1, 0.01, 0.001]
    # beta_s = [0.1, 0.01, 0.001]
    # alpha_s = [0.05, 0.1, 0.15, 0.4, 0.8]
    # beta_s = [0.05, 0.1, 0.15, 0.4, 0.8]
    # alpha_s = [0.075, 0.1, 0.125]
    # beta_s = [0.3, 0.4, 0.6]
    alpha_s = [0.125]
    beta_s = [0.25]
    gamma_s = [0.0001]
    gamma_i_s = [100, 200, 400, 600, 800, 1000, 1500]

    search_space = list(it.product(alpha_s, beta_s, gamma_s, gamma_i_s))

    best_pair = [[],[],[]]
    best_d = None
    best_ang_d = None
    best_i = None
    valid_pairs = []

    for abg in search_space:
        avg_d = 0
        avg_ang_d = 0
        avg_i = 0

        # Loop n times
        import multiprocessing as mp
        n = 16

        pool = mp.Pool()
        a = []
        [a.append(abg) for _ in range(n)]
        ans = pool.map(finetune_task, a)
        # End for loop

        avg = np.sum(np.array(ans), 0) / n

        avg_d = avg[0]
        avg_ang_d = avg[1]
        avg_i = avg[2]
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
        if (best_i is None or avg_i > best_i):
            best_pair[2] = abg
            best_i = avg_i

        if (avg_d < 0.1 and avg_ang_d < 0.017):
            valid_pairs.append(abg)

    print("================================BEST")
    print(alpha_s)
    print(beta_s)
    print(gamma_s)
    print("Distance, angle, intensity")
    print(best_pair)
    print(best_d)
    print(best_ang_d)
    print(best_i)
    print("Valid pairs:")
    print(valid_pairs)

def graph_task(arg):
    alpha = arg['alpha']
    beta = arg['beta']
    gamma = arg['gamma']

    maxGenerations = arg['maxGenerations']
    n = arg['n']

    preemptcond = arg['preemptcond']

    # target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))
    
    target_Tf = np.array([[ 5.90672098e-01, -7.80534494e-01, -2.04627410e-01, -2.12052914e+01],
    [ 7.94408753e-01,  6.06979364e-01, -2.21536601e-02, -2.50164149e+00],
    [ 1.41496311e-01, -1.49472256e-01,  9.78589208e-01,  2.70310590e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    return firefly_IK(target_Tf, maxGenerations, n, graph=True, alpha0=alpha, beta=beta, gamma=gamma, preemptcond=preemptcond)

def graph_FA_IK(arg):
    num_times = 8

    import multiprocessing as mp
    
    pool = mp.Pool()
    a = []
    [a.append(arg) for _ in range(num_times)]
    ans = pool.map(graph_task, a)

    fig, ax = plt.subplots(2)
    for i in range(0,num_times):
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

        x = np.array(list(range(1, len(ans[i][0])+1)))
        ax[0].plot(x, ans[i][0], '-')
        x = np.array(list(range(1, len(ans[i][1])+1)))
        ax[1].plot(x, ans[i][1], '-')
        
    plt.show()
    
    pool.close()

def debug(arg):
    alpha = arg['alpha']
    beta = arg['beta']
    gamma = arg['gamma']

    maxGenerations = arg['maxGenerations']
    n = arg['n']

    preemptcond = arg['preemptcond']

    target_Tf = np.array([[ 5.90672098e-01, -7.80534494e-01, -2.04627410e-01, -2.12052914e+01],
    [ 7.94408753e-01,  6.06979364e-01, -2.21536601e-02, -2.50164149e+00],
    [ 1.41496311e-01, -1.49472256e-01,  9.78589208e-01,  2.70310590e+02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))

    sln = firefly_IK(target_Tf, maxGenerations, n, debug=False, alpha0=alpha, beta=beta, gamma=gamma, preemptcond=preemptcond)

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

if __name__ == "__main__":
    arg = {
        'alpha': 0.125,
        'beta': 0.3,
        'gamma': 0.0001,
        'maxGenerations': 300,
        'n': 20,
        'preemptcond': {"dist_tol_mm": 0.1, "angle_tol_rad": 0.017}
    }


    # debug_profile(arg)
    # debug(arg)
    finetune_FA_IK()
    # graph_FA_IK(arg)


    # from time import perf_counter
    # start = perf_counter()

    # # No solution:
    # target_Tf = [[ 5.90672098e-01, -7.80534494e-01, -2.04627410e-01, -2.12052914e+01],
    # [ 7.94408753e-01,  6.06979364e-01, -2.21536601e-02, -2.50164149e+00],
    # [ 1.41496311e-01, -1.49472256e-01,  9.78589208e-01,  2.70310590e+02],
    # [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    
    # # target_Tf = f_kine([pi, -0.9197, 1.196, 0, 0.276, -pi/2.0])

    # # target_Tf = f_kine(np.array([random.uniform(-pi, pi) for _ in range(num_angles)]))
    # solve_IK(target_Tf, arg)

    # end = perf_counter()
    # print("Completed FA-IK in " + str(end-start) + "s")

    print("wait")