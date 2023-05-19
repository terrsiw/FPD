import random
import numpy as np
from scipy.optimize import fsolve
import pickle
import json


class Agent:
    """The class Agent saves all information about the agent. The closed loop information, properties as functions:
    learn and optimal functions for FPD.

    ss....number of states of the system
    aa....number of actions of the system
    h....horizon of FPD
    w....weight balancing two preferences
    s0...the initial state
    nu....the parameter of exploration
    si....the preferred state
    ai....the preferred action
    alfa....variable that needs to be optimized for finding ideal model
    sigma....parameter for creating model
    gam....function for FPD r_opt = ri*exp(d)/gam
    model....model of the system
    r....decision rule
    mi....ideal model of the system
    ri....ideal decision rule
    V....variable for learning model
    r_side....right side of the equation

        """

    def __init__(self, ss: int, aa: int, h: int, w: int, s0: int, nu: int, si: int, ai: np.ndarray, alfa: np.ndarray,
                 sigma: int) -> None:
        self.ss = ss
        self.aa = aa
        self.h = h
        self.w = w
        self.s0 = s0
        self.nu = nu
        self.si = si
        self.ai = ai
        self.alfa = alfa
        self.sigma = sigma
        self.gam = np.ones(ss)
        self.model = self.create_model()
        self.r = np.ones((self.aa, self.ss)) / self.aa
        self.mi = np.ones((self.ss, self.aa, self.ss)) / self.ss
        self.ri = np.ones((self.aa, self.ss)) / self.aa
        self.V = np.ones((self.ss, self.aa, self.ss))
        self.r_side = np.zeros(self.aa)

    def create_model(self):
        """ Create Model:
        create probability model with Gauss distribution and then normalization

        :return: model
        """

        model = np.ones(tuple([self.ss, self.aa, self.ss], ))

        for s1 in range(self.ss):
            for a in range(self.aa):
                for s2 in range(self.ss):
                    model[s2, a, s1] = np.exp(-(s2 - s1 - a) ** 2 / (2 * self.sigma ** 2))

        model = self.normalize_proba_model(model)

        return model

    def normalize_proba_model(self, model):
        """ Normalize Model:
                Normalization of the created model

                :return: model
                """
        for s1 in range(self.ss):
            for a in range(self.aa):
                model[:, a, s1] = model[:, a, s1] / np.sum(model[:, a, s1])

        return model

    def learn(self, data2):
        """ Learn the system and saves it to the model of the system


        :param data2: the structure where we save the data: observed states and chose actions
        :return: model
        """
        s = data2.states[data2.t]
        a = data2.actions[data2.t - 1]
        s1 = data2.states[data2.t - 1]
        self.V[s.astype(np.int64), a.astype(np.int64), s1.astype(np.int64)] = self.V[s.astype(np.int64), a.astype(
            np.int64), s1.astype(np.int64)] + 1

        self.model[:, a.astype(np.int64), s1.astype(np.int64)] = self.V[:, a.astype(np.int64),
                                                                 s1.astype(np.int64)] / np.sum(
            self.V[:, a.astype(np.int64), s1.astype(np.int64)])

    def opt_mi(self, a, s1):
        """ Calculating the ideal model

        :param a: action
        :param s1: state
        :return: mi: ideal model
        """

        for s in range(self.ss):
            self.mi[s, a, s1] = self.model[s, a, s1] * np.exp(-self.alfa[a, s1] * self.model[s, a, s1])

    def opt_mio(self, a, s1, o):
        """ Calculating the ideal model when we have uniform distributed model

        :param a: action
        :param s1: state
        :param o: auxiliary variable
        :return: mi: ideal model
        """

        for s in range(self.ss):
            self.mi[s, a, s1] = np.exp(-self.alfa[a, s1] * o[s, a, s1])

    def opt_ri(self, s1, d):
        """ Calculating the ideal decision rule

        :param s1: state
        :param d: a variable need for FPD
        :return: ri: ideal decision rule
        """
        for a in range(self.aa):
            self.ri[a, s1] = np.exp(-self.nu * d[a, s1])

        self.ri[:, s1] = self.ri[:, s1] / np.sum(self.ri[:, s1])

    def opt_function(self, alfa_var, lambda_var, a, s1, r_side):
        """ The function that is optimized by fsolve. We try to find the best alfa_var to f = 0.

        :param alfa_var: the parameter that is chosen to optimize the f
        :param lambda_var: an auxiliary variable for left_side=right_side
        :param a: action
        :param s1: state
        :param r_side: right side of the equation for optimization
        :return: f : we want to 0
        """
        l_side = alfa_var * lambda_var[a, s1] + np.log(
            np.sum(self.model[:, a, s1] * np.exp(-alfa_var * self.model[:, a, s1])))
        f = l_side - r_side

        return f

    def opto_function(self, alfa_var, a, s1, r_side, o):
        """ The function that is optimized by fsolve for uniform distributed model .
        We try to find the best alfa_var to f = 0.


        :param alfa_var: the parameter that is chosen to optimize the f
        :param a: action
        :param s1: state
        :param r_side: right side of the equation for optimization
        :param o: an auxiliary variable for uniform distributed model
        :return: f : we want to 0
        """
        l_side = np.log(np.sum(np.exp(-alfa_var * o[:, a, s1])) / self.ss)

        f = l_side - r_side

        return f

    def calculate_alfa(self):
        """ Calculate alfa (optimize it and use it for ideal model, using fsolve) and also FPD algorithm.

        :return: optimized ideal distribution and decision rule
        """
        lambda_var = np.zeros((self.aa, self.ss))
        rho = np.zeros((self.aa, self.ss))
        rho_max = np.zeros(self.ss)
        for s1 in range(self.ss):
            for a in range(self.aa):
                lambda_var[a, s1] = np.sum(self.model[:, a, s1] ** 2)
                rho[a, s1] = np.sum(self.model[self.si, a, s1])
                if np.any(self.ai == a):
                    rho[a, s1] = (1 - self.w) * rho[a, s1] + self.w
                if rho[a, s1] >= rho_max[s1]:
                    rho_max[s1] = rho[a, s1]

        for tau in range(self.h, 0, -1):
            d = np.zeros((self.aa, self.ss))
            d_opt = np.zeros((self.aa, self.ss))
            d_help = np.zeros(self.ss)
            for s1 in range(self.ss):
                var = np.zeros(self.aa)
                for a in range(self.aa):
                    var[a] = np.sum(self.model[:, a, s1] * np.log(rho[a, s1] / (rho_max[s1] * self.gam[:])))

                    if d_help[s1] < var[a]:
                        d_help[s1] = var[a]
                if d_help[s1] < 0:
                    d_help[s1] = 0

                for a in range(self.aa):
                    d_opt[a, s1] = d_help[s1] + np.log(rho_max[s1] / rho[a, s1])
                    if np.sum(self.model[:, a, s1] != np.ones(self.ss)) > 0:
                        r_side = d_opt[a, s1] + np.sum(self.model[:, a, s1] * np.log(self.gam[:]))

                        alfaa = self.alfa[a, s1]
                        alfaa = fsolve(self.opt_function, alfaa, args=(lambda_var, a, s1, r_side))
                        self.alfa[a, s1] = alfaa
                        self.opt_mi(a, s1)
                    else:
                        o = np.ones((self.ss, self.aa, self.ss))
                        o[self.ss, :, :] = - (self.ss - 1) * o[self.ss, self.aa, self.ss]
                        o = o / self.aa

                        r_side = d_opt[a, s1] + np.sum(
                            self.model[:, a, s1] * np.log((self.gam[:] * rho_max[s1]) / rho[a, s1]))

                        alfaa = self.alfa[a, s1]
                        alfaa = fsolve(self.opto_function, alfaa, args=(lambda_var, a, s1, r_side, o))
                        self.alfa[a, s1] = alfaa
                        self.opt_mio(a, s1, o)

                for a in range(self.aa):
                    self.mi[:, a, s1] = self.mi[:, a, s1] / np.sum(self.mi[:, a, s1])

                for a in range(self.aa):
                    d[a, s1] = np.sum(
                        self.model[:, a, s1] * np.log(self.model[:, a, s1] / (self.gam[:] * self.mi[:, a, s1])))

                self.opt_ri(s1, d_opt)
                self.gam[s1] = np.sum(self.ri[:, s1] * np.exp(-d_opt[:, s1]))
                self.r[:, s1] = (self.ri[:, s1] * np.exp(-d_opt[:, s1])) / self.gam[s1]


class Data2:

    def __init__(self, length_sim: int, s0: int) -> None:
        self.states = np.zeros(length_sim + 2)
        self.actions = np.zeros(length_sim + 1)
        self.length_sim = length_sim
        self.states[0] = s0
        self.t = 0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def simulate_system(ss: int, aa: int):
    A = 0.99
    # G = 1
    # B = G * (1 - A)
    var = 0.001
    number_dat = 50000
    V = 10 ** -5 * np.ones((ss, aa, ss))
    ra = np.ones(aa)
    y = np.ones(number_dat)
    a = np.ones(number_dat + 1)
    BB0 = np.zeros(number_dat)
    BB1 = np.zeros(number_dat)
    b0 = 0.03
    b1 = -0.01
    for t in range(2, number_dat - 1):
        # y[t] = A * y[t - 1] + B * (a[t] - aa/2) + var * np.random.normal(0, 1, 1)
        B_0 = b0 / (1 + y[t - 1])
        B_1 = b1 / (1 + y[t - 1])
        y[t] = A * y[t - 1] + B_0 * (a[t] - aa / 2) + B_1 * (a[t - 1] - aa / 2) + var * np.random.normal(0, 1, 1)
        a[t + 1] = dnoise(ra) + 1
        BB0[t - 1] = B_0
        BB1[t - 1] = B_1

    yy = np.floor((ss - 2) * (y - min(y)) / max(y))

    for t in range(1, number_dat - 1):
        V[yy[t].astype(np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] = V[yy[t].astype(
            np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] + 1

    for at in range(aa):
        for s1 in range(ss):
            V[:, at, s1] = V[:, at, s1] / np.sum(V[:, at, s1])

    with open("data_system", "wb") as f:
        pickle.dump(V, f, protocol=pickle.HIGHEST_PROTOCOL)

    return V


def dnoise(pr: np.ndarray) -> int:
    result = 0
    if pr.any() < 0:
        print("error")
    else:
        le = len(pr)
        pr = np.cumsum(pr)
        pr = pr / pr[le - 1]
        ru = np.random.uniform(0, 1, 1)[0]
        for i in range(le):
            if ru < pr[i]:
                result = i
                break
    return result


def generate(agent, data2, system):
    s1 = data2.states[data2.t]
    a = dnoise(agent.r[:, s1.astype(np.int64)])
    m = system[:, a, s1.astype(np.int64)]
    data2.states[data2.t + 1] = dnoise(m)
    data2.actions[data2.t] = a
    data2.t = data2.t + 1

    return data2


def initialization(length_sim, create_system=None):
    ss = 15
    aa = 7
    h = 2
    w = 0
    s0 = 0
    nu = 1
    si = 7
    ai = np.arange(start=0, stop=7, step=1)
    alfa = 1.2 * np.ones((aa, ss))
    sigma = 1
    random.seed(10)

    if create_system == "TRUE":
        system = simulate_system(ss, aa)
    else:
        system = load_object("data_system")

    agent = Agent(ss, aa, h, w, s0, nu, si, ai, alfa, sigma)
    data2 = Data2(length_sim, s0)
    return system, agent, data2


def main(length_sim, create_system):
    # length_sim = 500
    [system, agent, data2] = initialization(length_sim, create_system)

    while data2.t <= data2.length_sim:
        agent.calculate_alfa()
        data2 = generate(agent, data2, system)
        agent.learn(data2)

    print(agent)

    return agent, data2, system


# def save_object(obj):
#     try:
#         with open("data.pickle", "wb") as f:
#             pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#     except Exception as ex:
#         print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
