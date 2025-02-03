import scipy as sp
import numpy as np
class KeyRateCalculator():

    def __init__(self, detector_error, efficiency_alice, efficiency_bob, dark_count_alice, dark_count_bob, f_e):
        self.e_det = detector_error
        self.eta_a = efficiency_alice
        self.eta_b = efficiency_bob
        self.y_a = dark_count_alice
        self.y_b = dark_count_bob
        self.f_e = f_e


    def q_lambda(self, l):
        alice_red = (1-self.y_a) / np.power(1 + self.eta_a * l, 2)
        bob_red = (1 - self.y_b) / np.power(1 + self.eta_b * l, 2)
        corr_term = (1-self.y_a) * (1 - self.y_b) / np.power(1 + self.eta_a * l + self.eta_b * l -self.eta_a * self.eta_b * l, 2)
        return 1-alice_red - bob_red + corr_term

    def e_lambda(self, l):
        q_lambda = self.q_lambda(l)
        num = 2 * (0.5-self.e_det) * self.eta_a * self.eta_b * l * (1+l)
        denom = (1 + self.eta_a * l) * (1 + self.eta_b * l) *(1+ self.eta_a * l + self.eta_b * l - self.eta_a * self.eta_b * l)
        if q_lambda > 0:
            return 0.5 - num /(denom * q_lambda)
        else:
            return None

    def bin_entropy(self, p):
        if p >= 0:
            p_k = np.array([p, 1-p])
            return sp.stats.entropy(p_k, base=2)
        else:
            return None

    def e_lambda_simple(self, l):
        return (self.e_det + l + self.e_det * l) / (1 + 3 * l)


    def opt_lambda_eq(self, l):
        l = l[0]
        e_lambda = self.e_lambda_simple(l)
        if e_lambda != None:
            h_2_p = self.bin_entropy(e_lambda)
            if h_2_p != None:
                return (1 + 6 * l) * (1 - (1 + self.f_e)* h_2_p ) - l * (1+self.f_e) * (1-2 * self.e_det) * np.log2((1-e_lambda)/e_lambda)/(1 + 3 * l)

    def get_opt_lambda(self):
        root = sp.optimize.fsolve(self.opt_lambda_eq, x0 = [0.00])
        return root[0]



    def evaluate_values(self, l_opt):
        Q_lambda = self.q_lambda(l_opt)
        e_lamdba = self.e_lambda(l_opt)
        delta_b = e_lamdba
        epsilon = np.sqrt(200 * delta_b * (1-delta_b) / (1.5 * (10  ** 11) *Q_lambda))
        delta_p = delta_b + epsilon
        return Q_lambda, delta_b, delta_p

    def get_rate(self, Q_lambda, delta_b, delta_p):
        if delta_p >1:
            return 0
        else:
            rate = 0.5 * Q_lambda * (1-self.f_e * self.bin_entropy(delta_b) - self.bin_entropy(delta_p))
            try:
                assert rate != np.infty
            except:
                l_opt = self.get_opt_lambda()
                q_lambda, delta_b, delta_p = self.evaluate_values(l_opt)
                print("infinity discovered: delta_b: " + str(delta_b) + ", delta_p: " + str(delta_p))
            return max(rate, 0)


    def get_current_rate(self):
        l_opt = self.get_opt_lambda()
        q_lambda, delta_b, delta_p = self.evaluate_values(l_opt)
        return self.get_rate(q_lambda, delta_b, delta_p)


if __name__ == "__main__":
    krc = KeyRateCalculator(detector_error = 0.015, efficiency_alice = 0.1, efficiency_bob = 0.1, dark_count_alice = 6 * 10 ** -6, dark_count_bob= 6 * 10 ** -6, f_e = 1.16)
    l = krc.get_opt_lambda()
    print(str(l))
    rate = krc.get_current_rate()
    print(str(rate))