import scipy as sp
import numpy as np


class BB84KeyRates(object):

    def __init__(self, detector_error, efficiency_alice, efficiency_bob, dark_count_alice, dark_count_bob, f_e, eps_sec, mu_2, mu_1, p_1, p_2, p_3):
        self.e_det = detector_error
        self.eta_a = efficiency_alice
        self.eta_b = efficiency_bob
        self.y_a = dark_count_alice
        self.y_b = dark_count_bob
        self.f_e = f_e
        self.eps_sec = eps_sec
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = 0
        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3


    def get_nxkpm(self, k):
        if k == 1:
            mu = self.mu_1
        elif k== 2:
            mu = self.mu_2
        elif k==3:
            mu = self.mu_3

