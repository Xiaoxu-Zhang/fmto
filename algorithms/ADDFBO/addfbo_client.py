import copy
import numpy as np
from collections import defaultdict
from pathlib import Path
from pyfmto.framework import Client, record_runtime
from pyfmto.utilities import logger

from .addfbo_utils import SmtModel, DifferentialEvolution, Actions, ClientPackage

rng = np.random.default_rng()

_ROOT = Path(__file__).parent


class AddfboClient(Client):
    """
    gamma: 0.5  # weight factor for weighted lcb
    kappa: 1.4  # scale factor for std in lcb
    window: 5  # slide window for average distance calculation
    F: 0.4  # DE parameter
    CR: 0.9  # DE parameter
    max_gen: 20  # DE max generation
    num_elite: 5  # DE number of elite
    init_pop_size: 200  # initial population size of DE
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem=problem)
        # init control args
        self.problem.auto_update_solutions = True
        self.gamma = kwargs['gamma']
        self.kappa = kwargs['kappa']
        self.window = kwargs['window']

        # init model args
        self.local_model = SmtModel()
        self.global_model = SmtModel()

        # init DE operator
        self.de_operator = DifferentialEvolution(x_lb=self.lb, x_ub=self.ub, dim=self.dim,
                                                 max_gen=kwargs['max_gen'], F=kwargs['F'], CR=kwargs['CR'],
                                                 num_elite=kwargs['num_elite'], init_pop_size=kwargs['init_pop_size'])

        # Knowledge data
        self.d_aux = None
        self.d_aux_y_avg = None
        self.filtered_d_share_x = None
        self.use_global = True
        self.eval_info = defaultdict(list)
        self.version = 1
        self.round = 0
        self.last_y_min = 0.0
        self.server_update = None
        logger.info(f"{self.name} Initialized")

    @property
    def initialized(self):
        return self.d_aux is not None

    @property
    def all_x(self):
        return self.solutions.x

    @property
    def all_y(self):
        return self.solutions.y.squeeze()

    def update_aux_y(self):
        if not self.use_global:
            return
        y_avg = self.server_update.get('d_aux_y_avg')
        y_mask = self.server_update.get('col_mask')
        self.d_aux_y_avg = self.y_min + y_avg * (self.y_max - self.y_min)
        self.filtered_d_share_x = self.d_aux[y_mask]
        self.d_aux_y_avg = self.d_aux_y_avg[y_mask]

    @record_runtime('Total')
    def optimize(self):
        self.update_local_model()
        self.pull_aux()
        self.push_knowledge()
        self.pull_knowledge()
        self.update_scheme()
        self.update_aux_y()
        self.update_global_model()
        next_x = self._find_next_x()

        self.last_y_min = self.y_min
        self.problem.evaluate(next_x)
        self.update_improvement()
        self.version += 1

    def pull_aux(self):
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_AUX, version=self.version, dim=self.dim)
        resp = self.request_server(pkg, repeat=100, interval=0.1, msg="Pull auxiliary data")
        self.d_aux = self.problem.denormalize_x(resp)
        logger.debug(f"{self.name} auxiliary data updated to V{self.version}")

    def update_improvement(self):
        y_improve = self.last_y_min - self.solutions.y[-1]
        self.record_round_info('Improve', y_improve)
        if self.use_global:
            self.eval_info['g'].append(y_improve)
        else:
            self.eval_info['l'].append(y_improve)

    def update_scheme(self):
        if len(self.eval_info['l']) == 0 or len(self.eval_info['g']) == 0:
            self.use_global = not self.use_global
            self.record_round_info(f'Improve recent({self.window})', f"l({0}) g({0})")
            self.record_round_info('Probability', f"l({0.5}) g({0.5})")
        else:
            avg_improve_l = np.mean(self.eval_info['l'][-self.window:])
            avg_improve_g = np.mean(self.eval_info['g'][-self.window:])

            sign_l = np.sign(avg_improve_l)
            sign_g = np.sign(avg_improve_g)

            improv_sum = avg_improve_l + avg_improve_g
            if sign_l > 0 and sign_g > 0:
                prob_l = avg_improve_l / improv_sum
            elif sign_l < 0 and sign_g < 0:
                prob_l = avg_improve_g / improv_sum
            else:
                prob_l = 0.9 if sign_l > 0 else 0.1
            prob_g = 1 - prob_l
            self.use_global = np.random.choice(a=[False, True], p=[prob_l, prob_g])
            self.record_round_info(f'Improve recent({self.window})', f"l({avg_improve_l:.2f}) g({avg_improve_g:.2f})")
            self.record_round_info('Probability', f"l({prob_l:.2f}) g({prob_g:.2f})")

    @record_runtime('Push')
    def push_knowledge(self):
        d_share_y, d_share_y_std = self._predict_d_share(self.d_aux)
        pkg = ClientPackage(
            cid=self.id,
            action=Actions.PUSH_UPDATE,
            version=self.version,
            d_share_y=d_share_y,
            d_share_y_std=d_share_y_std)
        resp = self.request_server(pkg, msg="Push update")
        logger.debug(f"Response of update V{self.version}: {resp}")
        logger.debug(f"{self.name} pushed update V{self.version} to server")

    @record_runtime('Pull')
    def pull_knowledge(self):
        pkg = ClientPackage(cid=self.id, action=Actions.PULL_UPDATE, version=self.version)
        msg = f"Pull update [V {self.version}]"
        self.server_update = self.request_server(package=pkg, repeat=100, interval=1, msg=msg)

    @record_runtime('OptAF')
    def _find_next_x(self):
        next_x, ac_best = self._de_method(self._acq_eval)
        next_x = self._check_new_x(next_x)
        return next_x

    def _de_method(self, function):
        archive = copy.deepcopy(self.all_x)
        return self.de_operator.minimize(function, archive=archive)

    def _check_new_x(self, new_x):
        x_next = new_x.reshape(-1, self.dim)
        if new_x in self.all_x:
            x_from = 'random'
            x_next = self.problem.random_uniform_x(1)
        else:
            x_from = 'af'
        self.record_round_info('x from', x_from)
        return x_next

    @record_runtime('FitGP_L')
    def update_local_model(self):
        l_size = self.all_x.shape[0]
        self.record_round_info('L Size', f"{l_size}")
        self.local_model.fit(self.all_x, self.all_y)

    @record_runtime('FitGP_G')
    def update_global_model(self):
        g_size = self.filtered_d_share_x.shape[0] if self.use_global else '-'
        if self.use_global:
            self.global_model.fit(self.filtered_d_share_x, self.d_aux_y_avg)
        self.record_round_info('G Size', f"{g_size}")

    def _predict_d_share(self, d_share_x):
        d_share_y, d_share_y_std = self.local_model.predict(d_share_x, return_std=True)
        y_max, y_min = np.max(d_share_y), np.min(d_share_y)
        std_max, std_min = np.max(d_share_y_std), np.min(d_share_y_std)
        d_share_y = (d_share_y - y_min) / max((y_max - y_min), 1)
        d_share_y_std = (d_share_y_std - std_min) / max((std_max - std_min), 1)
        return d_share_y.squeeze(), d_share_y_std.squeeze()

    def _init_model(self):
        logger.debug(f"{self.name} initializing local GP model [x_shape {self.all_x.shape} y_shape {self.all_y.shape}]")
        self.local_model.fit(self.all_x, self.all_y)

    def _acq_eval(self, x_new):
        if len(x_new.shape) == 1:
            x_new = x_new.reshape(1, -1)
        return self.eval_lcb(x_new)

    def eval_lcb(self, x_new):
        lcb_l = self._lcb(x_new, self.local_model)
        if self.use_global:
            lcb_g = self._lcb(x_new, self.global_model)
            return self.gamma * lcb_l + (1 - self.gamma) * lcb_g
        else:
            return lcb_l

    def _lcb(self, x_new, model: SmtModel):
        mean_y_new, sigma_y_new = model.predict(x_new, return_std=True)
        lcb = mean_y_new - self.kappa * sigma_y_new
        return lcb.squeeze()
