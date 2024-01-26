# -*- coding: utf-8 -*-

import numpy as np

from emcee.moves import RedBlueMove


class IntMove(RedBlueMove):
    """
    # TODO:
    """

    def __init__(self, res_conf, **kwargs):
        self.res_conf = res_conf
        super(IntMove, self).__init__(**kwargs)

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        zz = ((self.a - 1.0) * random.rand(Ns) + 1) ** 2.0 / self.a
        factors = (ndim - 1.0) * np.log(zz)
        rint = random.randint(Nc, size=(Ns,))


        [randrange_float(var['min_val'], var['max_val'], var['step']) for var in conf_res['var_params_list']]

        return c[rint] - (c[rint] - s) * zz[:, None], factors
