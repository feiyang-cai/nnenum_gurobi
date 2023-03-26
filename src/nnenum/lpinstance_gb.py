'''
Ali A.Bigdeli
March 2023
Gurobi python interface using gurobipy
'''


import sys
import math
import time
from typing import List, Tuple, Dict

import numpy as np

from nnenum.util import Freezable
from nnenum.timerutil import Timers
from nnenum.settings import Settings

import gurobipy as gp
from gurobipy import GRB, LinExpr


class LpInstanceGB(Freezable):
    'Linear programming wrapper using Gurobipy'

    lp_time_limit_sec = 15.0

    def __init__(self, other_lpi=None): 
        'initialize the lp instance'

        # self.cached_vars = None

        if other_lpi is not None:
            Timers.tic('gb_copy_model')
            self.model = other_lpi.model.copy()
            Timers.toc('gb_copy_model')
        else:
            self.model = gp.Model()
            self.model.setParam('OutputFlag', False)

            # self.init_from_box(box)

        self.print_failure_msg = True
        self.need_update = False
        self.freeze_attrs()

    
    def __deepcopy__(self, _):
        if self.need_update:
            self.update()
            
        return LpInstanceGB(clone_hpoly=self)

    def update(self):
        """update the model"""

        assert self.need_update

        self.need_update = False
        self.model.update()    

    def get_vars(self):
        """get the vars"""

        self.model.update()
        return self.model.getVars()
    
    # def serialize(self):
    #     'serialize self.lp from a glpk instance into a tuple'

    #     Timers.tic('serialize')
        
    #     # get constraints as csr matrix
    #     lp_rows = self.get_num_rows()
    #     lp_cols = self.get_num_cols()



    #     inds_row = SwigArray.get_int_array(lp_cols + 1)
    #     vals_row = SwigArray.get_double_array(lp_cols + 1)

    #     data = []
    #     glpk_indices = []
    #     indptr = [0]

    #     for row in range(lp_rows):
    #         got_len = glpk.glp_get_mat_row(self.lp, row + 1, inds_row, vals_row)

    #         for i in range(1, got_len+1):
    #             data.append(vals_row[i])
    #             glpk_indices.append(inds_row[i])

    #         indptr.append(len(data))
            
    #     # rhs
    #     rhs = []
        
    #     for row in range(lp_rows):
    #         assert glpk.glp_get_row_type(self.lp, row + 1) == glpk.GLP_UP

    #         rhs.append(glpk.glp_get_row_ub(self.lp, row + 1))

    #     col_bounds = self._get_col_bounds()

    #     # remember to free lp object before overwriting with tuple
    #     glpk.glp_delete_prob(self.lp)
    #     self.lp = (data, glpk_indices, indptr, rhs, col_bounds)

    #     Timers.toc('serialize')


    def add_var(self, name, lb, ub):
        """add a bounded variable, get stored in self.vars"""

        self.model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
    
    
    def add_double_bounded_cols(self, names, lb, ub):
        'add a certain number of columns to the LP with the given lower and upper bound'

        assert lb != -np.inf

        lb = float(lb)
        ub = float(ub)
        assert lb <= ub, f"lb ({lb}) <= ub ({ub}). dif: {ub - lb}"

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:

            for i in range(num_vars):
                self.model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=names[i])
                self.model.update()

    def add_positive_cols(self, names):
        'add a certain number of columns to the LP with positive bounds'

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:

            for i in range(num_vars):
                self.model.addVar(lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name=names[i]) # var with lower bounds (0, inf)

    def add_dense_row(self, vec, rhs, normalize=True):
        '''
        add a row from a dense nd.array, row <= rhs
        '''

        Timers.tic('add_dense_row')

        assert isinstance(vec, np.ndarray)
        assert len(vec.shape) == 1 or vec.shape[0] == 1
        assert len(vec) == self.get_num_cols(), f"vec had {len(vec)} values, but lpi has {self.get_num_cols()} cols"

        if normalize and not Settings.SKIP_CONSTRAINT_NORMALIZATION:
            norm = np.linalg.norm(vec)
            
            if norm > 1e-9:
                vec = vec / norm
                rhs = rhs / norm

        variables = self.get_vars()

        self.model.addLConstr(LinExpr(vec, variables), GRB.LESS_EQUAL, rhs)

        self.need_update = True

        Timers.toc('add_dense_row')

    def dims(self):
        """return number of dimensions"""

        return len(self.get_vars())
    
    def get_num_cols(self):
        'get the number of cols in the lp'

        self.model.update()
        return self.model.NumVars
    
    def get_num_rows(self):
        'get the number of rows in the lp'
        
        self.model.update()
        return len(self.model.getConstrs())
    
    def minimize(self, obj_vec, fail_on_unsat=True):
        """return minimum point or raise MinimizeFailed exception"""

        if obj_vec is None:
            obj_vec = [0] * self.get_num_cols()

        min_start = time.perf_counter()

        if not isinstance(obj_vec, np.ndarray):
            obj_vec = np.array(obj_vec, dtype=float)

        variables = self.get_vars() #original code

        # do this after get_vars, since that sometimes will update
        if self.need_update:
            self.update()

        self.model.setObjective(obj_vec @ variables, GRB.MINIMIZE)

        #self.m.display() ## debug display model

        start = time.perf_counter()
        self.model.optimize()
        diff = time.perf_counter() - start


        if self.model.status != GRB.OPTIMAL:
            if self.print_failure_msg:
                statuses = ["ZERO?", "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF",
                            "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOL_LIMIT", "INTERRUPTED", "NUMERIC",
                            "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT"]

                code = self.model.status
                status = statuses[code]
                print(f"Gurobi.optimize() failed: status was {status} ({code})")

            rv = None
        else:
            vals: List[float] = []

            for v in variables:
                # assert self.vars[len(vals)] == v
                vals.append(v.x)

            rv = np.array(vals)


        diff = time.perf_counter() - min_start
        
        if rv is None and fail_on_unsat:
            raise UnsatError("minimize returned UNSAT and fail_on_unsat was True")

        return rv
    

class UnsatError(RuntimeError):
    'raised if an LP is infeasible'
    