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
            self.lp = other_lpi.lp.copy()
            Timers.toc('gb_copy_model')
        else:
            self.lp = gp.Model()
            self.lp.setParam('OutputFlag', False)

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
        self.lp.update()    

    def get_vars(self):
        """get the vars"""

        self.lp.update()
        return self.lp.getVars()
    
    def serialize(self):
        'serialize self.lp from a gurobi model into a tuple'

        Timers.tic('serialize')
        
        # get constraints as csr matrix
        A_sparse = self.lp.getA()
            
        # rhs
        rhs = self.lp.getAttr('rhs', self.lp.getConstrs())

        # Get the lower bounds of all variables
        lbs = self.lp.getAttr(gp.GRB.Attr.LB, self.lp.getVars())
        # Get the upper bounds of all variables
        ubs = self.lp.getAttr(gp.GRB.Attr.UB, self.lp.getVars())
        # Get the names of all variables
        var_names = [var.varName for var in self.lp.getVars()]

        # remember to free lp object before overwriting with tuple
        self.lp.dispose()
        self.lp = (A_sparse, rhs, lbs, ubs, var_names)

        Timers.toc('serialize')


    def deserialize(self):
        'deserialize self.lp from a tuple into a gorubi model'

        assert isinstance(self.lp, tuple)

        Timers.tic('deserialize')

        A_sparse, rhs, lbs, ubs, var_names = self.lp

        self.lp = gp.Model()
        self.lp.setParam('OutputFlag', False)

        n_variables = len(var_names)
        x = self.lp.addVars(n_variables, lb=lbs, ub=ubs, name=var_names)
        vars = x.select() # use select method to get values of tupledict as a list
        # Define constraints
        for i in range(len(rhs)):
            vec = np.squeeze(A_sparse[i].toarray())
            lhs_expr = LinExpr(vec, vars) 
            self.lp.addLConstr(lhs_expr, GRB.LESS_EQUAL, rhs[i], name=f"R{i}")

        self.lp.update()
        
        Timers.toc('deserialize')
    

    def __str__(self, plain_text=False):
        'get the LP as string (useful for debugging)'

        rows = self.get_num_rows()
        cols = self.get_num_cols()
        rv = "Lp has {} columns (variables) and {} rows (constraints)\n".format(cols, rows)

        rv += self.lp.display()

        return rv
    

    def add_var(self, name, lb, ub):
        """add a bounded variable, get stored in self.vars"""

        self.lp.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
    
    
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
                self.lp.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=names[i])
                self.lp.update()

    def add_positive_cols(self, names):
        'add a certain number of columns to the LP with positive bounds'

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:

            for i in range(num_vars):
                self.lp.addVar(lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS, name=names[i]) # var with lower bounds (0, inf)

    def add_cols(self, names):
        'add a certain number of columns to the LP'

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:

            for i in range(num_vars):
                self.lp.addVar(lb=float('-inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name=names[i]) # free variable (-inf, inf)

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

        self.lp.addLConstr(LinExpr(vec, variables), GRB.LESS_EQUAL, rhs)

        self.need_update = True

        Timers.toc('add_dense_row')

    def dims(self):
        """return number of dimensions"""

        return len(self.get_vars())
    
    def get_num_cols(self):
        'get the number of cols in the lp'

        self.lp.update()
        return self.lp.NumVars
    
    def get_num_rows(self):
        'get the number of rows in the lp'
        
        self.lp.update()
        return len(self.lp.getConstrs())
    
    def minimize(self, obj_vec, fail_on_unsat=True):
        """return minimum point or raise MinimizeFailed exception"""

        if obj_vec is None:
            obj_vec = np.zeros(self.get_num_cols(), dtype='float')

        min_start = time.perf_counter()

        if not isinstance(obj_vec, np.ndarray):
            obj_vec = np.array(obj_vec, dtype=float)

        variables = self.get_vars() #original code

        # do this after get_vars, since that sometimes will update
        if self.need_update:
            self.update()

        self.lp.setObjective(obj_vec @ variables, GRB.MINIMIZE)

        #self.m.display() ## debug display model

        start = time.perf_counter()
        Timers.tic('GB optimize')
        self.lp.optimize()
        Timers.toc('GB optimize')
        diff = time.perf_counter() - start


        if self.lp.status != GRB.OPTIMAL:
            if self.print_failure_msg:
                statuses = ["ZERO?", "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF",
                            "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOL_LIMIT", "INTERRUPTED", "NUMERIC",
                            "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT"]

                code = self.lp.status
                status = statuses[code]
                print(f"Gurobi.optimize() failed: status was {status} ({code})")

            rv = None
        else:
            vals = [v.x for v in variables]
            rv = np.array(vals)


        diff = time.perf_counter() - min_start
        
        if rv is None and fail_on_unsat:
            raise UnsatError("minimize returned UNSAT and fail_on_unsat was True")

        return rv
    

class UnsatError(RuntimeError):
    'raised if an LP is infeasible'

    