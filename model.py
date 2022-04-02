import gurobipy as gp
from gurobipy import GRB
import random
import math
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
FILE_PATH = './'+ "results//"

class Model():
    def __init__(self, data_set, qos_ub) -> None:
        self.qos, self.demand, self.bandwidth, self.site_name = data_set.qos, data_set.demand_values, data_set.bandwidth, data_set.site_name
        self.n_site, self.n_cust = data_set.n_site, data_set.n_cust # n_site 基站数量; n_cust:用户数
        self.qos_ub, self.n_time = qos_ub,  data_set.n_time # qos的上限; n_time:时刻数
        self.m = gp.Model('flow scheduled model') # 建立优化模型
        self.x = self.m.addVars(self.n_cust, self.n_site, self.n_time, lb=0, ub=GRB.INFINITY,vtype=GRB.INTEGER, name ='x') 
        self.w = self.m.addVars(self.n_site, self.n_time, lb=0, ub=GRB.INFINITY,vtype=GRB.INTEGER, name ='w')
        self.w_max = self.m.addVars(self.n_site, lb=0, ub=GRB.INFINITY,vtype=GRB.INTEGER, name ='w_max') # w_max:代表模型中的 s_hat
        
    def sample_time(self, quantile):
        self.quantile_point = math.ceil(quantile*self.n_time)
        self.sample_t = random.sample([i for i in range(self.n_time)], self.quantile_point) 
    
    def set_objective(self):
        self.m.setObjective(gp.quicksum(self.w_max[j] for j in range(self.n_site))) #目标函数 对应式(14)
    
    def set_qos_constrs(self):
        self.m.addConstrs((self.x[i,j,t] == 0 for i in range(self.n_cust) for j in range(self.n_site) for t in range(self.n_time) \
        if self.qos[i,j] >= self.qos_ub), name ='qos') # qos约束 对应式(2)
    
    def set_demand_constrs(self):
        self.m.addConstrs((gp.quicksum(self.x[i,j,t] for j in range(self.n_site)) == self.demand[i,t] for i in range(self.n_cust) \
        for t in range(self.n_time)), name = 'demand') # 需求约束 对应式(4)
    
    def set_bandwidth_constrs(self):
        self.m.addConstrs((self.w[j,t] <= self.bandwidth[j] for j in range(self.n_site) for t in range(self.n_time)), name = 'bandwidth') # 边缘节点带宽约束 对应式(5)
    
    def set_link_constrs(self):
        self.m.addConstrs((gp.quicksum(self.x[i,j,t] for i in range(self.n_cust)) == self.w[j,t] for j in range(self.n_site) \
        for t in range(self.n_time)), name ='link') # 中间变量约束 对应式(6)
    
    def set_quantile95_constrs(self):
        self.m.addConstrs((self.w_max[j] >= self.w[j,t] for j in range(self.n_site) for t in range(self.n_time) if t in self.sample_t), name = 'max') # 95分位数约束 对应式(10-11)

    def set_constrs(self):
        self.set_qos_constrs()
        self.set_demand_constrs()
        self.set_link_constrs()
        self.set_bandwidth_constrs()
        self.set_quantile95_constrs()
    
    def solve(self, time_limit):
        self.m.Params.TimeLimit = time_limit 
        self.m.optimize()
    
    def write_model(self):
        self.m.write('model.lp')
    
    def get_solution(self):
        vars = self.m.getVars() #获取求解结果
        self.x_dim = self.n_cust*self.n_site*self.n_time
        self.x_sol = np.array([v.x for v in vars[0:self.x_dim]]).reshape((self.n_cust, self.n_site, self.n_time))
        self.w_dim = self.n_site*self.n_time
        self.w_sol = np.array([v.x for v in vars[self.x_dim:self.x_dim+self.w_dim]]).reshape((self.n_site, self.n_time)).T
        self.w_max_dim = self.n_site
        self.w_max_sol = np.array([v.x for v in vars[self.x_dim+self.w_dim:self.x_dim+self.w_dim+self.w_max_dim]]).reshape((self.n_site, 1))
    
    def get_quantile(self):
        self.w_sorted = np.sort(self.w_sol, axis=0)
        self.w_q = self.w_sorted[self.quantile_point-1,:].reshape((self.n_site, 1))
        print("the 95 percent objective function: {}".format(np.sum(self.w_q)))

    def save_solution(self):
        pd.DataFrame(self.w_sol, columns=self.site_name).to_csv(FILE_PATH + "w_sol.csv")
        pd.DataFrame(np.hstack((self.w_max_sol, self.w_q, self.w_max_sol-self.w_q)), index=self.site_name, \
        columns=['max','95','diff']).to_csv(FILE_PATH + "w_max_sol.csv")
        pd.DataFrame(self.w_sorted, columns=self.site_name).to_csv(FILE_PATH + "w_sorted.csv")

