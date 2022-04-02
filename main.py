from data_set import Data_Set
from model import Model

data_set = Data_Set()
data_set.load() # 读入数据
data_set.handle_data() # 数据预处理

flow_schedule = Model(data_set=data_set, qos_ub=400) # qos_ub:qos的上限
flow_schedule.sample_time(quantile=0.95) # 生成采用集合
flow_schedule.set_objective()  # 设置目标函数
flow_schedule.set_constrs()    # 设置约束条件
flow_schedule.solve(time_limit = 300)  # 调用求解器求解
flow_schedule.get_solution()  # 获得最优解
flow_schedule.get_quantile()  # 获得原目标函数值
flow_schedule.save_solution()  # 保存解到本地文件 
flow_schedule.write_model()   # 保存优化模型







