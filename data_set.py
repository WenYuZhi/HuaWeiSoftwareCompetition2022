import numpy as np
import pandas as pd

FILE_PATH = './'+ "offlinedata//"

class Data_Set:
    def __init__(self) -> None:
        pass

    def load(self):
        self.demand = pd.read_csv(FILE_PATH + 'demand.csv')
        self.qos = pd.read_csv(FILE_PATH + 'qos.csv', header=0).to_numpy()[:,1:].T
        self.site_bandwidth = pd.read_csv(FILE_PATH + 'site_bandwidth.csv')
    
    def handle_data(self):
        self.time = list(self.demand['mtime']) 
        self.customer = list(self.demand.columns)[1:]
        self.demand_values = self.demand.to_numpy()[:, 1:].T
        self.bandwidth = list(self.site_bandwidth['bandwidth'])
        self.site_name = list(self.site_bandwidth['site_name'])
        self.n_cust, self.n_site, self.n_time = len(self.customer), len(self.site_name), len(self.time)
        self.__assert()
    
    def __assert(self):
        assert(self.qos.shape[0] == self.n_cust)
        assert(self.qos.shape[1] == self.n_site)
        assert(self.demand_values.shape[0] == self.n_cust)
        assert(self.demand_values.shape[1] == self.n_time)
        



