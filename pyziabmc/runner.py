import numpy as np

import pyziabmc.trader as trader

from pyziabmc.orderbook import Orderbook


class Runner(object):
    def __init__(self, mpi=1, **kwargs):
                #num_mms=1, mm_maxq=1, mm_quotes=12, mm_quote_range=60, mm_delta=0.025, 
                #num_takers=50, taker_maxq=1, 
                #informed_maxq=1, informed_runlength=1, informed_mu=0.01,
                #mu=0.001, delta=0.025, lambda0=100, wn=0.001, c_lambda=1.0, run_steps=100000,
                #h5filename='test.h5', alpha_pj=0):
        self.mpi = mpi
        self.exchange = Orderbook()
        if kwargs['Provider']:
            self.t_delta_p, self.provider_array = self.buildProviders(kwargs['numProviders'], kwargs['providerMaxQ'],
                                                                      kwargs['pAlpha'], kwargs['pDelta'])
            
            
            
    def buildProviders(self, numProviders, providerMaxQ, pAlpha, pDelta):
        providers_list = ['p%i' % i for i in range(numProviders)]
        if self.mpi==1:
            providers = np.array([trader.Provider(p,providerMaxQ,self.mpi,pDelta,pAlpha) for p in providers_list])
        #else:
            #providers = np.array([trader.Provider5(p,i,self.mpi,pDelta,pAlpha) for p,i in zip(providers_list,providerMaxQ)])
        t_delta_p = np.array([p.delta_p for p in providers])
        return t_delta_p, providers
    
if __name__ == '__main__':
    
    settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025}
        
    market1 = Runner(**settings)
    print(market1.provider_array)