import numpy as np

import pyziabmc.trader as trader

from pyziabmc.orderbook import Orderbook


class Runner(object):
    
    def __init__(self, mpi=1, run_steps=100000, **kwargs):
                #num_mms=1, mm_maxq=1, mm_quotes=12, mm_quote_range=60, mm_delta=0.025,
                #delta=0.025, lambda0=100, wn=0.001, c_lambda=1.0,
                #h5filename='test.h5'):
        self.exchange = Orderbook()
        self.mpi = mpi
        self.run_steps = run_steps
        if kwargs['Provider']:
            self.t_delta_p, self.provider_array = self.buildProviders(kwargs['numProviders'], kwargs['providerMaxQ'],
                                                                      kwargs['pAlpha'], kwargs['pDelta'])
            self.q_provide = kwargs['qProvide']
        if kwargs['Taker']:
            self.t_delta_t, self.taker_array = self.buildTakers(kwargs['numTakers'], kwargs['takerMaxQ'], 
                                                                kwargs['tMu'])
        if kwargs['InformedTrader']:
            informedTrades = kwargs['iMu']*np.sum(self.run_steps/self.t_delta_t) if kwargs['Taker'] else 1/kwargs['iMu']
            self.informed_trader = self.buildInformedTrader(kwargs['informedMaxQ'], kwargs['informedRunLength'], 
                                                            informedTrades)
        if kwargs['PennyJumper']:
            self.pennyjumper = self.buildPennyJumper()
            self.alpha_pj = kwargs['AlphaPJ']
                  
    def buildProviders(self, numProviders, providerMaxQ, pAlpha, pDelta):
        providers_list = ['p%i' % i for i in range(numProviders)]
        if self.mpi==1:
            providers = np.array([trader.Provider(p,providerMaxQ,pDelta,pAlpha) for p in providers_list])
        else:
            providers = np.array([trader.Provider5(p,providerMaxQ,pDelta,pAlpha) for p in providers_list])
        t_delta_p = np.array([p.delta_p for p in providers])
        return t_delta_p, providers
    
    def buildTakers(self, numTakers, takerMaxQ, tMu):
        takers_list = ['t%i' % i for i in range(numTakers)]
        takers = np.array([trader.Taker(t, takerMaxQ, tMu) for t in takers_list])
        t_delta_t = np.array([t.delta_t for t in takers])
        return t_delta_t, takers
    
    def buildInformedTrader(self, informedMaxQ, informedRunLength, informedTrades):
        return trader.InformedTrader('i0', informedMaxQ, self.run_steps, informedTrades, informedRunLength)
    
    def buildPennyJumper(self):
        return trader.PennyJumper('j0', 1, self.mpi)
    
    
if __name__ == '__main__':
    
    settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
                'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
                'InformedTrader': True, 'informedMaxQ': 1, 'informedRunLength': 2, 'iMu': 0.01,
                'PennyJumper': True, 'AlphaPJ': 0.05}
        
    market1 = Runner(**settings)
    print(market1.informed_trader, market1.informed_trader.delta_i)