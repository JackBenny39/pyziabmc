import numpy as np

import pyziabmc.trader as trader

from pyziabmc.orderbook import Orderbook


class Runner(object):
    
    def __init__(self, h5filename='test.h5', mpi=1, run_steps=100000, **kwargs):
                #lambda0=100
        self.exchange = Orderbook()
        self.h5filename = h5filename
        self.mpi = mpi
        self.run_steps = run_steps + 1
        if kwargs['Provider']:
            self.t_delta_p, self.provider_array = self.buildProviders(kwargs['numProviders'], kwargs['providerMaxQ'],
                                                                      kwargs['pAlpha'], kwargs['pDelta'])
            self.q_provide = kwargs['qProvide']
        if kwargs['Taker']:
            self.t_delta_t, self.taker_array = self.buildTakers(kwargs['numTakers'], kwargs['takerMaxQ'], kwargs['tMu'])
        if kwargs['InformedTrader']:
            informedTrades = kwargs['iMu']*np.sum(self.run_steps/self.t_delta_t) if kwargs['Taker'] else 1/kwargs['iMu']
            self.informed_trader = self.buildInformedTrader(kwargs['informedMaxQ'], kwargs['informedRunLength'], informedTrades)
        if kwargs['PennyJumper']:
            self.pennyjumper = self.buildPennyJumper()
            self.alpha_pj = kwargs['AlphaPJ']
        if kwargs['MarketMaker']:
            self.t_delta_m, self.marketmakers = self.buildMarketMakers(kwargs['MMMaxQ'], kwargs['NumMMs'], kwargs['MMQuotes'], 
                                                                       kwargs['MMQuoteRange'], kwargs['MMDelta'])
        self.q_take, self.lambda_t = self.makeQTake(kwargs['QTake'], kwargs['Lambda0'], kwargs['WhiteNoise'], kwargs['CLambda'])
                  
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

    def buildMarketMakers(self, mMMaxQ, numMMs, mMQuotes, mMQuoteRange, mMDelta):
        marketmakers_list = ['m%i' % i for i in range(numMMs)]
        if self.mpi==1:
            marketmakers = np.array([trader.MarketMaker(p, mMMaxQ, mMDelta, mMQuotes, mMQuoteRange) for p in marketmakers_list])
        else:
            marketmakers = np.array([trader.MarketMaker5(p, mMMaxQ, mMDelta, mMQuotes, mMQuoteRange) for p in marketmakers_list])
        t_delta_m = np.array([m.delta_p for m in marketmakers])
        return t_delta_m, marketmakers
    
    def makeQTake(self, q_take, lambda_0, wn, c_lambda):
        if q_take:
            noise = np.random.rand(2, self.run_steps)
            qt_take = np.empty_like(noise)
            qt_take[:,0] = 0.5
            for i in range(1, self.run_steps):
                qt_take[:,i] = qt_take[:,i-1] + (noise[:,i-1]>qt_take[:,i-1])*wn - (noise[:,i-1]<qt_take[:,i-1])*wn
            lambda_t = -lambda_0*(1 + (np.abs(qt_take[1] - 0.5)/np.sqrt(np.mean(np.square(qt_take[0] - 0.5))))*c_lambda)
            return qt_take[1], lambda_t
        else:
            qt_take = np.array([0.5]*self.run_steps)
            lambda_t = -lambda_0
            return qt_take, lambda_t
    
    
if __name__ == '__main__':
    
    settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
                'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
                'InformedTrader': True, 'informedMaxQ': 1, 'informedRunLength': 2, 'iMu': 0.01,
                'PennyJumper': True, 'AlphaPJ': 0.05,
                'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
                'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 1.0, 'Lambda0': 100}
        
    market1 = Runner(**settings)
    print(market1.q_take, market1.lambda_t)