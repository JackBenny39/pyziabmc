import random

import numpy as np

import pyziabmc.trader as trader

from pyziabmc.orderbook import Orderbook


class Runner(object):
    
    def __init__(self, h5filename='test.h5', mpi=1, prime1=20, run_steps=22, **kwargs):
        self.exchange = Orderbook()
        self.h5filename = h5filename
        self.mpi = mpi
        self.run_steps = run_steps + 1
        self.provider = kwargs.pop('Provider')
        if self.provider:
            self.t_delta_p, self.provider_array = self.buildProviders(kwargs['numProviders'], kwargs['providerMaxQ'],
                                                                      kwargs['pAlpha'], kwargs['pDelta'])
            self.q_provide = kwargs['qProvide']
        self.taker = kwargs.pop('Taker')
        if self.taker:
            self.t_delta_t, self.taker_array = self.buildTakers(kwargs['numTakers'], kwargs['takerMaxQ'], kwargs['tMu'])
        self.informed = kwargs.pop('InformedTrader')
        if self.informed:
            informedTrades = kwargs['iMu']*np.sum(self.run_steps/self.t_delta_t) if kwargs['Taker'] else 1/kwargs['iMu']
            self.informed_trader = self.buildInformedTrader(kwargs['informedMaxQ'], kwargs['informedRunLength'], informedTrades)
        self.pj = kwargs.pop('PennyJumper')
        if self.pj:
            self.pennyjumper = self.buildPennyJumper()
            self.alpha_pj = kwargs['AlphaPJ']
        self.marketmaker = kwargs.pop('MarketMaker')
        if self.marketmaker:
            self.t_delta_m, self.marketmakers = self.buildMarketMakers(kwargs['MMMaxQ'], kwargs['NumMMs'], kwargs['MMQuotes'], 
                                                                       kwargs['MMQuoteRange'], kwargs['MMDelta'])
        self.q_take, self.lambda_t = self.makeQTake(kwargs['QTake'], kwargs['Lambda0'], kwargs['WhiteNoise'], kwargs['CLambda'])
        self.liquidity_providers = self.makeLiquidityProviders()
        self.seedOrderbook()
        if self.provider:
            self.makeSetup(prime1, kwargs['Lambda0'])
        if self.pj:
            self.runMcsPJ(prime1)
        else:
            self.runMcs(prime1)
                  
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
        
    def makeLiquidityProviders(self):
        lp_dict = {}
        if self.provider:
            temp_dict = dict(zip([x.trader_id for x in self.provider_array], list(self.provider_array)))
            lp_dict.update(temp_dict)
        if self.marketmaker:
            temp_dict = dict(zip([x.trader_id for x in self.marketmakers], list(self.marketmakers)))
            lp_dict.update(temp_dict)
        if self.pj:
            lp_dict.update({'j0': self.pennyjumper})
        return lp_dict
    
    def seedOrderbook(self):
        seed_provider = trader.Provider('p999999', 1, 5, 0.05)
        self.liquidity_providers.update({'p999999': seed_provider})
        ba = random.choice(range(1000005, 1002001, 5))
        bb = random.choice(range(997995, 999996, 5))
        qask = {'order_id': 'p999999_a', 'timestamp': 0, 'type': 'add', 'quantity': 1, 'side': 'sell',
                'price': ba, 'exid': 99999999}
        qbid = {'order_id': 'p999999_b', 'timestamp': 0, 'type': 'add', 'quantity': 1, 'side': 'buy',
                'price': bb, 'exid': 99999999}
        seed_provider.local_book['p999999_a'] = qask
        self.exchange.add_order_to_book(qask)
        self.exchange.order_history.append(qask)
        seed_provider.local_book['p999999_b'] = qbid
        self.exchange.add_order_to_book(qbid)
        self.exchange.order_history.append(qbid)
        
    def makeSetup(self, prime1, lambda0):
        top_of_book = self.exchange.report_top_of_book(0)
        for current_time in range(1, prime1):
            for p in self.makeProviders(current_time):
                p.process_signal(current_time, top_of_book, self.q_provide, -lambda0)
                self.exchange.process_order(p.quote_collector[-1])
                top_of_book = self.exchange.report_top_of_book(current_time)
                
    def makeProviders(self, step):
        providers = self.provider_array[np.remainder(step, self.t_delta_p)==0]
        np.random.shuffle(providers)
        return providers
    
    def makeAll(self, step):
        trader_list = []
        if self.provider:
            providers_mask = np.remainder(step, self.t_delta_p)==0
            providers = np.vstack((self.provider_array, providers_mask)).T
            trader_list.append(providers)
        if self.taker:
            takers_mask = np.remainder(step, self.t_delta_t)==0
            takers = np.vstack((self.taker_array, takers_mask)).T
            trader_list.append(takers[takers_mask])
        if self.marketmaker:
            marketmakers_mask = np.remainder(step, self.t_delta_m)==0
            marketmakers = np.vstack((self.marketmakers, marketmakers_mask)).T
            trader_list.append(marketmakers)
        if self.informed:
            informed_mask = step in self.informed_trader.delta_i
            informed = np.squeeze(np.vstack((self.informed_trader, informed_mask)).T)
            trader_list.append(informed[informed_mask])
        all_traders = np.vstack(tuple(trader_list))
        np.random.shuffle(all_traders)
        return all_traders
    
    def runMcs(self, prime1):
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            for row in self.makeAll(current_time):
                if row[0].trader_type == 'Provider':
                    if row[1]:
                        row[0].process_signal(current_time, top_of_book, self.q_provide, self.lambda_t[current_time])
                        self.exchange.process_order(row[0].quote_collector[-1])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in row[0].cancel_collector:
                            self.exchange.process_order(c)
                            if self.exchange.confirm_modify_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                                row[0].confirm_cancel_local(self.exchange.confirm_modify_collector[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == 'MarketMaker':
                    if row[1]:
                        row[0].process_signal(current_time, top_of_book, self.q_provide)
                        for q in row[0].quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in row[0].cancel_collector:
                            self.exchange.process_order(c)
                            if self.exchange.confirm_modify_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                                row[0].confirm_cancel_local(self.exchange.confirm_modify_collector[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == 'InformedTrader':
                    row[0].process_signal(current_time)
                    self.exchange.process_order(row[0].quote_collector[-1])
                    if self.exchange.traded: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in self.exchange.confirm_trade_collector:
                            trader = self.liquidity_providers[c['trader']]
                            trader.confirm_trade_local(c)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    row[0].process_signal(current_time, self.q_take[current_time])
                    self.exchange.process_order(row[0].quote_collector[-1])
                    if self.exchange.traded: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in self.exchange.confirm_trade_collector:
                            trader = self.liquidity_providers[c['trader']]
                            trader.confirm_trade_local(c)
                        top_of_book = self.exchange.report_top_of_book(current_time)
            if not np.remainder(current_time, 2000):
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)
                
    def run_mcsPJ(self, prime1):
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            for row in self.makeAll(current_time):
                if row[0].trader_type == 'Provider':
                    if row[1]:
                        row[0].process_signal(current_time, top_of_book, self.q_provide, self.lambda_t[current_time])
                        self.exchange.process_order(row[0].quote_collector[-1])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in row[0].cancel_collector:
                            self.exchange.process_order(c)
                            if self.exchange.confirm_modify_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                                row[0].confirm_cancel_local(self.exchange.confirm_modify_collector[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == 'MarketMaker':
                    if row[1]:
                        row[0].process_signal(current_time, top_of_book, self.q_provide)
                        for q in row[0].quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    row[0].bulk_cancel(current_time)
                    if row[0].cancel_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in row[0].cancel_collector:
                            self.exchange.process_order(c)
                            if self.exchange.confirm_modify_collector: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                                row[0].confirm_cancel_local(self.exchange.confirm_modify_collector[0])
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif row[0].trader_type == 'InformedTrader':
                    row[0].process_signal(current_time)
                    self.exchange.process_order(row[0].quote_collector[-1])
                    if self.exchange.traded: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in self.exchange.confirm_trade_collector:
                            trader = self.trader_dict[c['trader']]
                            trader.confirm_trade_local(c)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    row[0].process_signal(current_time, self.q_take[current_time])
                    self.exchange.process_order(row[0].quote_collector[-1])
                    if self.exchange.traded: # <---- Check permission versus forgiveness here and elsewhere - move to methods?
                        for c in self.exchange.confirm_trade_collector:
                            trader = self.trader_dict[c['trader']]
                            trader.confirm_trade_local(c)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                if random.uniform(0,1) < self.alpha_pj:
                    self.pennyjumper.process_signal(current_time, top_of_book, self.q_take[current_time])
                    if self.pennyjumper.cancel_collector:
                        for c in self.pennyjumper.cancel_collector:
                            self.exchange.process_order(c)
                    if self.pennyjumper.quote_collector:
                        for q in self.pennyjumper.quote_collector:
                            self.exchange.process_order(q)
                    top_of_book = self.exchange.report_top_of_book(current_time)
            if not np.remainder(current_time, 2000):
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)
    
    
if __name__ == '__main__':
    
    settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
                'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
                'InformedTrader': False, 'informedMaxQ': 1, 'informedRunLength': 2, 'iMu': 0.01,
                'PennyJumper': False, 'AlphaPJ': 0.05,
                'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
                'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 1.0, 'Lambda0': 100}
        
    market1 = Runner(**settings)
    print(market1.exchange.report_top_of_book(21))