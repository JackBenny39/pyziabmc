import random
import time

import numpy as np
import pandas as pd

import pyziabmc.orderbook as orderbook
import pyziabmc.trader as trader

from pyziabmc.shared import Side, OType, TType


class Runner:
    
    def __init__(self, h5filename='test.h5', mpi=5, prime1=20, run_steps=100000, write_interval=5000, **kwargs):
        self.exchange = orderbook.Orderbook()
        self.h5filename = h5filename
        self.mpi = mpi
        self.run_steps = run_steps + 1
        self.liquidity_providers = {}
        self.provider = kwargs.pop('Provider')
        if self.provider:
            self.providers, self.num_providers = self.buildProviders(kwargs['numProviders'], kwargs['providerMaxQ'], 
                                                                     kwargs['pAlpha'], kwargs['pDelta'])
            self.q_provide = kwargs['qProvide']
        self.taker = kwargs.pop('Taker')
        if self.taker:
            self.takers = self.buildTakers(kwargs['numTakers'], kwargs['takerMaxQ'], kwargs['tMu'])
        self.informed = kwargs.pop('InformedTrader')
        if self.informed:
            if self.taker:
                takerTradeV = np.array([t.quantity*self.run_steps/t.delta_t for t in self.takers])
            informedTrades = np.int(kwargs['iMu']*np.sum(takerTradeV) if self.taker else 1/kwargs['iMu'])
            self.informed_trader = self.buildInformedTrader(kwargs['informedMaxQ'], kwargs['informedRunLength'], informedTrades, prime1)
        self.pj = kwargs.pop('PennyJumper')
        if self.pj:
            self.pennyjumper = self.buildPennyJumper()
            self.alpha_pj = kwargs['AlphaPJ']
        self.marketmaker = kwargs.pop('MarketMaker')
        if self.marketmaker:
            self.marketmakers = self.buildMarketMakers(kwargs['MMMaxQ'], kwargs['NumMMs'], kwargs['MMQuotes'], 
                                                       kwargs['MMQuoteRange'], kwargs['MMDelta'])
        self.traders, self.num_traders = self.makeAll()
        self.q_take, self.lambda_t = self.makeQTake(kwargs['QTake'], kwargs['Lambda0'], kwargs['WhiteNoise'], kwargs['CLambda'])
        self.seedOrderbook(kwargs['pAlpha'])
        if self.provider:
            self.makeSetup(prime1, kwargs['Lambda0'])
        if self.pj:
            self.runMcsPJ(prime1, write_interval)
        else:
            self.runMcs(prime1, write_interval)
        self.exchange.trade_book_to_h5(h5filename)
        self.qTakeToh5()
        self.mmProfitabilityToh5()
                  
    def buildProviders(self, numProviders, providerMaxQ, pAlpha, pDelta):
        ''' Providers id starts with 1
        '''
        provider_ids = [1000 + i for i in range(numProviders)]
        if self.mpi == 1:
            provider_list = [trader.Provider(p, providerMaxQ, pDelta, pAlpha) for p in provider_ids]
        else:
            provider_list = [trader.Provider5(p, providerMaxQ, pDelta, pAlpha) for p in provider_ids]
        self.liquidity_providers.update(dict(zip(provider_ids, provider_list)))
        return provider_list, len(provider_list)
    
    def buildTakers(self, numTakers, takerMaxQ, tMu):
        ''' Takers id starts with 2
        '''
        taker_ids = [2000 + i for i in range(numTakers)]
        return [trader.Taker(t, takerMaxQ, tMu) for t in taker_ids]
    
    def buildInformedTrader(self, informedMaxQ, informedRunLength, informedTrades, prime1):
        ''' Informed trader id starts with 5
        '''
        return trader.InformedTrader(5000, informedMaxQ, informedTrades, informedRunLength, prime1, self.run_steps)
    
    def buildPennyJumper(self):
        ''' PJ id starts with 4
        '''
        jumper = trader.PennyJumper(4000, 1, self.mpi)
        self.liquidity_providers.update({4000: jumper})
        return jumper

    def buildMarketMakers(self, mMMaxQ, numMMs, mMQuotes, mMQuoteRange, mMDelta):
        ''' MM id starts with 3
        '''
        marketmaker_ids = [3000 + i for i in range(numMMs)]
        if self.mpi == 1:
            marketmaker_list = [trader.MarketMaker(p, mMMaxQ, 0.005, mMDelta, mMQuotes, mMQuoteRange) for p in marketmaker_ids]
        else:
            marketmaker_list = [trader.MarketMaker5(p, mMMaxQ, 0.005, mMDelta, mMQuotes, mMQuoteRange) for p in marketmaker_ids]
        self.liquidity_providers.update(dict(zip(marketmaker_ids, marketmaker_list)))
        return marketmaker_list
    
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
            lambda_t = np.array([-lambda_0]*self.run_steps)
            return qt_take, lambda_t
    
    def makeAll(self):
        trader_list = []
        if self.provider:
            trader_list.extend(self.providers)
        if self.taker:
            trader_list.extend(self.takers)
        if self.marketmaker:
            trader_list.extend(self.marketmakers)
        if self.informed:
            trader_list.append(self.informed_trader)
        return trader_list, len(trader_list)
    
    def seedOrderbook(self, pAlpha):
        seed_provider = trader.Provider(9999, 1, 0.05, pAlpha)
        self.liquidity_providers.update({9999: seed_provider})
        ba = random.choice(range(1000005, 1002001, 5))
        bb = random.choice(range(997995, 999996, 5))
        qask = {'order_id': 1, 'trader_id': 9999, 'timestamp': 0, 'type': OType.ADD, 
                'quantity': 1, 'side': Side.ASK, 'price': ba}
        qbid = {'order_id': 2, 'trader_id': 9999, 'timestamp': 0, 'type': OType.ADD,
                'quantity': 1, 'side': Side.BID, 'price': bb}
        seed_provider.local_book[1] = qask
        self.exchange.add_order_to_book(qask)
        self.exchange.add_order_to_history(qask)
        seed_provider.local_book[2] = qbid
        self.exchange.add_order_to_book(qbid)
        self.exchange.add_order_to_history(qbid)
        
    def makeSetup(self, prime1, lambda0):
        top_of_book = self.exchange.report_top_of_book(0)
        for current_time in range(1, prime1):
            ps = random.sample(self.providers, self.num_providers)
            for p in ps:
                if not current_time % p.delta_t:
                    self.exchange.process_order(p.process_signal(current_time, top_of_book, self.q_provide, -lambda0))
                    top_of_book = self.exchange.report_top_of_book(current_time)    
    
    def doCancels(self, trader):
        for c in trader.cancel_collector:
            self.exchange.process_order(c)
                    
    def confirmTrades(self):
        for c in self.exchange.confirm_trade_collector:
            contra_side = self.liquidity_providers[c['trader']]
            contra_side.confirm_trade_local(c)
    
    def runMcs(self, prime1, write_interval):
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            traders = random.sample(self.traders, self.num_traders)
            for t in traders:
                if t.trader_type == TType.Provider:
                    if not current_time % t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time, top_of_book, self.q_provide, self.lambda_t[current_time]))
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    t.bulk_cancel(current_time)
                    if t.cancel_collector:
                        self.doCancels(t)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif t.trader_type == TType.MarketMaker:
                    if not current_time % t.quantity:
                        t.process_signal(current_time, top_of_book, self.q_provide)
                        for q in t.quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    t.bulk_cancel(current_time)
                    if t.cancel_collector:
                        self.doCancels(t)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif t.trader_type == TType.Taker:
                    if not current_time % t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time, self.q_take[current_time]))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    if current_time in t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
            if not current_time % write_interval:
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)
                
    def runMcsPJ(self, prime1, write_interval):
        top_of_book = self.exchange.report_top_of_book(prime1)
        for current_time in range(prime1, self.run_steps):
            traders = random.sample(self.traders, self.num_traders)
            for t in traders:
                if t.trader_type == TType.Provider:
                    if not current_time % t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time, top_of_book, self.q_provide, self.lambda_t[current_time]))
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    t.bulk_cancel(current_time)
                    if t.cancel_collector:
                        self.doCancels(t)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif t.trader_type == TType.MarketMaker:
                    if not current_time % t.quantity:
                        t.process_signal(current_time, top_of_book, self.q_provide)
                        for q in t.quote_collector:
                            self.exchange.process_order(q)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                    t.bulk_cancel(current_time)
                    if t.cancel_collector:
                        self.doCancels(t)
                        top_of_book = self.exchange.report_top_of_book(current_time)
                elif t.trader_type == TType.Taker:
                    if not current_time % t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time, self.q_take[current_time]))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
                else:
                    if current_time in t.delta_t:
                        self.exchange.process_order(t.process_signal(current_time))
                        if self.exchange.traded:
                            self.confirmTrades()
                            top_of_book = self.exchange.report_top_of_book(current_time)
                if random.random() < self.alpha_pj:
                    self.pennyjumper.process_signal(current_time, top_of_book, self.q_take[current_time])
                    if self.pennyjumper.cancel_collector:
                        for c in self.pennyjumper.cancel_collector:
                            self.exchange.process_order(c)
                    if self.pennyjumper.quote_collector:
                        for q in self.pennyjumper.quote_collector:
                            self.exchange.process_order(q)
                    top_of_book = self.exchange.report_top_of_book(current_time)
            if not current_time % write_interval:
                self.exchange.order_history_to_h5(self.h5filename)
                self.exchange.sip_to_h5(self.h5filename)
                
    def qTakeToh5(self):
        temp_df = pd.DataFrame({'qt_take': self.q_take, 'lambda_t': self.lambda_t})
        temp_df.to_hdf(self.h5filename, 'qtl', append=True, format='table', complevel=5, complib='blosc')
        
    def mmProfitabilityToh5(self):
        for m in self.marketmakers:
            temp_df = pd.DataFrame(m.cash_flow_collector)
            temp_df.to_hdf(self.h5filename, 'mmp', append=True, format='table', complevel=5, complib='blosc')
    
    
if __name__ == '__main__':
    
    print(time.time())
    
    settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
                'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
                'InformedTrader': False, 'informedMaxQ': 1, 'informedRunLength': 1, 'iMu': 0.005,
                'PennyJumper': False, 'AlphaPJ': 0.05,
                'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
                'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 10.0, 'Lambda0': 100}
    
    for j in range(51, 61):
        random.seed(j)
        np.random.seed(j)
    
        start = time.time()
        
        h5_root = 'python_2mpi5_%d' % j
        h5dir = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\Trial 901\\'
        h5_file = '%s%s.h5' % (h5dir, h5_root)
    
        market1 = Runner(h5filename=h5_file, **settings)

        print('Run %d: %.1f seconds' % (j, time.time() - start))