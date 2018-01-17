import math
import random

import numpy as np


class ZITrader:
    '''
    ZITrader generates quotes (dicts) based on mechanical probabilities.
    
    A general base class for specific trader types.
    Public attributes: quote_collector
    Public methods: none
    '''

    def __init__(self, name, maxq):
        '''
        Initialize ZITrader with some base class attributes and a method
        
        quote_collector is a public container for carrying quotes to the exchange
        '''
        self._trader_id = name # trader id
        self._quantity = self._make_q(maxq)
        self.quote_collector = []
        self._quote_sequence = 0
        
    def __repr__(self):
        return 'Trader({0}, {1})'.format(self._trader_id, self._quantity)
    
    def _make_q(self, maxq):
        '''Determine order size'''
        default_arr = np.array([1, 5, 10, 25, 50])
        return random.choice(default_arr[default_arr<=maxq])
    
    def _make_add_quote(self, time, side, price):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        order_id = '%s_%d' % (self._trader_id, self._quote_sequence)
        return {'order_id': order_id, 'timestamp': time, 'type': 'add', 'quantity': self._quantity, 
                'side': side, 'price': price}
        
        
class Provider(ZITrader):
    '''
    Provider generates quotes (dicts) based on make probability.
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector, local_book
    Public methods: confirm_cancel_local, confirm_trade_local, process_signal, bulk_cancel
    '''
        
    def __init__(self, name, maxq, delta, alpha):
        '''Provider has own delta; a local_book to track outstanding orders and a 
        cancel_collector to convey cancel messages to the exchange.
        '''
        ZITrader.__init__(self, name, maxq)
        self.trader_type = 'Provider'
        self._delta = delta
        self.delta_p = math.floor(random.expovariate(alpha) + 1)*self._quantity
        self.local_book = {}
        self.cancel_collector = []
                
    def __repr__(self):
        return 'Trader({0}, {1}, {2})'.format(self._trader_id, self._quantity, self.trader_type)
    
    def _make_cancel_quote(self, q, time):
        return {'type': 'cancel', 'timestamp': time, 'order_id': q['order_id'], 'quantity': q['quantity'],
                'side': q['side'], 'price': q['price']}
        
    def confirm_cancel_local(self, cancel_dict):
        del self.local_book[cancel_dict['order_id']]

    def confirm_trade_local(self, confirm):
        to_modify = self.local_book.get(confirm['order_id'], "WTF???")
        if confirm['quantity'] == to_modify['quantity']:
            self.confirm_cancel_local(to_modify)
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
            
    def bulk_cancel(self, time):
        '''bulk_cancel cancels _delta percent of outstanding orders'''
        self.cancel_collector.clear()
        lob = len(self.local_book)
        if lob > 0:
            order_keys = list(self.local_book.keys())
            orders_to_delete = np.random.ranf(lob)
            for idx in range(lob):
                if orders_to_delete[idx] < self._delta:
                    self.cancel_collector.append(self._make_cancel_quote(self.local_book.get(order_keys[idx]), time))

    def process_signal(self, time, qsignal, q_provider, lambda_t):
        '''Provider buys or sells with probability related to q_provide'''
        self.quote_collector.clear()
        if np.random.uniform(0,1) < q_provider:
            price = self._choose_price_from_exp('bid', qsignal['best_ask'], lambda_t)
            side = 'buy'
        else:
            price = self._choose_price_from_exp('ask', qsignal['best_bid'], lambda_t)
            side = 'sell'
        q = self._make_add_quote(time, self._quantity, side, price)
        self.local_book[q['order_id']] = q
        self.quote_collector.append(q)            
      
    def _choose_price_from_exp(self, side, inside_price, lambda_t):
        '''Prices chosen from an exponential distribution'''
        # make pricing explicit for now. Logic scales for other mpi.
        plug = np.int(lambda_t*np.log(np.random.rand()))
        if side == 'bid':
            price = inside_price-1-plug
        else:
            price = inside_price+1+plug
        return price
    
    
class Provider5(Provider):
    '''
    Provider5 generates quotes (dicts) based on make probability.
    
    Subclass of Provider
    '''

    def __init__(self, name, maxq, delta, alpha):
        '''Provider has own delta; a local_book to track outstanding orders and a 
        cancel_collector to convey cancel messages to the exchange.
        '''
        Provider.__init__(self, name, maxq, delta, alpha)

    def _choose_price_from_exp(self, side, inside_price, lambda_t):
        '''Prices chosen from an exponential distribution'''
        plug = np.int(lambda_t*np.log(np.random.rand()))
        if side == 'bid':
            price = np.int(5*np.floor((inside_price-1-plug)/5))
        else:
            price = np.int(5*np.ceil((inside_price+1+plug)/5))
        return price
    
    
class Taker(ZITrader):
    '''
    Taker generates quotes (dicts) based on take probability.
        
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal 
    '''

    def __init__(self, name, maxq, mu):
        ZITrader.__init__(self, name, maxq)
        self.trader_type = 'Taker'
        self.delta_t = math.floor(random.expovariate(mu) + 1)*self._quantity
        
    def __repr__(self):
        return 'Trader({0}, {1}, {2})'.format(self._trader_id, self._quantity, self.trader_type)
        
    def process_signal(self, time, q_taker):
        '''Taker buys or sells with 50% probability.'''
        self.quote_collector.clear()
        if random.uniform(0,1) < q_taker: # q_taker > 0.5 implies greater probability of a buy order
            price = 2000000 # agent buys at max price (or better)
            side = 'buy'
        else:
            price = 0 # agent sells at min price (or better)
            side = 'sell'
        q = self._make_add_quote(time, self._quantity, side, price)
        self.quote_collector.append(q)
        
        
class InformedTrader(Taker):
    '''
    InformedTrader generates quotes (dicts) based upon a fixed direction
    
    Subclass of Taker
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal
    '''
    
    def __init__(self, name, maxq, run_steps, informed_trades, runlength):
        ZITrader.__init__(self, name, maxq)
        self.trader_type = 'InformedTrader'
        self.delta_i = self.make_delta_i(run_steps, informed_trades, runlength)
        self._side = np.random.choice(['buy', 'sell'])
        self._price = 0 if self._side == 'sell' else 2000000
        
    def __repr__(self):
        return 'Trader({0}, {1}, {2})'.format(self._trader_id, self._quantity, self.trader_type)
        
    def process_signal(self, time):
        '''InformedTrader buys or sells pre-specified attribute.'''
        q = self._make_add_quote(time, self._quantity, self._side, self._price)
        self.quote_collector.append(q)
        
    def make_delta_i(self, run_steps, informed_trades, runlength):
        t_delta_i = np.random.choice(run_steps, size=np.int(informed_trades/(runlength*self._quantity)), replace=False)
        if runlength > 1:
            stack1 = t_delta_i
            s_length = len(t_delta_i)
            for i in range(1, runlength):
                temp = t_delta_i+i
                stack2 = np.unique(np.hstack((stack1, temp)))
                repeats = (i+1)*s_length - len(set(stack2))
                new_choice_set = set(range(run_steps)) - set(stack2)
                extras = np.random.choice(list(new_choice_set), size=repeats, replace=False)
                stack1 = np.hstack((stack2, extras))
            t_delta_i = stack1
        return np.sort(t_delta_i)

    