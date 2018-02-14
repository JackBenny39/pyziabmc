# distutils: language = c++

import random

import numpy as np
cimport numpy as np


cdef class ZITrader:
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
        self.trader_id = name # trader id
        self.trader_type = 'ZITrader'
        self._quantity = self._make_q(maxq)
        self.quote_collector = []
        self._quote_sequence = 0
        
    def __repr__(self):
        return 'Trader({0}, {1})'.format(self.trader_id, self._quantity)
    
    cdef int _make_q(self, int maxq):
        '''Determine order size'''
        cdef np.ndarray default_arr = np.array([1, 5, 10, 25, 50], dtype=np.int)
        return random.choice(default_arr[default_arr<=maxq])
    
    cdef dict _make_add_quote(self, int time, str side, int price):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        cdef str order_id = '%s_%d' % (self.trader_id, self._quote_sequence)
        cdef str qtype = 'add'
        return {'order_id': order_id, 'timestamp': time, 'type': qtype, 'quantity': self._quantity, 
                'side': side, 'price': price}
        
        
cdef class Provider(ZITrader):
    '''
    Provider generates quotes (dicts) based on make probability.
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector, local_book
    Public methods: confirm_cancel_local, confirm_trade_local, process_signal, bulk_cancel
    '''
        
    def __init__(self, str name, int maxq, float delta, alpha=None):
        '''Provider has own delta; a local_book to track outstanding orders and a 
        cancel_collector to convey cancel messages to the exchange.
        '''
        ZITrader.__init__(self, name, maxq)
        self.trader_type = 'Provider'
        self._delta = delta
        if alpha is not None:
            '''np.int is faster than math.floor; 
               random.expovariate is faster than numpy.random.exponential (for size == 1)'''
            self.delta_p = np.int(random.expovariate(alpha) + 1)*self._quantity
        self.local_book = {}
        self.cancel_collector = []
                
    def __repr__(self):
        return 'Trader({0}, {1}, {2})'.format(self.trader_id, self._quantity, self.trader_type)
    
    cdef dict _make_cancel_quote(self, dict q, int time):
        return {'type': 'cancel', 'timestamp': time, 'order_id': q['order_id'], 'quantity': q['quantity'],
                'side': q['side'], 'price': q['price']}
        
    cpdef confirm_cancel_local(self, dict cancel_dict):
        del self.local_book[cancel_dict['order_id']]
        
    cpdef confirm_trade_local(self, dict confirm):
        cdef dict to_modify = self.local_book.get(confirm['order_id'])
        if confirm['quantity'] == to_modify['quantity']:
            self.confirm_cancel_local(to_modify)
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
          
    cpdef bulk_cancel(self, int time):
        '''bulk_cancel cancels _delta percent of outstanding orders'''
        cdef int i, lob
        cdef str idx
        cdef list order_keys
        cdef np.ndarray orders_to_delete
        self.cancel_collector.clear()
        lob = len(self.local_book)
        if lob > 0:
            order_keys = list(self.local_book.keys())
            orders_to_delete = np.random.ranf(lob)
            for i, idx in enumerate(order_keys):
                if orders_to_delete[i] < self._delta:
                    self.cancel_collector.append(self._make_cancel_quote(self.local_book.get(idx), time))
                    
    cpdef process_signal(self, int time, dict qsignal, float q_provider, float lambda_t):
        '''Provider buys or sells with probability related to q_provide'''
        cdef int price
        cdef str side, buysell
        cdef dict q
        self.quote_collector.clear()
        if random.uniform(0,1) < q_provider:
            buysell = 'bid'
            price = self._choose_price_from_exp(buysell, qsignal['best_ask'], lambda_t)
            side = 'buy'
        else:
            buysell = 'ask'
            price = self._choose_price_from_exp(buysell, qsignal['best_bid'], lambda_t)
            side = 'sell'
        q = self._make_add_quote(time, side, price)
        self.local_book[q['order_id']] = q
        self.quote_collector.append(q)
        
    cdef int _choose_price_from_exp(self, str buysell, int inside_price, float lambda_t):
        '''Prices chosen from an exponential distribution'''
        cdef int plug
        plug = np.int(lambda_t*np.log(np.random.rand()))
        if buysell == 'bid':
            return inside_price-1-plug
        else:
            return inside_price+1+plug
        
        
cdef class Provider5(Provider):
    '''
    Provider5 generates quotes (dicts) based on make probability.
    
    Subclass of Provider
    '''

    def __init__(self, str name, int maxq, float delta, float alpha):
        '''Provider has own delta; a local_book to track outstanding orders and a 
        cancel_collector to convey cancel messages to the exchange.
        '''
        Provider.__init__(self, name, maxq, delta, alpha=alpha)

    cdef int _choose_price_from_exp(self, str buysell, int inside_price, float lambda_t):
        '''Prices chosen from an exponential distribution'''
        cdef int plug
        plug = np.int(lambda_t*np.log(np.random.rand()))
        if buysell == 'bid':
            return np.int(5*np.floor((inside_price-1-plug)/5))
        else:
            return np.int(5*np.ceil((inside_price+1+plug)/5))
        
        
cdef class MarketMaker(Provider):
    '''
    MarketMaker generates a series of quotes near the inside (dicts) based on make probability.
    
    Subclass of Provider
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector (from Provider),
    cash_flow_collector
    Public methods: confirm_cancel_local (from Provider), confirm_trade_local, process_signal 
    '''

    def __init__(self, str name, int maxq, float delta, int num_quotes, int quote_range):
        '''_num_quotes and _quote_range determine the depth of MM quoting;
        _position and _cashflow are stored MM metrics
        '''
        Provider.__init__(self, name, maxq, delta)
        self.trader_type = 'MarketMaker'
        self._num_quotes = num_quotes
        self._quote_range = quote_range
        self._position = 0
        self._cash_flow = 0
        self.delta_p = self._quantity
        self.cash_flow_collector = []
                      
    def __repr__(self):
        return 'Trader({0}, {1}, {2}, {3})'.format(self.trader_id, self._quantity, self.trader_type, self._num_quotes)
    
    cpdef confirm_trade_local(self, dict confirm):
        '''Modify _cash_flow and _position; update the local_book'''
        cdef dict to_modify
        if confirm['side'] == 'buy':
            self._cash_flow -= confirm['price']*confirm['quantity']
            self._position += confirm['quantity']
        else:
            self._cash_flow += confirm['price']*confirm['quantity']
            self._position -= confirm['quantity']
        to_modify = self.local_book.get(confirm['order_id'])
        if confirm['quantity'] == to_modify['quantity']:
            self.confirm_cancel_local(to_modify)
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
        self._cumulate_cashflow(confirm['timestamp'])
        
    cdef void _cumulate_cashflow(self, int timestamp):
        self.cash_flow_collector.append({'mmid': self.trader_id, 'timestamp': timestamp, 'cash_flow': self._cash_flow,
                                         'position': self._position})
    
    cpdef process_signal(self, int time, dict qsignal, float q_provider, float lambda_t):
        '''
        MM chooses prices from a grid determined by the best prevailing prices.
        MM never joins the best price if it has size=1.
        ''' 
        cdef int max_bid_price, min_ask_price, price
        cdef np.ndarray prices
        cdef str side
        cdef dict q
        self.quote_collector.clear()
        if random.uniform(0,1) < q_provider:
            max_bid_price = qsignal['best_bid'] if qsignal['bid_size'] > 1 else qsignal['best_bid'] - 1
            prices = np.random.choice(range(max_bid_price-self._quote_range+1, max_bid_price+1), size=self._num_quotes)
            side = 'buy'
        else:
            min_ask_price = qsignal['best_ask'] if qsignal['ask_size'] > 1 else qsignal['best_ask'] + 1
            prices = np.random.choice(range(min_ask_price, min_ask_price+self._quote_range), size=self._num_quotes)
            side = 'sell'
        for price in prices:
            q = self._make_add_quote(time, side, price)
            self.local_book[q['order_id']] = q
            self.quote_collector.append(q)
            
            
cdef class MarketMaker5(MarketMaker):
    '''
    MarketMaker5 generates a series of quotes near the inside (dicts) based on make probability.
    
    Subclass of MarketMaker
    Public methods: process_signal 
    '''
    
    def __init__(self, str name, int maxq, float delta, int num_quotes, int quote_range):
        '''
        _num_quotes and _quote_range determine the depth of MM quoting;
        _position and _cashflow are stored MM metrics
        '''
        MarketMaker.__init__(self, name, maxq, delta, num_quotes, quote_range)
        self._p5ask = [1/20, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/30]
        self._p5bid = [1/30, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/20]
               
    cpdef process_signal(self, int time, dict qsignal, float q_provider, float lambda_t):
        '''
        MM chooses prices from a grid determined by the best prevailing prices.
        MM never joins the best price if it has size=1.
        ''' 
        cdef int max_bid_price, min_ask_price, price
        cdef np.ndarray prices
        cdef str side
        cdef dict q
        self.quote_collector.clear()
        if random.uniform(0,1) < q_provider:
            max_bid_price = qsignal['best_bid'] if qsignal['bid_size'] > 1 else qsignal['best_bid'] - 5
            prices = np.random.choice(range(max_bid_price-self._quote_range, max_bid_price+1, 5), size=self._num_quotes, p=self._p5bid)
            side = 'buy'
        else:
            min_ask_price = qsignal['best_ask'] if qsignal['ask_size'] > 1 else qsignal['best_ask'] + 5
            prices = np.random.choice(range(min_ask_price, min_ask_price+self._quote_range+1, 5), size=self._num_quotes, p=self._p5ask)
            side = 'sell'
        for price in prices:
            q = self._make_add_quote(time, side, price)
            self.local_book[q['order_id']] = q
            self.quote_collector.append(q)
            
            
cdef class PennyJumper(ZITrader):
    '''
    PennyJumper jumps in front of best quotes when possible
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector
    Public methods: confirm_trade_local (from ZITrader)
    '''
    
    def __init__(self, str name, int maxq, int mpi):
        '''
        Initialize PennyJumper
        
        cancel_collector is a public container for carrying cancel messages to the exchange
        PennyJumper tracks private _ask_quote and _bid_quote to determine whether it is alone
        at the inside or not.
        '''
        ZITrader.__init__(self, name, maxq)
        self.trader_type = 'PennyJumper'
        self._mpi = mpi
        self.cancel_collector = []
        self._ask_quote = None
        self._bid_quote = None
        
    def __repr__(self):
        return 'Trader({0}, {1}, {2}, {3})'.format(self.trader_id, self._quantity, self._mpi, self.trader_type)
    
    cdef dict _make_cancel_quote(self, dict q, int time):
        return {'type': 'cancel', 'timestamp': time, 'order_id': q['order_id'], 'quantity': q['quantity'],
                'side': q['side'], 'price': q['price']}

    cpdef confirm_trade_local(self, dict confirm):
        '''PJ has at most one bid and one ask outstanding - if it executes, set price None'''
        if confirm['side'] == 'buy':
            self._bid_quote = None
        else:
            self._ask_quote = None
            
    cpdef process_signal(self, int time, dict qsignal, float q_taker):
        '''PJ determines if it is alone at the inside, cancels if not and replaces if there is an available price 
        point inside the current quotes.
        '''
        cdef int price
        cdef str side
        cdef dict q
        self.quote_collector.clear()
        self.cancel_collector.clear()
        if qsignal['best_ask'] - qsignal['best_bid'] > self._mpi:
            # q_taker > 0.5 implies greater probability of a buy order; PJ jumps the bid
            if random.uniform(0,1) < q_taker:
                if self._bid_quote: # check if not alone at the bid
                    if self._bid_quote['price'] < qsignal['best_bid'] or self._bid_quote['quantity'] < qsignal['bid_size']:
                        self.cancel_collector.append(self._make_cancel_quote(self._bid_quote, time))
                        self._bid_quote = None
                if not self._bid_quote:
                    price = qsignal['best_bid'] + self._mpi
                    side = 'buy'
                    q = self._make_add_quote(time, side, price)
                    self.quote_collector.append(q)
                    self._bid_quote = q
            else:
                if self._ask_quote: # check if not alone at the ask
                    if self._ask_quote['price'] > qsignal['best_ask'] or self._ask_quote['quantity'] < qsignal['ask_size']:
                        self.cancel_collector.append(self._make_cancel_quote(self._ask_quote, time))
                        self._ask_quote = None
                if not self._ask_quote:
                    price = qsignal['best_ask'] - self._mpi
                    side = 'sell'
                    q = self._make_add_quote(time, side, price)
                    self.quote_collector.append(q)
                    self._ask_quote = q
        else: # spread = mpi
            if self._bid_quote: # check if not alone at the bid
                if self._bid_quote['price'] < qsignal['best_bid'] or self._bid_quote['quantity'] < qsignal['bid_size']:
                    self.cancel_collector.append(self._make_cancel_quote(self._bid_quote, time))
                    self._bid_quote = None
            if self._ask_quote: # check if not alone at the ask
                if self._ask_quote['price'] > qsignal['best_ask'] or self._ask_quote['quantity'] < qsignal['ask_size']:
                    self.cancel_collector.append(self._make_cancel_quote(self._ask_quote, time))
                    self._ask_quote = None


cdef class Taker(ZITrader):
    '''
    Taker generates quotes (dicts) based on take probability.
        
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal 
    '''
    def __init__(self, str name, int maxq, float mu):
        ZITrader.__init__(self, name, maxq)
        self.trader_type = 'Taker'
        self.delta_t = np.int(random.expovariate(mu) + 1)*self._quantity
        
    def __repr__(self):
        return 'Trader({0}, {1}, {2})'.format(self.trader_id, self._quantity, self.trader_type)
        
    cpdef process_signal(self, int time, float q_taker):
        '''Taker buys or sells with 50% probability.'''
        cdef int price
        cdef str side
        cdef dict q
        self.quote_collector.clear()
        if random.uniform(0,1) < q_taker: # q_taker > 0.5 implies greater probability of a buy order
            price = 2000000 # agent buys at max price (or better)
            side = 'buy'
        else:
            price = 0 # agent sells at min price (or better)
            side = 'sell'
        q = self._make_add_quote(time, side, price)
        self.quote_collector.append(q)
        
        
cdef class InformedTrader(ZITrader):
    '''
    InformedTrader generates quotes (dicts) based upon a fixed direction
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal
    '''
    
    def __init__(self, str name, int maxq, int run_steps, int informed_trades, int runlength):
        ZITrader.__init__(self, name, maxq)
        self.trader_type = 'InformedTrader'
        self.delta_i = self.make_delta_i(run_steps, informed_trades, runlength)
        self._side = np.random.choice(['buy', 'sell'])
        self._price = 0 if self._side == 'sell' else 2000000
        
    def __repr__(self):
        return 'Trader({0}, {1}, {2})'.format(self.trader_id, self._quantity, self.trader_type)
        
    cpdef process_signal(self, int time, dict qsignal, float q_provider, float lambda_t):
        '''InformedTrader buys or sells pre-specified attribute.'''
        cdef dict q
        q = self._make_add_quote(time, self._side, self._price)
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
        return set(t_delta_i)
    