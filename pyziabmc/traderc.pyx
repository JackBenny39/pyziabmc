cimport cython
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
        
    def __init__(self, str name, int maxq, float delta, float alpha=None):
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
        
    cdef void confirm_cancel_local(self, dict cancel_dict):
        del self.local_book[cancel_dict['order_id']]
        
    cpdef void confirm_trade_local(self, dict confirm):
        cdef dict to_modify = self.local_book.get(confirm['order_id'])
        if confirm['quantity'] == to_modify['quantity']:
            self.confirm_cancel_local(to_modify)
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
    
    @cython.boundscheck(False)         
    cpdef void bulk_cancel(self, int time):
        '''bulk_cancel cancels _delta percent of outstanding orders'''
        cdef unsigned int lob, idx
        cdef list order_keys
        cdef np.ndarray[np.int_t, ndim=1, negative_indices=False, mode='c'] orders_to_delete
        self.cancel_collector.clear()
        lob = len(self.local_book)
        if lob > 0:
            order_keys = list(self.local_book.keys())
            orders_to_delete = np.random.ranf(lob)
            for idx in range(lob):
                if orders_to_delete[idx] < self._delta:
                    self.cancel_collector.append(self._make_cancel_quote(self.local_book.get(order_keys[idx]), time))
                    
    cpdef void process_signal(self, int time, dict qsignal, float q_provider, float lambda_t):
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



        