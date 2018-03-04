from unittest.mock import _side_effect_methods
from pyziabmc.orderbookc import price
cdef class ZITrader:
    cdef public str trader_id, trader_type
    cdef int _quantity, _quote_sequence
    cdef public list quote_collector
    
    cdef int _make_q(self, int maxq)
    cdef dict _make_add_quote(self, int time, str side, int price)
 
    
cdef class Provider(ZITrader):
    cdef float _delta
    cdef public int delta_p
    cdef public list cancel_collector
    cdef public dict local_book
    
    cdef dict _make_cancel_quote(self, dict q, int time)
    cpdef confirm_cancel_local(self, dict cancel_dict)
    cpdef confirm_trade_local(self, dict confirm)
    cpdef bulk_cancel(self, int time)
    cpdef process_signal(self, int time, dict qsignal, float q_provider, float lambda_t)
    cdef int _choose_price_from_exp(self, str buysell, int inside_price, float lambda_t)
    
    
cdef class Provider5(Provider):

    cdef int _choose_price_from_exp(self, str buysell, int inside_price, float lambda_t)
    
    
cdef class MarketMaker(Provider):
    cdef int _num_quotes, _quote_range, _position, _cash_flow
    cdef public list cash_flow_collector
    
    cpdef confirm_trade_local(self, dict confirm)
    cdef void _cumulate_cashflow(self, int timestamp)
    cpdef process_signal(self, int time, dict qsignal, float q_provider, float lambda_t)
    
    
cdef class MarketMaker5(MarketMaker):
    cdef list _p5ask, _p5bid
    
    cpdef process_signal(self, int time, dict qsignal, float q_provider, float lambda_t)
    
    
cdef class PennyJumper(ZITrader):
    cdef int _mpi
    cdef public list cancel_collector
    cdef dict _ask_quote, _bid_quote
    
    cdef dict _make_cancel_quote(self, dict q, int time)
    cpdef confirm_trade_local(self, dict confirm)
    cpdef process_signal(self, int time, dict qsignal, float q_taker)
    
    
cdef class Taker(ZITrader):
    cdef public int delta_t
    
    cpdef process_signal(self, int time, float q_taker)
    
    
cdef class InformedTrader(ZITrader):
    cdef public set delta_i
    cdef str _side
    cdef int _price
    
    cpdef process_signal(self, int time, float q_taker)
    cpdef make_delta_i(self, int run_steps, int informed_trades, int runlength)
    
