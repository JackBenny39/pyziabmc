from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from pyziabmc.sharedc cimport Side, TType, Order, Quote, OneOrder

import numpy as np
cimport numpy as np

ctypedef unordered_map[int, Order] LocalBook

ctypedef vector[Order*] OrderV
ctypedef vector[Order] OrderS

cdef TType trader_type


cdef class ZITrader:

    cdef public int trader_id, quantity
    cdef int _quote_sequence
    cdef OrderV quote_collector
    
    cdef int _make_q(self, int maxq)
    cdef Order _make_add_quote(self, int time, Side side, int price)
 
    
cdef class Provider(ZITrader):

    cdef public int delta_t
    cdef double _delta
    cdef OrderS cancel_collector
    cdef LocalBook local_book
    
    cdef int _make_delta(self, double pAlpha)
    cdef Order _make_cancel_quote(self, Order &q, int time)
    cpdef confirm_trade_local(self, Quote &confirm)
    cpdef bulk_cancel(self, int time)
    cpdef Order process_signalp(self, int time, dict qsignal, double q_provider, double lambda_t)
    cdef int _choose_price_from_exp(self, Side side, int inside_price, double lambda_t)
    
    
cdef class Provider5(Provider):

    cdef int _choose_price_from_exp(self, Side side, int inside_price, double lambda_t)
    
    
cdef class MarketMaker(Provider):
    
    cdef int _num_quotes, _quote_range, _position, _cash_flow
    cdef public list cash_flow_collector
    
    cpdef confirm_trade_local(self, Quote &confirm)
    cdef void _cumulate_cashflow(self, int timestamp)
    cdef void process_signalm(self, int time, dict qsignal, double q_provider)
    
    
cdef class MarketMaker5(MarketMaker):
    
    cdef np.ndarray _p5ask, _p5bid
    
    cdef void process_signalm(self, int time, dict qsignal, double q_provider)
    
    
cdef class PennyJumper(ZITrader):
    
    cdef int _mpi
    cdef OrderS cancel_collector
    cdef bint _has_ask, _has_bid
    cdef Order _ask_quote, _bid_quote
    
    cdef dict _make_cancel_quote(self, Order &q, int time)
    cpdef confirm_trade_local(self, Quote &confirm)
    cdef void process_signalj(self, int time, dict qsignal, double q_taker)
    
    
cdef class Taker(ZITrader):

    cdef public int delta_t
    
    cdef int _make_delta(self, double tMu)
    cpdef Order process_signalt(self, int time, double q_taker)
    
    
cdef class InformedTrader(ZITrader):
    
    cdef Side _side
    cdef int _price
    cdef public set delta_t
    
    cdef set _make_delta(self, int informedTrades, int informedRunLength, int start, int stop)
    cpdef Order process_signali(self, int time)
    