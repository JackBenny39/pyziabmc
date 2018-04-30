cdef class Orderbook:
    cdef list _bid_book_prices, _ask_book_prices, _sip_collector
    cdef public list order_history, confirm_modify_collector, confirm_trade_collector, trade_book
    cdef dict _bid_book, _ask_book, _lookup
    cdef int _order_index, _ex_index
    cdef public bint traded
    
    cdef void _add_order_to_history(self, dict order)
    cpdef add_order_to_book(self, dict order)
    cdef void _add_order_to_lookup(self, int trader_id, int order_id, int ex_id)
    cdef void _remove_order(self, str order_side, int order_price, str order_id)
    cdef void _modify_order(self, str order_side, int order_quantity, str order_id, int order_price)
    cdef void _add_trade_to_book(self, str resting_order_id, int resting_timestamp, str incoming_order_id, 
                                 int timestamp, int price, int quantity, str side)
    cdef void _confirm_trade(self, int timestamp, str order_side, int order_quantity, str order_id, int order_price)
    cdef void _confirm_modify(self, int timestamp, str order_side, int order_quantity, str order_id)
    cpdef process_order(self, dict order)
    cdef void _match_trade(self, dict order)
    cpdef report_top_of_book(self, int now_time)