from libcpp.pair cimport pair


cdef enum Side:
    BID = 1
    ASK = 2
    
    
cdef enum OType:
    ADD = 1
    CANCEL = 2
    MODIFY = 3
    
    
cdef enum TType:
    ZITrader = 0
    Provider = 1
    MarketMaker = 2
    PennyJumper = 3
    Taker = 4
    Informed = 5
    

ctypedef struct Quote:
    int trader_id
    int order_id
    int timestamp
    int qty
    Side side
    int price
    
    
ctypedef struct Order:
    int trader_id
    int order_id
    int timestamp
    OType type
    int quantity
    Side side
    int price
    
    
ctypedef pair[int, Order] OneOrder    