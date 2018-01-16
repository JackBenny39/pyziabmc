import unittest

from pyziabmc.trader import ZITrader


class TestTrader(unittest.TestCase):
    
    def setUp(self):
        self.z1 = ZITrader('z1', 5)
        
# ZITrader tests
    
    def test_make_q(self):
        self.assertLessEqual(self.z1._quantity, 5)
    
    def test_make_add_quote(self):
        time = 1
        side = 'sell'
        price = 125
        q = self.z1._make_add_quote(time, side, price)
        expected = {'order_id': 'z1_1', 'timestamp': 1, 'type': 'add', 'quantity': self.z1._quantity, 'side': 'sell', 
                    'price': 125}
        self.assertDictEqual(q, expected)
