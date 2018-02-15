import random
import time

import numpy as np

import pyziabmc.runnerc as runnerc

if __name__ == '__main__':
    
    j = 5
    random.seed(j)
    np.random.seed(j)
    
    start = time.time()
    print(start)
    
    h5_root = 'mm1_cython_all_%d' % j
    h5dir = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\TempTests\\'
    h5_file = '%s%s.h5' % (h5dir, h5_root)
    
    settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
                'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
                'InformedTrader': False, 'informedMaxQ': 1, 'informedRunLength': 2, 'iMu': 0.01,
                'PennyJumper': True, 'AlphaPJ': 0.05,
                'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
                'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 1.0, 'Lambda0': 100}
        
    market1 = runnerc.Runner(h5filename=h5_file, **settings)

    print('Run 2: %.2f minutes' % ((time.time() - start)/60))