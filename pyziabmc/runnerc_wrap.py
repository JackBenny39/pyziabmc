import random
import time

import numpy as np

import pyziabmc.runnerc as runnerc

end = 6
trial_no = 80017
#h5_out = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\Trial %d\\ABMSmallCapSum.h5' % trial_no

settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
            'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
            'InformedTrader': False, 'informedMaxQ': 1, 'informedRunLength': 1, 'iMu': 0.005,
            'PennyJumper': False, 'AlphaPJ': 0.05,
            'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
            'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 1.0, 'Lambda0': 100}

if __name__ == '__main__':
    
    print(time.time())
    for j in range(6, 7):
        random.seed(j)
        np.random.seed(j)
    
        start = time.time()
    
        h5_file = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\TempTests\\mm1_cython_timertest_%d.h5' % j #(trial_no, j)
        
        market1 = runnerc.Runner(h5filename=h5_file, **settings)

        print('Run %d: %.2f minutes' % (j, (time.time() - start)/60))