import pandas as pd

def loadData(fn):
     return pd.read_json(fn).to_dict(orient='list')

def outputResult(results):
    print('Method: {}'.format(results['Method']))
    print('AvgPrec@1: {} (std = {})'.format(results['AvgPrec@1'],
                                            results['StdPrec@1']))
    print('MAP: {} (std = {})'.format(results['MAP'], results['StdAP']))
    print('\n')