import numpy as np

def precision_at_k(y_pred, y, k=1):
    tp = .0
    for i in range(k):
        tp += (y_pred[i] in set(y))

    return tp / k


def average_precision(y_pred, y):
    precisions = list()
    tp = .0
    for i in range(len(y_pred)):
        if (y_pred[i] in set(y)):
            tp += 1
            precisions.append(tp / (i + 1))

    return np.mean(precisions)


def get_results(pred_scores, y, method):
    results = {'Method': method, 'Prec@1': [], 'Prec@5': [], 'Prec@10': [],
               'AvgPrec': [], 'MAP': 0}

    for i in range(len(pred_scores)):
        results['Prec@1'].append(precision_at_k(pred_scores[i], y[i], k=1))
        results['AvgPrec'].append(average_precision(pred_scores[i], y[i]))

    # evaluation (MAP - mean average precision)
    results['MAP'] = np.mean(results['AvgPrec'])
    results['StdAP'] = np.std(results['AvgPrec'])
    results['AvgPrec@1'] = np.mean(results['Prec@1'])
    results['StdPrec@1'] = np.std(results['Prec@1'])

    return results