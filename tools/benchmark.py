import numpy as np
from sklearn import metrics
import pandas as pd

def auc(test_df):
    y = test_df['toxicity']
    pred = test_df['prediction']
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return metrics.auc(fpr, tpr)
    
def Mp(data, p=-5.0):
    return np.average(data ** p) ** (1/p)
    
def get_jigsaw_score(pred, test_df):
    test_df['prediction'] = pred
    test_df['toxicity'] = test_df['toxicity'] >= 0.5
    test_df['bool_pred'] = test_df['prediction'] >= 0.5
    
    overall = auc(test_df)
    
    groups = ['black', 'white', 'male', 'female',
          'christian', 'jewish', 'muslim',
          'psychiatric_or_mental_illness',
          'homosexual_gay_or_lesbian']

    categories = pd.DataFrame(columns = ['SUB', 'BPSN', 'BNSP'], index = groups)


    for group in groups:
        test_df[group] = test_df[group] >= 0.5
        categories.loc[group,'SUB'] = auc(test_df[test_df[group]])
        bpsn = ((~test_df[group] & test_df['toxicity'])    #background positive
            | (test_df[group] & ~test_df['toxicity'])) #subgroup negative
        categories.loc[group,'BPSN'] = auc(test_df[bpsn])
        bnsp = ((~test_df[group] & ~test_df['toxicity'])   #background negative
            | (test_df[group] & test_df['toxicity']))  #subgrooup positive
        categories.loc[group,'BNSP'] = auc(test_df[bnsp])

    categories.loc['Mp',:] = categories.apply(Mp, axis= 0)
    leaderboard = (np.sum(categories.loc['Mp',:]) + overall) / 4
    
    
    categories = categories.reset_index()
    categories[['SUB', 'BPSN', 'BNSP']] = categories[['SUB', 'BPSN', 'BNSP']].astype(float).round(4)
    categories = categories.rename(columns={'index':'subgroup'})

    return categories,leaderboard




    