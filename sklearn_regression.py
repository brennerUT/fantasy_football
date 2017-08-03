import pandas as pd
import os
import sklearn.linear_model as linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

def ridge_pipeline(df,df_2017):
    x_unscaled=df.filter(regex=('.*_proj_pts'))
    x=StandardScaler().fit_transform(x_unscaled)
    y=df.actual_points
    param_grid={'alpha':[0.0001,0.001,0.1,1,10,100,500,1000]}
    grid = GridSearchCV(linear_model.Ridge(), param_grid, cv=10,n_jobs=-1)
    grid.fit(x, y)
    best_alph=grid.best_params_['alpha']
    if best_alph in [0.0001,1000]:
        return 'WARNING: best_alph at endpoint'+': '+str(best_alph)
    else:
        ridge=linear_model.Ridge(alpha=best_alph).fit(x,y)
        x_2017=df_2017.loc[:,list(x_unscaled)]
        x_2017=StandardScaler().fit_transform(x_2017)
        df_2017['my_proj']=ridge.predict(x_2017)
        return df_2017.loc[:,['player','position','my_proj']].sort_values('my_proj',ascending=False)
df_2017=pd.read_csv('2017_projections.csv')
masterDf=pd.read_csv('2016_actual_and_proj.csv')
d={}
for pos in ['DST','QB','RB','WR','TE']:
    result=ridge_pipeline(masterDf.loc[masterDf['position']==pos],df_2017.loc[df_2017['position']==pos])
    d[pos]=result
myProjDf=pd.concat([df for df in d.values()],ignore_index=True)
