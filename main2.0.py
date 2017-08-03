import pandas as pd
import os
import sklearn.linear_model as linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
os.chdir('C:/Users/ericb/Desktop/FF_machine_learning')
#%%  scrape ESPN projections --------------------------------------------------
def scrape_espn_proj(year):
    year=str(year)
    posDic={'QB':'0','RB':'2','WR':'4','TE':'6','DST':'16'}
    cutoffs={'QB':40,'RB':85,'WR':90,'TE':35,'DST':15}
    dfDic={}
    def procc_page(pos,page):
        res=requests.get('http://games.espn.com/ffl/tools/projections?&seasonId='+year+'&leagueId=156099&slotCategoryId='+posDic[pos]+'&startIndex='+str(40*(page-1)))
        df=pd.io.html.read_html(res.text)[0]
        df.drop([0,1],inplace=True)
        df.rename(columns=df.loc[2,:],inplace=True)
        df.drop(2,inplace=True)
        df=df.loc[:,['RNK','PLAYER, TEAM POS','PTS']]
        df.RNK=pd.to_numeric(df.RNK)
        df=df.loc[df['RNK'] <= cutoffs[pos]]
        df['PLAYER, TEAM POS']=df['PLAYER, TEAM POS'].str.replace(r'\s(Q|SSPD)$','')
        # Ty Montgomery
        df['player']=df['PLAYER, TEAM POS'].str.rsplit(None,2).str[0].str.replace(',','')
        df['position']=df['PLAYER, TEAM POS'].str.rsplit(None,2).str[2]
        df.drop('PLAYER, TEAM POS',axis=1,inplace=True)
        df.rename(columns={'RNK':'posRank','PTS':'espn_proj_pts'},inplace=True)
        return df
    for pos in posDic.keys():
        if pos in ['WR','RB']:
            dfDic[pos]=pd.concat([procc_page(pos,pg) for pg in [1,2,3]],ignore_index=True)
        else:
            dfDic[pos]=procc_page(pos,1)
    bigdf=pd.concat([df for df in dfDic.values()],ignore_index=True)
    bigdf.drop('posRank',axis=1,inplace=True)
    bigdf=bigdf[['player','position','espn_proj_pts']]
    bigdf.to_csv(year+'_espn_projections.csv',index=False)
    return
#scrape_espn_proj(2016)
#scrape_espn_proj(2017)
#%% scrape actual scores from ESPN --------------------------------------------
def scrape_act_pts():
    posDic={'QB':'0','RB':'2','WR':'4','TE':'6','DST':'16'}
    mastDf=pd.read_csv('2016_espn_projections.csv')
    mastDf['actual_pts']=np.nan
    def procc_page(pos,name='',page=1,mastDf=mastDf):
        res=requests.get('http://games.espn.com/ffl/leaders?&leagueId=156099&slotCategoryId='+posDic[pos]+'&avail=-1&startIndex='+str(50*(page-1))+'&search='+name)
        df=pd.io.html.read_html(res.text)[0]
        df.drop([0,1],inplace=True)
        df.rename(columns=df.loc[2,:],inplace=True)
        df.drop(2,inplace=True)
        df=df.loc[:,['PLAYER, TEAM POS','PTS']]
        df['PLAYER, TEAM POS']=df['PLAYER, TEAM POS'].str.replace(r'\s(Q|SSPD)$','')
        # Ty Montgomery
        df['player']=df['PLAYER, TEAM POS'].str.rsplit(None,2).str[0].str.replace(',','')
        df['position']=df['PLAYER, TEAM POS'].str.rsplit(None,2).str[2]
        df.drop('PLAYER, TEAM POS',axis=1,inplace=True)
        df.rename(columns={'PTS':'actual_pts_x'},inplace=True)
        df=df[['player','position','actual_pts_x']]
        if name!='' and df.shape[0]>1:
            return 'WARNING: multiple players named '+name
        mastDf=mastDf.merge(df,on=['player','position'],how='left')
        mastDf['actual_pts'].fillna(mastDf['actual_pts_x'],inplace=True)
        mastDf.drop('actual_pts_x',axis=1,inplace=True)
        return mastDf
    for pos in posDic.keys():
        mastDf=procc_page(pos,mastDf=mastDf)
        if pos in ['WR','RB']:
            mastDf=procc_page(pos,page=2,mastDf=mastDf)
    nullDf=mastDf[mastDf.actual_pts.isnull()]
    for index, row in nullDf.iterrows():
        try:
            mastDf=procc_page(row['position'],name=row['player'],mastDf=mastDf)
        except ValueError:
            print(row['player'])
    mastDf.position=mastDf.position.str.replace('D/ST','DST')
    return mastDf
#scrape_act_pts().to_csv('2016_actual_and_espn_proj.csv',index=False)
#%% scrape projections from FantasyData ---------------------------------------
years_ago=0
#def scrape_FD(years_ago):
mastDf=pd.read_csv('2017_espn_projections.csv')
mastDf.position=mastDf.position.str.replace('D/ST','DST')
mastDf['fantasyData_proj_pts']=np.nan
mastDf['player_espn']=mastDf['player']
mastDf.player=mastDf.player.str.replace('.','')
mastDf.player=mastDf.player.str.lower()
posDic={'QB':'1','RB':'2','WR':'3','TE':'4','DST':'6','all':'0'}
def procc_page(pos='all',name='',team=0,mastDf=mastDf):
    res=requests.get('https://fantasydata.com/nfl-stats/fantasy-football-weekly-projections.aspx?fs=0&stype=0&sn='
                     +str(years_ago)+'&scope=0&w=0&ew=0&s='+name+'&t='+str(team)+'&p='
                     +posDic[pos]+'&st=FantasyPoints&d=1&ls=&live=false&pid=false&minsnaps=4')
    df=pd.io.html.read_html(res.text)[0]
    df.columns=df.iloc[0] #changes header row
    df.drop(0,inplace=True)
    df=df[['Player','Pos','Fantasy Points',]]
    df.rename(columns={'Player':'player','Pos':'position','Fantasy Points':'fantasyData_proj_pts_x'},inplace=True)
    df.fantasyData_proj_pts_x=pd.to_numeric(df.fantasyData_proj_pts_x)
    if name!='' and df.shape[0]>2:
        return mastDf,'WARNING: multiple players named '+name
    
    df.loc[df.position=='DST','player']=df.loc[df.position=='DST','player'].str.split().str[-1]
    df.player=df.player.str.lower()
    mastDf=mastDf.merge(df,on=['player','position'],how='left')
    mastDf['fantasyData_proj_pts'].fillna(mastDf['fantasyData_proj_pts_x'],inplace=True)
    mastDf.drop('fantasyData_proj_pts_x',axis=1,inplace=True)
    return mastDf
for pos in posDic.keys():
    mastDf=procc_page(pos,mastDf=mastDf)
for t in range(1,33):
    mastDf=procc_page(team=t,mastDf=mastDf)
mastDf.player=mastDf.player.str.replace(r'\s(jr|sr|v)$','')
nullDf=mastDf[mastDf.fantasyData_proj_pts.isnull()]
for index, row in nullDf.iterrows():
    try:
        mastDf=procc_page(row['position'],name=row['player'],mastDf=mastDf)
    except:
        print(row['player'])
mastDf.drop('player',axis=1,inplace=True)
mastDf.rename(columns={'player_espn':'player'},inplace=True)
mastDf.to_csv('2017_projections.csv',index=False)
#%% fantasySharks processing --------------------------------------------------
year='2017'
mastDf=pd.read_csv('2017_projections.csv')

mastDf['fantasySharks_proj_pts']=np.nan
mastDf['player_espn']=mastDf['player']
mastDf.player=mastDf.player.str.replace(r'\s(Jr.|Sr.|V|III)$','')
mastDf.player=mastDf.player.str.replace("'",'')
mastDf.player=mastDf.player.str.replace('Robert Kelley','Rob Kelley')
mastDf.player=mastDf.player.str.replace('Benjamin Cunningham','Benny Cunningham')
df=pd.read_csv(year+'_fantasySharks.csv')
df=df.loc[df['team'] != 'FA']
df=df[['player','playerposition','points']]
df=df.rename(columns = {'playerposition':'position','points':'fantasySharks_proj_pts_x'})
df.drop_duplicates(inplace=True)
new_Ty=df.loc[df.player=='Ty Montgomery',:]
new_Ty['position']='RB'
df=df.append(new_Ty,ignore_index=True)
mastDf=mastDf.merge(df,on=['player','position'],how='left')
mastDf['fantasySharks_proj_pts'].fillna(mastDf['fantasySharks_proj_pts_x'],inplace=True)
player=mastDf['player_espn']
mastDf.drop(['fantasySharks_proj_pts_x','player','player_espn'],axis=1,inplace=True)
mastDf.insert(0, 'player', player)

mastDf.to_csv('2017_projections.csv',index=False)
#%% scrape FFToday ------------------------------------------------------------
year=str(2017)

posDic={'QB':'10','RB':'20','WR':'30','TE':'40','DST':'99'}
mastDf=pd.read_csv('2017_projections.csv')
mastDf['ffToday_proj_pts']=np.nan
mastDf['player_espn']=mastDf['player']
mastDf.player=mastDf.player.str.replace(r'\s(Sr.|V)$','')
mastDf.player=mastDf.player.str.replace('Robert Kelley','Rob Kelley')
mastDf.player=mastDf.player.str.replace('Benjamin Cunningham','Benny Cunningham')
mastDf.player=mastDf.player.str.replace('Duke Johnson Jr.','Duke Johnson')
def procc_page(pos,page=1,mastDf=mastDf):
    res=requests.get('http://www.fftoday.com/rankings/playerproj.php?Season='
                     +year+'&PosID='+posDic[pos]+'&LeagueID=26955&cur_page='+str(page-1))
    df=pd.io.html.read_html(res.text)[10]
    df.columns=df.iloc[1]
    df.drop([0,1],inplace=True)
    if pos=='DST':
        df.rename(columns={'Team':'player','FPts':'ffToday_proj_pts_x'},inplace=True)
        df.player=df.player.str.split().str[-1]
    else:
        df.rename(columns={'Player  Sort First: Last:':'player','FPts':'ffToday_proj_pts_x'},inplace=True)
    df['position']=pos
    df=df[['player','position','ffToday_proj_pts_x']]
    mastDf=mastDf.merge(df,on=['player','position'],how='left')
    mastDf['ffToday_proj_pts'].fillna(mastDf['ffToday_proj_pts_x'],inplace=True)
    mastDf.drop('ffToday_proj_pts_x',axis=1,inplace=True)
    return mastDf
for pos in posDic.keys():
    mastDf=procc_page(pos,mastDf=mastDf)
    if pos in ['WR','RB']:
        mastDf=procc_page(pos,page=2,mastDf=mastDf)
nullDf=mastDf[mastDf.ffToday_proj_pts.isnull()]
player=mastDf['player_espn']
mastDf.drop(['player','player_espn'],axis=1,inplace=True)
mastDf.insert(0, 'player', player)
mastDf.to_csv('2017_projections.csv',index=False)
#%% CBS scraping --------------------------------------------------------------
year=str(2016)

mastDf=pd.read_csv('2016_actual_and_proj.csv')
mastDf['cbs_proj_pts']=np.nan
mastDf['player_espn']=mastDf['player']
mastDf.player=mastDf.player.str.replace(r'\s(Jr.|Sr.|V)$','')
mastDf.player=mastDf.player.str.replace('Robert Kelley','Rob Kelley')
mastDf.player=mastDf.player.str.replace('Benjamin Cunningham','Benny Cunningham')
def procc_page(pos,page=1,mastDf=mastDf):
    res=requests.get('http://www.cbssports.com/fantasy/football/stats/sortable/points/'
                     +pos+'/standard/projections/'+year+'/?&print_rows=9999')
    df=pd.io.html.read_html(res.text)[0]
    if pos=='DST':
        df.columns=df.iloc[1]
        df.drop([0,1],inplace=True)
    else:
        df.columns=df.iloc[2]
        df.drop([0,1,2],inplace=True)
    df.rename(columns={'Player':'player','FPTS':'cbs_proj_pts_x'},inplace=True)
    df['position']=pos
    df=df[['player','position','cbs_proj_pts_x']]
    df=df[:-1]
    df['player']=df.player.str.split(',').str[0]
    mastDf=mastDf.merge(df,on=['player','position'],how='left')
    mastDf['cbs_proj_pts'].fillna(mastDf['cbs_proj_pts_x'],inplace=True)
    mastDf.drop('cbs_proj_pts_x',axis=1,inplace=True)
    return mastDf
for pos in posDic.keys():
    mastDf=procc_page(pos,mastDf=mastDf)
nullDf=mastDf[mastDf.cbs_proj_pts.isnull()]
player=mastDf['player_espn']
mastDf.drop(['player','player_espn'],axis=1,inplace=True)
mastDf.insert(0, 'player', player)
mastDf.to_csv('2016_actual_and_proj.csv',index=False)
#%% ffAnalytics, etc. pooling -------------------------------------------------
def ffAnaly_proj(year):
    year=str(year)
    if year=='2017':
        mastFN=year+'_projections.csv'
    else:
        mastFN=year+'_actual_and_proj.csv'
    mastDf=pd.read_csv(mastFN)
    mastDf['player_espn']=mastDf['player']
    mastDf.player=mastDf.player.str.replace(r'\s(Jr.|Sr.|V|III)$','')
    mastDf.player=mastDf.player.str.replace("'",'')
    mastDf.player=mastDf.player.str.replace('Robert Kelley','Rob Kelley')
    mastDf.player=mastDf.player.str.replace('Benjamin Cunningham','Benny Cunningham')
    for file in os.listdir():
        if file.endswith('_'+year+'.csv'):
            source=file[:-9]
            mastDf[source+'_proj_pts']=np.nan
            df=pd.read_csv(file)
            df.loc[df['player']=='Jeremy Maclin','team']='BAL'
            df.loc[df['player']=='Justin Forsett','team']='BAL'
            df=df.loc[df['team'] != 'FA']
            df=df[['player','playerposition','points']]
            df=df.rename(columns = {'playerposition':'position','points':source+'_proj_pts_x'})
            df.drop_duplicates(inplace=True)
            if df.loc[df.player=='Ty Montgomery',:].shape[0]==1:
                new_Ty=df.loc[df.player=='Ty Montgomery',:]
                if new_Ty.position.max()=='WR':
                    new_Ty.loc[:,'position']='RB'
                else:
                    new_Ty.loc[:,'position']='WR'
                df=df.append(new_Ty,ignore_index=True)
            mastDf=mastDf.merge(df,on=['player','position'],how='left')
            mastDf[source+'_proj_pts'].fillna(mastDf[source+'_proj_pts_x'],inplace=True)
            mastDf.drop(source+'_proj_pts_x',axis=1,inplace=True)
    player=mastDf['player_espn']
    mastDf.drop(['player','player_espn'],axis=1,inplace=True)
    mastDf.insert(0, 'player', player)
    mastDf.to_csv(mastFN,index=False)
    return
#ffAnaly_proj(2017)
#%% scrape Scout.com ----------------------------------------------------------
year=str(2017)
pos='QB'

mastDf=pd.read_csv('2017_projections.csv')
mastDf['scout_proj_pts']=np.nan
#mastDf['player_espn']=mastDf['player']
#mastDf.player=mastDf.player.str.replace(r'\s(Jr.|Sr.|V)$','')
#mastDf.player=mastDf.player.str.replace('Robert Kelley','Rob Kelley')
#mastDf.player=mastDf.player.str.replace('Benjamin Cunningham','Benny Cunningham')
posDic={'QB':'quarterback','RB':'running-back','WR':'wide-receiver','TE':'tight-end','DST':'defense'}
#def procc_page(pos,page=1,mastDf=mastDf):
if year == '2016':
    weekly='-weekly'
elif year== '2017':
    weekly=''
#else:
#    return 'error'
res=requests.get('http://www.scout.com/fantasy/'+year+weekly+'-fantasy-football-'+posDic[pos]+'-rankings?seasonYear='+year+'&week=0&ppr=NON-PPR')
df=pd.io.html.read_html(res.text)[0]

if pos=='DST':
    df.columns=df.iloc[1]
    df.drop([0,1],inplace=True)
else:
    df.columns=df.iloc[2]
    df.drop([0,1,2],inplace=True)
df.rename(columns={'Player':'player','FPTS':'cbs_proj_pts_x'},inplace=True)
df['position']=pos
df=df[['player','position','cbs_proj_pts_x']]
df=df[:-1]
df['player']=df.player.str.split(',').str[0]
mastDf=mastDf.merge(df,on=['player','position'],how='left')
mastDf['cbs_proj_pts'].fillna(mastDf['cbs_proj_pts_x'],inplace=True)
mastDf.drop('cbs_proj_pts_x',axis=1,inplace=True)
#    return mastDf
for pos in posDic.keys():
    mastDf=procc_page(pos,mastDf=mastDf)
nullDf=mastDf[mastDf.cbs_proj_pts.isnull()]
player=mastDf['player_espn']
mastDf.drop(['player','player_espn'],axis=1,inplace=True)
mastDf.insert(0, 'player', player)
mastDf.to_csv('2016_actual_and_proj.csv',index=False)
#%% machine learning ----------------------------------------------------------
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
myProjDf.to_csv('my_projections.csv',index=False)
#add D. Pumphrey,Christian Hackenberg,
myProjDf=pd.read_csv('my_projections.csv')
# calculate Flex VOR
vorDic={'DST':3.0,'QB':7.0,'TE':10.0,'RB':39.0,'WR':43.0}
myProjDf['posRank'] = myProjDf.groupby(['position'])['my_proj'].rank(ascending=False)
for name, group in myProjDf.groupby('position'):
    repl=vorDic[name]
    vor=group.loc[group['posRank'].between(repl-1,repl+1),'my_proj'].mean()
    myProjDf.loc[myProjDf.position==name,'VOR']=group['my_proj']-vor
myProjDf.sort_values('VOR',ascending=False,inplace=True)
myProjDf.player.to_csv('player_rankings.txt',index=False,header=False)
