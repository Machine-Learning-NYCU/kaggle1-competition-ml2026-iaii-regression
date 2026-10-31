"""Estimate launch-cohort price growth from fitting-only first-quarter sales."""
import numpy as np
import pandas as pd
from features import MISSING


class LaunchIndex:
    def __init__(self,months=24):self.months=months

    def fit(self,x,y,area):
        t=x.transaction_year.to_numpy(float)*12+x.transaction_month.to_numpy(float)
        self.origin=float(np.nanmax(t))
        group=x.canonical_project
        first=pd.Series(t,index=x.index).groupby(group).transform('min').to_numpy()
        q=np.log(np.asarray(y,float)/np.asarray(area,float))
        cell=x.geo_district+'|'+x.building_type+'|'+x.rooms.fillna(-1).astype(str)+'|'+pd.Series(
            np.floor(np.log(area)/.25).astype(int),index=x.index).astype(str)
        a=pd.DataFrame(dict(city=x.geo_city,project=group,cell=cell,t=first,q=q))
        keep=(group!=MISSING)&(t-first<3)&(first>self.origin-self.months)&(q>np.log(10000))&(q<np.log(1500000))
        a=a.loc[keep].groupby(['city','project','cell','t']).q.agg(['median','size']).reset_index()
        diversity=a.groupby('cell').project.transform('nunique')
        a=a.loc[diversity>=3].copy()
        a['w']=a['size'].clip(upper=20).astype(float)
        a['w']*=np.minimum(1,30/a.groupby('project').w.transform('sum'))
        for col in ['t','median']:
            mean=(a[col]*a.w).groupby(a.cell).transform('sum')/a.w.groupby(a.cell).transform('sum')
            a['d_'+col]=a[col]-mean
        a['num']=a.w*a.d_t*a.d_median.clip(-.4,.4)
        a['den']=a.w*a.d_t**2
        self.global_rate=float(np.clip(a.num.sum()/max(a.den.sum(),1),-.01,.025))
        stat=a.groupby('city')[['num','den']].sum()
        self.rates={city:float(np.clip((r.num+10000*self.global_rate)/(r.den+10000),-.01,.025)) for city,r in stat.iterrows()}
        self.stats=stat.to_dict(orient='index');self.rows=len(a);self.projects=a.project.nunique()
        return self

    def excess_factor(self,x,repeat_index,strength=1.):
        t=x.transaction_year.to_numpy(float)*12+x.transaction_month.to_numpy(float)
        launch=x.geo_city.map(self.rates).fillna(self.global_rate).to_numpy(float)
        repeat=x.geo_city.map(repeat_index.rates).fillna(repeat_index.global_rate).to_numpy(float)
        delta=np.clip(launch-repeat,-.008,.01)
        return np.exp(strength*delta*np.clip(t-self.origin,0,12))
