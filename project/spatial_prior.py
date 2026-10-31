"""Group-excluded hierarchical local prices, using only fitting labels.

Every query subtracts its entire project's contribution at EVERY level,
including the global fallback. Roads share a vocabulary across address slots.
The mapping is a supervised fitting-partition representation, not a claim that
each training row observes only earlier-in-time labels within that partition.
"""
import numpy as np
import pandas as pd
from features import MISSING


def project_groups(x):
    return x.canonical_project.where(x.canonical_project.ne(MISSING),x.geo_district+'|'+x.address_base).astype(str)


def effective_area(x, mode='total'):
    building=x.building_area.to_numpy(float)
    land=x.land_area.to_numpy(float)
    area=np.where(building>1,building,np.maximum(np.nan_to_num(land,nan=1),1))
    if mode=='net':
        parking=x.parking_area.fillna(0).to_numpy(float)
        valid=(building>1)&(parking>=0)&(parking<.95*building)
        area-=.75*np.where(valid,parking,0.)
    if mode not in ['total','net']:raise ValueError(mode)
    return np.maximum(area,1.)


class CrossProjectPrior:
    def keys(self,x):
        road_type=lambda c:x[c].where(x[c].eq(MISSING),x[c]+'|'+x.building_type)
        return {'global':[pd.Series('global',index=x.index)],
                'city':[x.geo_city], 'district':[x.geo_district],
                'district_type':[x.district_type],
                'road':[x.geo_road1,x.geo_road2],
                'road_type':[road_type('geo_road1'),road_type('geo_road2')],
                'section':[x.geo_section], 'cross':[x.geo_cross],
                'builder':[x.builder_prefix_2]}

    def fit(self,x,y,area):
        x=x.reset_index(drop=True)
        group=project_groups(x)
        month=x.transaction_year.to_numpy(float)*12+x.transaction_month.to_numpy(float)
        self.origin=float(month.max())
        q=np.clip(np.log(np.asarray(y,float)/np.asarray(area,float)),np.log(1000),np.log(3e6))
        w=np.exp(-np.maximum(self.origin-month,0)/24)
        mass=pd.Series(w).groupby(group).transform('sum').to_numpy()
        w*=np.minimum(1.,30/np.maximum(mass,1e-12))
        self.tables={}
        for level,slots in self.keys(x).items():
            parts=[]
            for slot,key in enumerate(slots):
                valid=key.ne(MISSING)&key.ne('')
                if slot:valid&=key.ne(slots[0])
                part=pd.DataFrame({'key':key,'group':group,'n':w,'s':w*q,
                    's2':w*q*q,'month':w*(month-self.origin)})
                parts.append(part.loc[valid])
            records=pd.concat(parts,ignore_index=True)
            by_group=records.groupby(['key','group'],sort=False)[['n','s','s2','month']].sum()
            total=by_group.groupby(level='key',sort=False).sum()
            self.tables[level]=(total,by_group)
        return self

    def lookup(self,level,key,groups):
        total,by_group=self.tables[level]
        result=total.reindex(key).fillna(0).to_numpy(float,copy=True)
        index=pd.MultiIndex.from_arrays([key,groups],names=['key','group'])
        result-=by_group.reindex(index).fillna(0).to_numpy(float)
        result[:,0]=np.maximum(result[:,0],0.)
        result[result[:,0]<1e-8]=0
        return result

    @staticmethod
    def shrink(stats,parent,strength):
        n,s,_,_=stats.T
        return (s+strength*parent)/(n+strength)

    def transform(self,x):
        x=x.reset_index(drop=True);groups=project_groups(x);keys=self.keys(x)
        out={};values={};counts={}
        parent=np.full(len(x),np.log(150000.))
        ancestry={'global':None,'city':'global','district':'city','district_type':'district',
                  'road':'district','road_type':'district_type','section':'district',
                  'cross':'road','builder':'city'}
        strengths={'global':1.,'city':5.,'district':8.,'district_type':10.,'road':8.,
                   'road_type':6.,'section':8.,'cross':6.,'builder':20.}
        for level,slots in keys.items():
            prior=parent if ancestry[level] is None else values[ancestry[level]]
            stats=[self.lookup(level,key,groups) for key in slots]
            if len(stats)>1:
                # A repeated road token contributes once, independent of slot order.
                stats[1][slots[1].eq(slots[0]).to_numpy()]=0
            pooled=np.sum(stats,axis=0)
            values[level]=self.shrink(pooled,prior,strengths[level])
            counts[level]=pooled[:,0]
            out[f'spatial_{level}']=values[level]
            out[f'spatial_{level}_count']=np.log1p(pooled[:,0])
            mean=pooled[:,1]/np.maximum(pooled[:,0],1e-12)
            out[f'spatial_{level}_sd']=np.sqrt(np.maximum(pooled[:,2]/np.maximum(pooled[:,0],1e-12)-mean**2,0))
            out[f'spatial_{level}_age']=-pooled[:,3]/np.maximum(pooled[:,0],1e-12)
        anchor=np.where(counts['road']>0,.5*(values['cross']+values['road_type']),values['section'])
        out['spatial_anchor']=anchor
        return pd.DataFrame(out,dtype=np.float32)
