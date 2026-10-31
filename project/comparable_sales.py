"""Training-only, within-project nearest comparable transactions."""
import numpy as np
import pandas as pd
from features import MISSING


def identity(x):
    return x.canonical_project.where(x.canonical_project.ne(MISSING),x.geo_district+'|'+x.address_base)


def coordinates(x):
    return np.column_stack([
        np.log(x.building_area.clip(lower=1)).fillna(0)/.18,
        x.parking_fraction.fillna(0)/.08,
        x.floor.fillna(0)/8,
        x.rooms.fillna(0)/1.5,
        x.total_floors.fillna(0)/8,
        x.parking_count.fillna(0)/1.5,
    ]).astype(float)


class ComparableSales:
    def fit(self,x,y,base):
        xx=x.reset_index(drop=True)
        self.coordinates=coordinates(xx)
        self.unit=np.asarray(y,float)/np.asarray(base,float)
        self.type=xx.building_type.to_numpy()
        self.day=xx.transaction_days.to_numpy(float)
        self.groups={key:np.asarray(ix) for key,ix in pd.Series(np.arange(len(xx))).groupby(identity(xx),sort=False)}
        return self

    def predict(self,x,base):
        xx=x.reset_index(drop=True)
        c=coordinates(xx);kind=xx.building_type.to_numpy();day=xx.transaction_days.to_numpy(float)
        prediction=np.full(len(xx),np.nan);confidence=np.zeros(len(xx))
        for key,indices in pd.Series(np.arange(len(xx))).groupby(identity(xx),sort=False):
            if key not in self.groups:continue
            train=self.groups[key]
            for start in range(0,len(indices),128):
                query=indices.to_numpy()[start:start+128]
                distance=np.abs(c[query,None,:]-self.coordinates[None,train,:]).sum(axis=2)
                distance+=5*(kind[query,None]!=self.type[None,train])
                age=np.maximum(day[query,None]-self.day[None,train],0)
                distance+=age/730
                k=min(7,len(train))
                near=np.argpartition(distance,k-1,axis=1)[:,:k]
                dd=np.take_along_axis(distance,near,axis=1)
                weights=np.exp(-dd+dd.min(axis=1,keepdims=True))
                values=self.unit[train[near]]
                order=np.argsort(values,axis=1)
                ww=np.take_along_axis(weights,order,axis=1)
                vv=np.take_along_axis(values,order,axis=1)
                median=(np.cumsum(ww,axis=1)>=ww.sum(axis=1,keepdims=True)/2).argmax(axis=1)
                prediction[query]=vv[np.arange(len(query)),median]*np.asarray(base)[query]
                confidence[query]=np.exp(-dd.min(axis=1)/5)*min(len(train)/10,1)
        return prediction,confidence
