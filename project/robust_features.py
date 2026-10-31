"""Local-address representation and a training-only repeat-project price index.

Uses competition fields only. No geocoding, external prices, or test labels.
"""
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from features import make_features, normalize, MISSING

NUM = '零〇一二三四五六七八九十百千0-9'


def address_parts(address, district):
    address, district = normalize(address), normalize(district)
    city = re.match(r'^(.{2,3}[市縣])', address)
    city = city.group(1) if city else ''
    local = address[len(city):]
    # The source sometimes repeats 新竹市 or the administrative district.
    for _ in range(4):
        for prefix in [city, district]:
            if prefix and local.startswith(prefix):
                local = local[len(prefix):]
    if city == '桃園縣':
        city = '桃園市'
        district = re.sub('[市鎮鄉]$', '區', district)
    if city in ['新竹市', '嘉義市']:
        sub = re.match(r'^(東區|西區|北區|香山區)', local)
        if sub:
            district = sub.group(1)
            local = local[len(district):]
    local = re.sub(r'^.{1,5}里(?:\d+鄰)?', '', local)
    local = re.sub(r'^\d+鄰', '', local)
    section = re.search(r'([^\d,，、及與路街]{1,8}段)(?:\d|之)', local)
    section = section.group(1) if section else ''
    # Remove parcel identifiers before finding roads at the same location.
    roads_text = re.sub(r'[^,，、路街]{1,10}段[\d之,，、.\-]+地號', ' ', local)
    roads_text = re.sub(r'(路|街|段)(?:與|及|和|VS|vs)', r'\1 ', roads_text)
    roads = re.findall(r'[\u4e00-\u9fffA-Za-z0-9]{1,12}?(?:大道|路|街)(?:[' + NUM + r']+段)?', roads_text)
    roads = [re.sub(r'^(與|及)', '', r) for r in roads][:2]
    key = city + '|' + district
    road1 = roads[0] if roads else ''
    road2 = roads[1] if len(roads)>1 else ''
    lane = re.search(r'([' + NUM + r']+)巷', local)
    house = re.search(r'(\d+)(?:之\d+)?號', local)
    parcel = re.search(r'段(\d+)', local)
    # Text has no floor/unit/house number: embeddings describe the neighborhood.
    location = key + '|' + '|'.join(roads) + '|' + section
    return (city, key, key+'|'+road1 if road1 else MISSING,
            key+'|'+road2 if road2 else MISSING,
            key+'|'+'|'.join(sorted(roads)) if roads else key+'|'+section,
            key+'|'+section if section else MISSING,
            key+'|'+road1+'|'+lane.group(1) if lane else MISSING,
            float(house.group(1)) if house else np.nan,
            float(parcel.group(1)) if parcel else np.nan, location)


def enhanced_features(raw, base=None):
    if base is None:
        x, cats, groups, offset = make_features(raw)
    else:
        x, cats, groups, offset = base
        x, cats = x.copy(), list(cats)
    parts = pd.DataFrame([address_parts(a,d) for a,d in zip(raw['土地位置建物門牌'], raw['鄉鎮市區'])],
                         columns=['geo_city','geo_district','geo_road1','geo_road2','geo_cross',
                                  'geo_section','geo_lane','house_number','parcel_number','location_text'])
    for col in parts:
        x[col] = parts[col].to_numpy()
        if col not in ['house_number','parcel_number','location_text']:
            cats.append(col)
    project = raw['建案名稱'].map(normalize).str.replace(r'[^\w一-龥]', '', regex=True)
    for size in [2,3]:
        col = f'builder_prefix_{size}'
        x[col] = (parts.geo_city+'|'+project.str[:size]).where(project.str.len()>=size, MISSING)
        cats.append(col)
    x['canonical_project'] = (parts.geo_district+'|'+project).where(project.ne(''), MISSING)
    cats.append('canonical_project')
    tower = raw['棟及號'].map(normalize).str.extract(r'^([A-Za-z]+)',expand=False).fillna('')
    x['tower_letter'] = tower.replace('', MISSING)
    cats.append('tower_letter')
    x['geo_road_type'] = x.geo_road1+'|'+x.building_type
    x['district_type'] = x.geo_district+'|'+x.building_type
    cats += ['geo_road_type','district_type']
    x['area_per_parking'] = x.parking_area/x.parking_count.replace(0,np.nan)
    x['area_per_building'] = x.building_area/x.building_count.clip(lower=1)
    x['land_per_building'] = x.land_area/x.building_count.clip(lower=1)
    x['usable_per_building'] = x.usable_area/x.building_count.clip(lower=1)
    x['floor_gap'] = x.total_floors-x.floor
    x['room_total'] = x.rooms+x.living_rooms+x.bathrooms
    x['usable_area_per_bath'] = x.usable_area/x.bathrooms.replace(0,np.nan)
    x['has_parking'] = (x.parking_count>0).astype(float)
    for col in x:
        if col in cats:
            x[col] = x[col].fillna(MISSING).astype(str)
        elif col != 'location_text':
            x[col] = pd.to_numeric(x[col],errors='coerce').replace([np.inf,-np.inf],np.nan).astype(np.float32)
    area = np.exp(np.asarray(offset,float))
    return x, cats, np.asarray(groups), area


class AddressEmbedding:
    """Target-free character embeddings fitted to fitting addresses only."""
    def fit(self, x):
        text = x.location_text.unique()
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4), min_df=2,
                                         max_features=30000, sublinear_tf=True, dtype=np.float32)
        a = self.vectorizer.fit_transform(text)
        self.svd = TruncatedSVD(n_components=min(40,a.shape[0]-1,a.shape[1]-1), random_state=2026)
        with threadpool_limits(limits=8):
            self.svd.fit(a)
        return self

    def transform(self, x):
        unique = x.location_text.unique()
        with threadpool_limits(limits=8):
            embedded = self.svd.transform(self.vectorizer.transform(unique))
        ix = pd.Index(unique).get_indexer(x.location_text)
        result = x.drop(columns='location_text').copy()
        for j in range(embedded.shape[1]):
            result[f'address_svd_{j}'] = embedded[ix,j].astype(np.float32)
        return result


class RepeatProjectIndex:
    """Shrink recent within-project price slopes; never fit on evaluation labels.

    Project/type/room/size-cell fixed effects reduce changing transaction mix.
    The index is a predictive extrapolation assumption, not a causal estimate.
    """
    def fit(self, x, y, area):
        month = x.transaction_year.to_numpy(float)*12+x.transaction_month.to_numpy(float)
        self.origin = float(np.nanmax(month))
        q = np.log(np.asarray(y)/area)
        cell = (x.canonical_project+'|'+x.building_type+'|'+x.rooms.fillna(-1).astype(str)
                +'|'+np.floor(np.log(area)/.2).astype(int).astype(str))
        a = pd.DataFrame({'city':x.geo_city,'project':x.canonical_project,'cell':cell,'t':month,'q':q})
        a = a[(a.t>self.origin-24)&(a.project!=MISSING)&a.q.between(np.log(10000),np.log(1500000))]
        a = a.groupby(['city','project','cell','t'],observed=True).q.agg(['median','size']).reset_index()
        a['w'] = a['size'].clip(upper=20).astype(float)
        # Cap total influence per project, avoiding a huge launch setting the city index.
        a['w'] *= np.minimum(1., 150/a.groupby('project').w.transform('sum'))
        for col in ['t','median']:
            weighted = a[col]*a.w
            mean = weighted.groupby(a.cell).transform('sum')/a.w.groupby(a.cell).transform('sum')
            a['d_'+col] = a[col]-mean
        a['num'] = a.w*a.d_t*a.d_median.clip(-.25,.25)
        a['den'] = a.w*a.d_t**2
        stat = a.groupby('city')[['num','den']].sum()
        self.global_rate = float(np.clip(stat.num.sum()/max(stat.den.sum(),1),-.008,.015))
        self.rates = {city:float(np.clip((row.num+15000*self.global_rate)/(row.den+15000),-.008,.015))
                      for city,row in stat.iterrows()}
        self.stats = stat.to_dict(orient='index')
        return self

    def factor(self, x):
        months = x.transaction_year.to_numpy(float)*12+x.transaction_month.to_numpy(float)
        rate = x.geo_city.map(self.rates).fillna(self.global_rate).to_numpy(float)
        # Detrend recent training history; do not impose today's rate on a decade ago.
        horizon = np.clip(months-self.origin,-36,12)
        return np.exp(rate*horizon)
