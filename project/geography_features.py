"""Non-price geographic features from fixed 2020 public road observations.

Addresses join to coordinates, never to external house prices. Road medians
are approximate locations, not parcel geocoding or measures of accident risk.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from features import normalize, chinese_number
from train_mae import checksum, dump, save_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/geography_v1'
EARTH_KM = 6371.0088
ROAD_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]{1,15}?(?:大道|路|街)(?:[零〇一二三四五六七八九十百千0-9]+段)?')


def distance(lon1, lat1, lon2, lat2):
    x1, y1, x2, y2 = map(np.radians, [lon1, lat1, lon2, lat2])
    h = np.sin((y2-y1)/2)**2+np.cos(y1)*np.cos(y2)*np.sin((x2-x1)/2)**2
    return 2*EARTH_KM*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


@lru_cache(maxsize=600000)
def address_keys(address, district_hint=''):
    s = normalize(address)
    m = re.match(r'^(.{2,3}?[市縣])', s)
    if not m:
        return ('', '', '', '', '', '')
    original_city = m.group(1)
    city = original_city.replace('桃園縣', '桃園市')
    local = s[len(original_city):]
    for _ in range(4):
        if local.startswith(original_city):
            local = local[len(original_city):]
    m = re.match(r'^(.{1,4}?區|.{1,4}?[鄉鎮]|.{1,4}?市)', local)
    district = m.group(1) if m else normalize(district_hint)
    if m:
        local = local[len(district):]
    if city == '桃園市':
        district = re.sub(r'[鄉鎮市]$', '區', district)
    if not district or district == city:
        return (city, '', '', '', '', '')
    key = city+'|'+district
    # Intersection records can repeat city/district on the second street.
    local = local.replace(original_city, '').replace(district, '')
    local = re.sub(r'(^|[/,，、與及和])[^路街段巷號/]{1,6}[里村](?:\d+鄰)?', r'\1', local)
    local = re.sub(r'^\d+鄰', '', local)
    local = re.sub(r'[^/,，、路街]{1,10}段[\d之,，、.\-]+地號', ' ', local)
    local = re.sub(r'(路|街|段)(?:與|及|和|交叉|交會|VS|vs)', r'\1/', local)
    roads = []
    for match in ROAD_RE.finditer(local):
        road = re.sub(r'^(?:與|及|和)', '', match.group())
        road = re.sub(r'([零〇一二三四五六七八九十百千0-9]+)段',
                      lambda v: str(int(chinese_number(v.group(1))))+'段', road)
        if road not in roads:
            roads.append(road)
        if len(roads) == 2:
            break
    first = key+'|'+roads[0] if roads else ''
    second = key+'|'+roads[1] if len(roads)>1 else ''
    cross = key+'|'+'&'.join(sorted(roads)) if len(roads)>1 else ''
    house = re.search(r'(?:大道|路|街)(?:[零〇一二三四五六七八九十百千0-9]+段)?(\d+)(?:之\d+)?號', local)
    segment = first+'|'+str(int(house.group(1))//100) if house and len(roads)==1 else ''
    return city, key, first, second, cross, segment


def parse_addresses(address, hints=None):
    if hints is None:
        hints = ['']*len(address)
    return pd.DataFrame([address_keys(str(a), str(h)) for a, h in zip(address, hints)],
        columns=['city', 'district', 'road1', 'road2', 'cross', 'segment'])


def table(points, key):
    a = points.loc[points[key].ne(''), [key, 'longitude', 'latitude']]
    g = a.groupby(key, sort=True)
    result = g.agg(longitude=('longitude', 'median'), latitude=('latitude', 'median'),
                   support=('longitude', 'size'))
    radius = distance(a.longitude.to_numpy(), a.latitude.to_numpy(),
        a[key].map(result.longitude).to_numpy(), a[key].map(result.latitude).to_numpy())
    result['spread_km'] = pd.Series(radius, index=a.index).groupby(a[key]).quantile(.9)
    return result


class RoadGeography:
    def fit(self, references):
        expected = {'address', 'longitude', 'latitude', 'event_date'}
        if set(references) != expected:
            raise ValueError('External reference schema must contain only address, coordinates, event date')
        a = references.copy().reset_index(drop=True)
        dates = pd.to_datetime(a.event_date)
        if not dates.between('2020-01-01', '2020-12-31').all():
            raise ValueError('Only the fixed 2020 observation year is permitted')
        valid = a.longitude.between(118, 123)&a.latitude.between(21.8, 26.5)
        self.audit = {'input_rows':len(a), 'invalid_coordinates':int((~valid).sum())}
        a = a.loc[valid].reset_index(drop=True)
        keys = parse_addresses(a.address)
        points = pd.concat([a[['longitude', 'latitude']], keys], axis=1)
        district = table(points, 'district')
        radius = distance(points.longitude.to_numpy(), points.latitude.to_numpy(),
            points.district.map(district.longitude).to_numpy(), points.district.map(district.latitude).to_numpy())
        valid = np.isfinite(radius)&(radius<=75)
        self.audit['unparsed_or_inconsistent_district'] = int((~valid).sum())
        points = points.loc[valid].reset_index(drop=True)
        self.tables = {key:table(points, key) for key in ['city', 'district', 'cross', 'segment']}
        roads = pd.concat([points[['longitude', 'latitude', 'road1']].rename(columns={'road1':'road'}),
            points.loc[points.road2.ne(''), ['longitude', 'latitude', 'road2']].rename(columns={'road2':'road'})], ignore_index=True)
        self.tables['road'] = table(roads, 'road')
        self.tree = BallTree(np.radians(points[['latitude', 'longitude']].to_numpy()), metric='haversine')
        self.audit['retained_rows'] = len(points)
        self.audit['table_keys'] = {k:len(v) for k, v in self.tables.items()}
        self.audit['reference_year'] = 2020
        self.audit['external_price_fields_used'] = False
        return self

    def locate(self, address, hints=None):
        keys = parse_addresses(address, hints)
        n = len(keys)
        chosen = pd.DataFrame({'longitude':np.full(n, np.nan), 'latitude':np.full(n, np.nan),
            'support':np.zeros(n), 'spread_km':np.full(n, np.nan), 'level':np.zeros(n)})
        for key, source, level, minimum, max_spread in [
                ('city', 'city', 1, 10, 200), ('district', 'district', 2, 5, 75),
                ('road1', 'road', 3, 3, 15), ('segment', 'segment', 4, 3, 2),
                ('cross', 'cross', 5, 2, 1)]:
            t = self.tables[source].reindex(keys[key]).reset_index(drop=True)
            ok = (t.support>=minimum)&(t.spread_km<=max_spread)
            chosen.loc[ok, ['longitude', 'latitude', 'support', 'spread_km']] = t.loc[ok].to_numpy()
            chosen.loc[ok, 'level'] = level
        return chosen, keys

    def transform(self, raw):
        address = raw['土地位置建物門牌'].fillna('').astype(str)
        hints = raw['鄉鎮市區'].fillna('').astype(str)
        located, keys = self.locate(address, hints)
        result = pd.DataFrame(index=np.arange(len(raw)))
        result['ext_longitude'] = located.longitude
        result['ext_latitude'] = located.latitude
        result['ext_location_level'] = located.level
        result['ext_location_support_log'] = np.log1p(located.support)
        result['ext_location_spread_km'] = located.spread_km
        for region in ['city', 'district']:
            t = self.tables[region]
            lon = keys[region].map(t.longitude)
            lat = keys[region].map(t.latitude)
            result[f'ext_{region}_longitude'] = lon
            result[f'ext_{region}_latitude'] = lat
            result[f'ext_distance_{region}_km'] = distance(located.longitude, located.latitude, lon, lat)
        good = located.latitude.notna()&located.longitude.notna()
        for radius in [1, 3]:
            result[f'ext_reference_density_{radius}km_log'] = np.nan
            if good.any():
                counts = self.tree.query_radius(np.radians(located.loc[good, ['latitude', 'longitude']].to_numpy()),
                    r=radius/EARTH_KM, count_only=True)
                result.loc[good, f'ext_reference_density_{radius}km_log'] = np.log1p(counts)
        return result.astype(np.float32)


def load_source(name):
    path = OUT/'sources'/name
    metadata = json.loads(path.with_suffix('.json').read_text())
    if checksum(path) != metadata['sha256']:
        raise ValueError('Changed external source: '+name)
    a = pd.read_csv(path, usecols=['發生時間', '發生地點', '經度', '緯度'])
    events = a['發生時間'].str.match(r'^\d+年\d+月\d+日', na=False)
    footers = a.loc[~events, '發生時間'].tolist()
    if not all(str(s).startswith(('資料提供日期：', '事故類別：')) for s in footers):
        raise ValueError('Unrecognized source date rows')
    a = a.loc[events].reset_index(drop=True)
    d = a['發生時間'].str.extract(r'^(\d+)年(\d+)月(\d+)日').astype(int)
    dates = pd.to_datetime(pd.DataFrame({'year':d[0]+1911, 'month':d[1], 'day':d[2]}))
    return pd.DataFrame({'address':a['發生地點'], 'longitude':pd.to_numeric(a['經度'], errors='coerce'),
        'latitude':pd.to_numeric(a['緯度'], errors='coerce'), 'event_date':dates})


def prepare():
    from train_robust import data
    references = pd.concat([load_source('accident_2020_a2_h1.csv'), load_source('accident_2020_a2_h2.csv')], ignore_index=True)
    model = RoadGeography().fit(references)
    save_model(OUT/'geocoder.joblib', model)
    held = load_source('accident_2020_a1.csv')
    located, _ = model.locate(held.address)
    errors = distance(held.longitude, held.latitude, located.longitude, located.latitude)
    audit = dict(model.audit)
    audit['heldout_a1'] = []
    for level in range(1, 6):
        good = (located.level==level)&np.isfinite(errors)&held.longitude.between(118, 123)&held.latitude.between(21.8, 26.5)
        if good.any():
            audit['heldout_a1'].append(dict(level=level, rows=int(good.sum()),
                median_error_km=float(np.median(errors[good])), p90_error_km=float(np.quantile(errors[good], .9))))
    audit['source_vintage'] = '2020 events; all three CSV footers declare provision on 2021-07-01, before the earliest forecast cutoff 2021-08-10. Downloaded bytes are not independently authenticated as the original snapshot. No house-price labels exist in this reference.'
    audit['declared_source_provision_date'] = '2021-07-01'
    dump(OUT/'geocoder_audit.json', audit)
    raw, _, _ = data()
    test = pd.read_parquet(ROOT/'data/test.parquet')
    features = {'train':model.transform(raw), 'test':model.transform(test)}
    save_model(OUT/'features.joblib', features)
    coverage = {name:dict(rows=len(x), levels={str(k):int(v) for k,v in x.ext_location_level.value_counts().sort_index().items()},
        road_or_better_fraction=float((x.ext_location_level>=3).mean())) for name,x in features.items()}
    dump(OUT/'coverage.json', coverage)
    print(json.dumps({'geocoder':audit, 'coverage':coverage}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    # Persist importable class names, so a different inference script can load it.
    from geography_features import prepare as prepare_importable
    prepare_importable()
