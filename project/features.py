"""Stateless Taiwan transaction features. No labels or test-fitted statistics."""
import re
import unicodedata

import numpy as np
import pandas as pd

TARGET = '總價元'
ID = '編號'
MISSING = '__MISSING__'


def normalize(value):
    if pd.isna(value):
        return ''
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value))).replace('台', '臺')


def chinese_number(value):
    text = re.split('[,，、]', normalize(value))[0].replace('兩', '二').replace('地下', '-')
    if not text:
        return np.nan
    found = re.search(r'-?\d+(?:\.\d+)?', text)
    if found:
        return float(found.group())
    text = re.sub(r'[^零〇一二三四五六七八九十百千負-]', '', text)
    if not text:
        return np.nan
    negative = '-' in text or '負' in text
    digits = {c: i for i, c in enumerate('零一二三四五六七八九')}
    digits['〇'] = 0
    result, current = 0, 0
    for char in text:
        if char in digits:
            current = digits[char]
        elif char in {'十', '百', '千'}:
            result += (current or 1) * {'十': 10, '百': 100, '千': 1000}[char]
            current = 0
    return float((-1 if negative else 1) * (result + current))


def roc_date(series):
    # Source fields use numeric ROC yyyMMdd or yyyMM, sometimes stored as floats.
    raw = pd.to_numeric(series, errors='coerce').astype(float)
    raw = raw.where(raw >= 10000)
    month_only = raw.between(10000, 99999) | raw.between(190001, 209912)
    raw = raw.where(~month_only, raw * 100 + 1)
    year = np.floor(raw / 10000)
    year = year.where(year >= 1911, year + 1911)
    month = np.floor(raw / 100) % 100
    day = raw % 100
    result = pd.to_datetime(pd.DataFrame({'year': year, 'month': month, 'day': day}), errors='coerce')
    return result.where(year.between(1900, 2026))


def numeric(frame, column):
    return pd.to_numeric(frame[column], errors='coerce').astype(float)


def extract_road(address):
    # A regex character class would wrongly stop at 大 in 大園區 or 大安區.
    road = address.str.extract(r'^(.+?(?:路|街|大道)(?:[一二三四五六七八九十\d]+段)?)', expand=False)
    return road.fillna(address.str.replace(r'\d.*$', '', regex=True))


def make_features(frame):
    frame = frame.reset_index(drop=True)
    text = {c: frame[c].map(normalize) for c in frame if c not in [TARGET, ID]}
    out = {}
    nums = {
        'land_area': '土地移轉總面積平方公尺', 'building_area': '建物移轉總面積平方公尺',
        'parking_area': '車位移轉總面積平方公尺', 'rooms': '建物現況格局-房',
        'living_rooms': '建物現況格局-廳', 'bathrooms': '建物現況格局-衛',
    }
    for name, col in nums.items():
        out[name] = numeric(frame, col)
    cats = {
        'district': '鄉鎮市區', 'transaction_type': '交易標的', 'urban_zone': '都市土地使用分區',
        'rural_zone': '非都市土地使用分區', 'rural_use': '非都市土地使用編定',
        'building_type': '建物型態', 'purpose': '主要用途', 'material': '主要建材',
        'partition': '建物現況格局-隔間', 'management': '有無管理組織', 'parking_type': '車位類別',
    }
    for name, col in cats.items():
        out[name] = text[col]
    address = text['土地位置建物門牌']
    out['city'] = address.str.extract(r'^(.{2,3}[市縣])', expand=False).fillna('')
    out['city_district'] = out['city'] + '|' + out['district']
    out['road'] = extract_road(address)
    out['address_base'] = address.str.replace(r'(?:[一二三四五六七八九十百\d]+[樓層]|[A-Za-z\d-]+[Ff]).*$', '', regex=True)
    project = text['建案名稱']
    out['project'] = (out['city_district'] + '|' + project).where(project.ne(''), '')
    # Floor-independent identity prevents nearby units masquerading as independent groups.
    group = out['project'].where(project.ne(''), out['city_district'] + '|' + out['address_base'])
    tower = text['棟及號'].str.replace(r'[-之]?\d+[Ff樓層].*$', '', regex=True)
    out['tower'] = (out['project'] + '|' + tower).where(project.ne('') & tower.ne(''), '')
    out['project_building_type'] = (out['project'] + '|' + out['building_type']).where(project.ne(''), '')
    floors = text['移轉層次']
    out['floor'] = floors.map(chinese_number)
    out['total_floors'] = text['總樓層數'].map(chinese_number)
    out['floor_fraction'] = out['floor'] / out['total_floors'].replace(0, np.nan)
    out['is_top_floor'] = (out['floor'] == out['total_floors']).astype(float)
    out['is_ground_floor'] = (out['floor'] == 1).astype(float)
    out['is_basement'] = floors.str.contains('地下').astype(float)
    out['whole_building'] = floors.str.contains('全').astype(float)
    out['multi_floor'] = floors.str.contains('[,，、]').astype(float)
    dates = roc_date(frame['交易年月日'])
    completion = roc_date(frame['建築完成年月'])
    out['transaction_year'] = dates.dt.year
    out['transaction_month'] = dates.dt.month
    out['transaction_days'] = (dates - pd.Timestamp('2000-01-01')).dt.days
    out['completion_year'] = completion.dt.year
    out['age_years'] = (dates - completion).dt.days / 365.25
    out['age_years'] = out['age_years'].where(out['age_years'].between(-10, 150))
    out['pre_sale'] = completion.isna().astype(float)
    counts = text['交易筆棟數']
    for name, token in [('land_count', '土地'), ('building_count', '建物'), ('parking_count', '車位')]:
        out[name] = pd.to_numeric(counts.str.extract(token + r'(\d+)', expand=False), errors='coerce')
    out['usable_area'] = out['building_area'] - out['parking_area'].fillna(0)
    out['usable_area'] = out['usable_area'].where(out['usable_area'] > 0, out['building_area'])
    for col in ['land_area', 'building_area', 'parking_area', 'usable_area']:
        out['log_' + col] = np.log1p(out[col].clip(lower=0))
    out['parking_fraction'] = out['parking_area'] / out['building_area'].replace(0, np.nan)
    out['land_building_ratio'] = out['land_area'] / out['building_area'].replace(0, np.nan)
    out['area_per_room'] = out['usable_area'] / (out['rooms'] + out['living_rooms']).replace(0, np.nan)
    # Use semantic flags only; free text can contain prices, compensation, or post-sale outcomes.
    note = text['備註']
    for name, pattern in {
        'special_relationship': '親友|親屬|特殊關係|關係人', 'partial_right': '持分|部分移轉|部分產權',
        'decoration': '裝潢|裝修', 'balcony': '陽臺|陽台|露臺|露台',
        'unregistered': '未登記|增建', 'presale_note': '預售', 'land_only_note': '僅土地',
    }.items():
        out['note_' + name] = note.str.contains(pattern).astype(float)
    out['missing_fields'] = frame.drop(columns=[TARGET, ID], errors='ignore').isna().sum(axis=1)
    result = pd.DataFrame(out)
    cat_cols = list(cats) + ['city', 'city_district', 'road', 'address_base', 'project', 'tower', 'project_building_type']
    for col in result:
        if col in cat_cols:
            result[col] = result[col].fillna('').replace('', MISSING).astype(str)
        else:
            result[col] = pd.to_numeric(result[col], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(np.float32)
    # Monetary scale is deterministic, not fitted. This branch predicts price/m² residuals.
    offset = np.log(result['building_area'].where(result['building_area'] > 1, result['land_area']).fillna(1).clip(lower=1).to_numpy())
    return result, cat_cols, group.astype(str).to_numpy(), offset


def duplicate_groups(frame):
    features = frame.drop(columns=[ID, TARGET], errors='ignore').fillna('').astype(str)
    return pd.util.hash_pandas_object(features, index=False).to_numpy()


class CategoryEncoder:
    """LightGBM category vocabulary learned only from the current fitting partition."""
    def fit(self, x, columns):
        self.columns = list(columns)
        self.levels = {c: sorted(x[c].unique()) for c in columns}
        return self

    def transform(self, x):
        result = x.copy()
        for c in self.columns:
            result[c] = pd.Categorical(x[c], categories=self.levels[c])
        return result
