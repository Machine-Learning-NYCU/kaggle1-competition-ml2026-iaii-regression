"""Extract published offer schedules; no registered test-sale prices are read."""
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/policy_v1'
SOURCES={
    'airport_handbook_202408':dict(kind='airport',published='2024-08-29',discount_per_ping=0),
    'nearby_handbook_202307':dict(kind='nearby',published='2023-07-12',discount_per_ping=30000),
}


def unit_key(value):
    text=unicodedata.normalize('NFKC',str(value)).upper().replace('棟','').replace('號','').replace(' ','')
    text=text.replace('－','-').replace('–','-').replace('Ｆ','F')
    m=re.fullmatch(r'([A-Z]+\d+)-(\d+)F',text)
    return f'{m[1]}-{int(m[2])}F' if m else ''


def extract(text,source,spec):
    houses=[];parking=[];project=None
    for page,content in enumerate(text.split('\f'),1):
        for line in content.splitlines():
            if spec['kind']=='airport':
                found=re.search(r'街廓\s*([12356])',line)
                if found:project='桃園航空城機場園區街廓'+found[1]+'基地'
            else:
                found=re.search(r'(A11|A18-[123]|G17)\s*基地',line)
                if found:project='桃園航空城'+found[1]+'基地'
            fields=line.split()
            if not fields or project is None:continue
            unit=unit_key(fields[0])
            price=re.search(r'(?<![\d,])(\d{1,3}(?:,\d{3}){2,})(?![\d,])',line)
            if unit and price:
                before=line[:price.start()].split()[1:]
                numbers=[float(v) for v in before if re.fullmatch(r'\d+\.\d+',v)]
                if len(numbers) not in [6,7]:raise ValueError(f'Unexpected house row: {line}')
                main,balcony,shared,ping,sqm,land=numbers[:6]
                if abs(sqm*.3025-ping)>.025:raise ValueError(f'Area units mismatch: {line}')
                quoted=int(price[1].replace(',',''))
                houses.append(dict(project=project,unit=unit,area_ping=ping,area_sqm=sqm,
                    land_ping=land,terrace_ping=numbers[6] if len(numbers)==7 else 0,
                    printed_component_sum_error_ping=main+balcony+shared-ping,
                    published_house_land_price=quoted,policy_cut_per_ping=spec['discount_per_ping'],
                    adjusted_house_land_price=quoted-ping*spec['discount_per_ping'],
                    source=source,source_page=page,source_published=spec['published']))
            m=re.match(r'^\s*(B[12])\s+(\d+)\s+(大|中|小|無障礙)\s+([\d,]+)',line)
            if m:
                parking.append(dict(project=project,level=m[1],space=int(m[2]),size=m[3],
                    price=int(m[4].replace(',','')),source=source,source_page=page))
    h=pd.DataFrame(houses);p=pd.DataFrame(parking)
    if h.empty or p.empty:raise ValueError(f'Missing schedule data: {source}')
    if h.duplicated(['project','unit']).any():raise ValueError('Ambiguous project/unit schedule')
    if p.duplicated(['project','level','space']).any():raise ValueError('Ambiguous parking schedule')
    return h,p


def main():
    houses=[];parking=[]
    for source,spec in SOURCES.items():
        h,p=extract((OUT/'sources'/f'{source}.txt').read_text(),source,spec)
        houses.append(h);parking.append(p)
    h=pd.concat(houses,ignore_index=True);p=pd.concat(parking,ignore_index=True)
    h.to_csv(OUT/'house_schedules.csv',index=False,encoding='utf-8-sig')
    p.to_csv(OUT/'parking_schedules.csv',index=False,encoding='utf-8-sig')
    print('Houses',h.groupby('project').size().to_dict())
    print('Parking',p.groupby('project').agg(count=('price','size'),median=('price','median'),minimum=('price','min'),maximum=('price','max')).to_dict('index'))


if __name__=='__main__':main()
