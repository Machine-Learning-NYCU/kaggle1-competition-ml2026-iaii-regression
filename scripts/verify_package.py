#!/usr/bin/env python3
"""Verify the light submission package without private data or ML dependencies."""
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads((ROOT/path).read_text())


def main():
    copied=read_json('copied_sources.json')
    for name, expected in copied.items():
        assert digest(ROOT/name)==expected['sha256'], f'Changed copied evidence/source: {name}'
    expected={
        6:'53bde5d7fa8c97d95ab8897c905b38cf01e16cda6a6eea48f8605a5b848aadd6',
        7:'c6984c5d86f20e794c774d65c52bdf46e4c0b2c7012ad57ff33f43982cc8eda7',
    }
    frames={}
    for version, sha in expected.items():
        path=ROOT/f'submissions/submission1017_v{version}.csv'
        assert digest(path)==sha
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader=csv.DictReader(handle)
            assert reader.fieldnames==['編號','總價元']
            rows=list(reader)
        assert len(rows)==47073
        ids=[r['編號'] for r in rows]
        prices=[int(r['總價元']) for r in rows]
        assert len(set(ids))==47073 and all(p>0 for p in prices)
        frames[version]=(ids,prices)
    assert frames[6][0]==frames[7][0]
    changed=sum(x!=y for x,y in zip(frames[6][1],frames[7][1]))
    assert changed==28374
    history=read_json('evidence/leaderboard_history.json')
    score={r['submitted_filename']:Decimal(str(r['score_mae'])) for r in history}
    old=score['submission1017_v6.csv'];new=score['submission1017_v7.csv']
    assert old-new==Decimal('29043.5252')
    assert min(score,key=score.get)=='submission1017_v7.csv'
    seeds=read_json('evidence/geography_v1/seed_check.json')
    confirmation=read_json('evidence/geography_v1/confirmation.json')
    historical=[]
    for name,record in list(seeds['folds'].items())+[('confirmation',confirmation['metrics'])]:
        m=record['all']
        gain=100*(1-m['candidate_mae']/m['baseline_mae'])
        assert gain>0
        historical.append({'window':name,'improvement_percent':gain})
    policy=read_json('evidence/policy_generalization_v1/audit.json')
    policy_scores={}
    for name,v in policy['validation'].items():
        m=v['fixed'];gain=100*(1-m['row_mae']/m['baseline_mae'])
        assert abs(gain-m['improvement_pct'])<1e-9
        policy_scores[name]={'offer_mae':m['row_mae'],'improvement_percent':gain}
    assert policy['inference']['actual_excluded_unit_offer_labels_loaded'] is False
    assert policy['deployment_model_changed'] is False
    assert policy['new_submission_created'] is False
    assert read_json('evidence/policy_generalization_v1/verification.json')['tests_passed']==46
    result={'status':'passed','copied_files_verified':len(copied),'csv_rows_each':47073,
            'v6_v7_changed_rows':changed,'public_improvement_ntd':str(old-new),
            'historical_improvements':historical,'policy_offer_results':policy_scores,
            'private_score_known':False,'check_scope':'Hash/schema/aggregate consistency; not model retraining or fresh evaluation.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
