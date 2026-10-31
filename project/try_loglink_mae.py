"""A pre-evaluation experiment: exact price MAE with a positive log-link ratio."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from features import TARGET, roc_date
from market import CausalMarket
from mae_models import anchor_price, SCALE
from train import scores, subset

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/loglink_mae_experiment'


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_parquet(ROOT/'data/train.parquet')
    y = pd.to_numeric(raw[TARGET], errors='coerce')
    date = roc_date(raw['交易年月日'])
    raw = raw.loc[y.notna() & (y > 0) & date.between('2000-01-01', '2024-08-10')].copy()
    raw['_date'] = date.loc[raw.index]
    raw = raw.sort_values('_date', kind='stable').reset_index(drop=True)
    y = pd.to_numeric(raw[TARGET]).to_numpy(float)
    date = raw['_date']
    x, cats, groups, offset = joblib.load(ROOT/'artifacts/run_mae_v1/features.joblib')
    fit = np.where(date <= '2023-09-30')[0]
    tune = np.where((date > '2023-09-30') & (date <= '2023-12-31'))[0]
    pre = np.where(date <= '2023-12-31')[0]
    cal = np.where((date > '2023-12-31') & (date <= '2024-04-30'))[0]
    market = CausalMarket().fit(subset(x, fit), np.log(y[fit])-offset[fit])
    xf, xt = market.transform(subset(x, fit)), market.transform(subset(x, tune))
    bf, bt = anchor_price(xf, offset[fit]), anchor_price(xt, offset[tune])
    params = dict(iterations=4000, learning_rate=.04, depth=8, loss_function='LogLinQuantile:alpha=0.5',
        eval_metric='LogLinQuantile:alpha=0.5', l2_leaf_reg=12, random_strength=.5,
        bootstrap_type='Bayesian', bagging_temperature=.5, max_ctr_complexity=1,
        one_hot_max_size=10, has_time=True, task_type='GPU', devices='0', thread_count=16,
        random_seed=2026, verbose=500, allow_writing_files=False)
    model = CatBoostRegressor(**params)
    model.fit(Pool(xf, y[fit]/bf, cat_features=cats, weight=bf/SCALE),
              eval_set=Pool(xt, y[tune]/bt, cat_features=cats, weight=bt/SCALE), early_stopping_rounds=200)
    rounds = model.tree_count_
    market = joblib.load(ROOT/'artifacts/run_mae_v1/market_dev.joblib')
    xp, xc = market.transform(subset(x, pre)), market.transform(subset(x, cal))
    bp, bc = anchor_price(xp, offset[pre]), anchor_price(xc, offset[cal])
    params['iterations'] = rounds
    model = CatBoostRegressor(**params)
    model.fit(Pool(xp, y[pre]/bp, cat_features=cats, weight=bp/SCALE))
    p = bc*np.exp(np.clip(model.predict(xc, prediction_type='RawFormulaVal'), -30, 30))
    record = {'loss': 'weighted LogLinQuantile alpha=.5 equals price MAE up to a fixed multiplier',
              'iterations': rounds, 'calibration': scores(y[cal], p), 'audit_not_used': True}
    (OUT/'calibration.json').write_text(json.dumps(record, indent=2))
    np.savez_compressed(OUT/'calibration_predictions.npz', y=y[cal], predictions=p)
    joblib.dump({'model': model, 'params': params}, OUT/'dev_model.joblib', compress=3)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
