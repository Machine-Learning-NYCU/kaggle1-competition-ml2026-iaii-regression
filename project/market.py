"""Strictly causal, hierarchical empirical-Bayes comparable-sales statistics.

Each query only sees fitting-partition transactions in strictly earlier months.
Validation labels never enter this object; training rows cannot see their own
labels or other labels from their month. Trends extrapolate at most 12 months.
"""
import numpy as np
import pandas as pd


def month_index(x):
    return (x['transaction_year'].fillna(2000) * 12 + x['transaction_month'].fillna(1)).to_numpy(dtype=np.int64)


class CausalMarket:
    levels = ['city_district', 'road', 'project']

    def fit(self, x, log_unit_price):
        months = month_index(x)
        self.tables = {}
        for level in ['global'] + self.levels:
            keys = np.repeat('global', len(x)) if level == 'global' else x[level].to_numpy()
            a = pd.DataFrame({'key': keys, 'month': months, 'value': log_unit_price})
            a = a.groupby(['key', 'month'], sort=False)['value'].agg(['sum', 'count']).reset_index()
            tables = {}
            for key, part in a.groupby('key', sort=False):
                if key == '__MISSING__':
                    continue
                part = part.sort_values('month')
                tables[key] = (part.month.to_numpy(), np.r_[0., part['sum'].cumsum().to_numpy()],
                               np.r_[0., part['count'].cumsum().to_numpy()])
            self.tables[level] = tables
        return self

    def transform(self, x):
        months = month_index(x)
        result = x.copy()
        parent = np.full(len(x), np.log(200000.))  # fixed weak prior in NTD / m²
        for level in ['global'] + self.levels:
            keys = np.repeat('global', len(x)) if level == 'global' else x[level].to_numpy()
            mean = parent.copy()
            count = np.zeros(len(x))
            trend = np.zeros(len(x))
            age = np.full(len(x), 120.)
            for key, indices in pd.Series(np.arange(len(x))).groupby(keys, sort=False):
                if key not in self.tables[level]:
                    continue
                ix = indices.to_numpy()
                m, sums, counts = self.tables[level][key]
                end = np.searchsorted(m, months[ix], side='left')
                last = m[np.maximum(end - 1, 0)]
                # Recent window anchored at the last observable transaction, not future date.
                start = np.searchsorted(m, last - 11, side='left')
                middle = np.searchsorted(m, last - 5, side='left')
                n = counts[end] - counts[start]
                s = sums[end] - sums[start]
                n0 = counts[middle] - counts[start]
                n1 = counts[end] - counts[middle]
                old = (sums[middle] - sums[start]) / np.maximum(n0, 1)
                recent = (sums[end] - sums[middle]) / np.maximum(n1, 1)
                slope = np.clip((recent - old) / 6, -.015, .015)
                slope *= np.minimum(n0, n1) / (np.minimum(n0, n1) + 30)
                slope[(n0 < 3) | (n1 < 3)] = 0
                elapsed = np.clip(months[ix] - last, 0, 12)
                prior_strength = {'global': 20, 'city_district': 30, 'road': 15, 'project': 8}[level]
                estimate = (s + prior_strength * parent[ix]) / (n + prior_strength)
                estimate += slope * elapsed * n / (n + prior_strength)
                seen = end > 0
                mean[ix] = np.where(seen, estimate, parent[ix])
                count[ix] = np.where(seen, n, 0)
                trend[ix] = np.where(seen, slope, 0)
                age[ix] = np.where(seen, np.clip(months[ix] - last, 0, 120), 120)
            result[f'market_{level}'] = mean.astype(np.float32)
            result[f'market_{level}_count'] = np.log1p(count).astype(np.float32)
            result[f'market_{level}_trend'] = trend.astype(np.float32)
            result[f'market_{level}_age'] = age.astype(np.float32)
            parent = mean
        return result
