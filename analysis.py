"""
Copyright (C) 2019 Shane Steinert-Threlkeld
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
import numpy as np
import pandas as pd
from plotnine import *
import experiments


def gather_conditions(exp):
    all_res = []
    for base in exp:
        results = pd.read_csv(base + '/all_trials.csv')
        # read params, and make it have as many rows as results
        params = pd.concat(
            [pd.read_csv(base + '/params.csv')]*len(results),
            ignore_index=True)
        combined = pd.concat([results, params], axis=1)
        combined['name'] = base
        all_res.append(combined)
    return pd.concat(all_res, ignore_index=True)


def descriptives(results, group_vars=['s1pred', 'correct_id'], out_file=None):
    grouped = results.groupby(['name'] + group_vars)
    descriptives = grouped[['name', 'correct', 'reward']].describe()
    # reshape
    descriptives = descriptives.stack(0).reset_index()
    descriptives = descriptives.rename(columns={'level_3': 'measure'})
    descriptives['ci'] = (1.96 * descriptives['std'] /
                             np.sqrt(descriptives['count']))
    print(descriptives)
    # TODO: rename variables; here? in main/exp? elsewhere?
    plot = ggplot(descriptives[descriptives['measure'] == 'correct'],
                 aes(x='correct_id', fill='s1pred'))
    """
    plot += geom_col(aes(y='mean'), position=position_dodge(0.75), width=0.75)# , position='dodge')
    plot += geom_errorbar(
        aes(ymin='mean - ci', ymax='mean + ci'),
        width=0.1,
        position=position_dodge(0.75))
    """
    plot += geom_line(aes(y='mean', group='s1pred', color='s1pred'))
    plot += geom_point(aes(y='mean', color='s1pred'))
    plot += geom_errorbar(
        aes(ymin='mean - ci', ymax='mean + ci', color='s1pred'),
        width=0.05)
    # plot += facet_wrap('measure')
    if out_file:
        plot.save(out_file, width=8, height=6, dpi=300)
    else:
        print(plot)


def full_analysis(exp, group_vars=['s1pred', 'correct_id']):
    results = gather_conditions(exp)
    descriptives(results, group_vars, 'descriptives.png')


if __name__ == '__main__':

    for exp in experiments.all_exps:
        full_analysis(exp)
