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
    for condition in exp['conditions']:
        base = exp['name'] + '/' + condition
        results = pd.read_csv(base + '/all_trials.csv')
        # read params, and make it have as many rows as results
        params = pd.concat(
            [pd.read_csv(base + '/params.csv')]*len(results),
            ignore_index=True)
        combined = pd.concat([results, params], axis=1)
        combined['name'] = condition
        combined['how_nontrivial'] = measure_nontrivial(exp, condition)
        all_res.append(combined)
    return pd.concat(all_res, ignore_index=True)


def neg_ci(arr):
    return np.mean(arr) - 1.96*np.std(arr)/np.sqrt(len(arr))


def pos_ci(arr):
    return np.mean(arr) + 1.96*np.std(arr)/np.sqrt(len(arr))


def bar_plot(data, y_var, out_file=None):

    plot = (ggplot(data, aes(x='correct_id', y=y_var))
                   + geom_point(aes(color='s1pred'),
                                alpha=0.4, position=position_dodge(0.7))
                   + stat_summary(aes(fill='s1pred', color='s1pred'),
                                  fun_y=np.mean, geom='bar', width=0.7, alpha=0.2,
                                  position=position_dodge(0.7), size=1.25)
                   + stat_summary(aes(color='s1pred'),
                                  geom='errorbar',
                                  fun_ymin= neg_ci, fun_ymax=pos_ci,
                                  position=position_dodge(0.7),
                                  width=0.1, size=1.25)
           )
    if out_file:
        pass
    else:
        print(plot)


def line_plot(data, y_var, out_file=None):

    plot = (ggplot(data, aes(x='correct_id', y=y_var))
                   + geom_point(aes(color='s1pred'),
                                alpha=0.4, position=position_jitter(0.05, 0.05))
                   + stat_summary(aes(fill='s1pred', color='s1pred',
                                      group='s1pred'),
                                  fun_y=np.mean, geom='line', size=0.75)
                   + stat_summary(aes(color='s1pred'),
                                  geom='errorbar',
                                  fun_ymin= neg_ci, fun_ymax=pos_ci,
                                  # position=position_dodge(0.7),
                                  width=0.1, size=0.75)
           )
    if out_file:
        pass
    else:
        print(plot)


def descriptives(results, group_vars=['s1pred', 'correct_id'],
                 describe_vars=['correct', 'reward', 'how_nontrivial'],
                 measure_vars=['correct', 'how_nontrivial'], out_file=None):

    bar_plot(results, 'correct')
    line_plot(results, 'how_nontrivial')

    grouped = results.groupby(['name'] + group_vars)
    descriptives = grouped[['name'] + describe_vars].describe()
    # reshape
    descriptives = descriptives.stack(0).reset_index()
    descriptives = descriptives.rename(columns={'level_3': 'measure'})
    descriptives['ci'] = (1.96 * descriptives['std'] /
                             np.sqrt(descriptives['count']))
    print(descriptives)
    # TODO: rename variables; here? in main/exp? elsewhere?
    plot = ggplot(descriptives[descriptives['measure'].isin(measure_vars)],
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
    if len(measure_vars) > 1:
        plot += facet_wrap('measure', scales='free_y')
    if out_file:
        plot.save(out_file, width=10, height=6, dpi=300)
    else:
        print(plot)


def nontrivial(sender, p_threshold=0.75):
    """ Whether one of sender's choices depends on the message it received from
    a previous sender.

    Args:
        sender: nparray of shape [states, send1_msgs, send2_msgs]
        p_threshold:    prob when a sender is considered to make a given choice
    """
    num_states, num_s1, num_s2 = sender.shape
    # normalize urns to probabilities
    sender /= sender.sum(axis=2, keepdims=True)
    choices = sender > p_threshold

    def conditioned_choice(state_choices):
        """ Whether the sender sends different messages based on incoming
        message in a given state. """
        # TODO: check that this generalizes beyond the 2x2 case
        some = np.logical_xor.reduce(state_choices)
        # conditions on at least 2 incoming messages, to rule out case where
        # there's only ever one incoming message
        return np.sum(some) >= 2

    # normalize to a "percentage"
    return sum(float(conditioned_choice(choices[state]))
               for state in range(num_states))


def measure_nontrivial(exp, condition):
    filenames = ['{}/{}/trial_{}_sender2.npy'.format(
        exp['name'], condition, trial)
        for trial in range(exp['conditions'][condition]['num_trials'])]
    how_non = [nontrivial(np.load(fn)) for fn in filenames]
    return how_non


def full_analysis(exp, group_vars=['s1pred', 'correct_id']):
    results = gather_conditions(exp)
    descriptives(results, group_vars=group_vars,
                 out_file=(exp['name'] + '/descriptives.png'))


if __name__ == '__main__':

    for exp in experiments.all_exps:
        full_analysis(exp)
