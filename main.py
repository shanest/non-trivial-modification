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
import itertools
import os
import numpy as np
import pandas as pd
import experiments


def prob_vector(states, pred, weights):
    pred_values = [state[pred] for state in states]
    weight_values = np.array([weights[strength] for strength in pred_values])
    return weight_values / sum(weight_values)


def assign_rewards(reward, pairs, eps=1e-6):
    # *args: pairs of (urn, choice)
    for urn, choice in pairs:
        urn[choice] += reward
        if urn[choice] <= 0:
            urn[choice] = eps


def run_trial(trial_num,
              num_preds=2, num_strengths=2, pos_reward=1.0, neg_reward=0.0,
              pred_cost=0.0, m2cost=0.0, strength_weights=[1.0, 2.0],
              s1pred=True, correct_id=False,
              num_iters=50000, num_eval=2000, out_dir='exp',
              **kwargs):  # kwargs just b/c how run_experiment calls this

    preds = list(range(num_preds))

    # TODO: refactor so that states are integers, and functions return the
    # relevant values, instead of building the structure into the states??
    states = list(itertools.product(range(num_strengths), repeat=num_preds))
    num_states = len(states)
    state_idxs = list(range(num_states))

    pred_prob_vectors = [prob_vector(states, pred, strength_weights)
                         for pred in preds]

    num_s1 = num_preds
    s1_msgs = list(range(num_s1))

    # TODO: generalize num_preds part here for general fn on state space?
    # TODO: sender1 gets pred ===> modification, but if gets state ===>
    # intersection; that could be the story
    num_urns_s1 = num_preds if s1pred else num_states
    sender1 = np.ones((num_urns_s1, num_s1))

    num_s2 = num_strengths
    s2_msgs = list(range(num_s2))
    # obj, msg1 --> msg2 prob
    sender2 = np.ones((num_states, num_s1, num_s2))

    receiver1 = np.ones((num_s1, num_s2, num_states))
    # msg1, obj --> obj prob

    results = []
    eval_trials = []

    for idx in range(num_iters + num_eval):

        # state_id = np.random.choice(state_idxs)
        pred = np.random.choice(preds)
        state_id = np.random.choice(state_idxs, p=pred_prob_vectors[pred])

        # TODO: get this same pattern but with sender1 seeing the entire state,
        # not just the predator.  is it possible???
        s1urn = sender1[pred] if s1pred else sender1[state_id]
        msg1 = np.random.choice(s1_msgs, p=s1urn/sum(s1urn))

        s2urn = sender2[state_id, msg1]
        msg2 = np.random.choice(s2_msgs, p=s2urn/sum(s2urn))

        r1urn = receiver1[msg1, msg2]
        state_guess = np.random.choice(state_idxs, p=r1urn/sum(r1urn))

        correct = (float(state_id == state_guess) if correct_id else
                   float(states[state_id][pred] == states[state_guess][pred]))
        results.append(correct)
        if idx % 1000 == 0:
            print('Last 1000: {}'.format(sum(results[-1000:])))
            print('state:\t{}\nguess:\t{}\npred:\t{}\nmsg:\t{}{}\nyes:\t{}'.format(
                states[state_id], states[state_guess], pred, msg1, msg2, correct))

        reward = correct
        reward = correct * (pos_reward + neg_reward) - neg_reward
        reward -= m2cost * float(msg2 == 1)
        # reward -= pred_cost*int(states[state_id][pred] == 1))
        # TODO: record training progress as well?
        if idx < num_iters:
            # reinforce
            assign_rewards(reward,
                           zip([s1urn, s2urn, r1urn],
                               [msg1, msg2, state_guess]))
        else:
            # evaluating
            eval_trials.append(
                (pred, states[state_id], msg1, msg2, state_guess, correct,
                 reward))

    # save things
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_base = '{}/trial_{}'.format(out_dir, trial_num)

    # eval trials
    eval_trials = pd.DataFrame(
        eval_trials,
        columns=['pred', 'state', 'msg1', 'msg2', 'state_guess',
                 'correct', 'reward'])
    eval_trials.to_csv(file_base + '_eval.csv')

    # agents
    np.save(file_base + '_sender1', sender1)
    np.save(file_base + '_sender2', sender2)
    np.save(file_base + '_receiver1', receiver1)

    # print things for good measure
    print(states)
    print(sender1)
    print(sender2)
    print(receiver1)

    # return avg correct and reward for eval
    return {col: sum(eval_trials[col]) / len(eval_trials[col])
            for col in ['correct', 'reward']}


def run_experiment(config):
    for base in config:
        results = []
        for trial in range(config[base]['num_trials']):
            results.append(
                run_trial(trial, out_dir=base, **config[base])
            )
            results[-1].update({'trial_num': trial})
        # write results and parametes
        pd.DataFrame(results).to_csv(base + '/all_trials.csv')
        # TODO: bug-fix: since strength_weights is a list and the rest are
        # scalars, pandas makes this DataFrame have multiple rows, one for each
        # value in strength_weights
        pd.DataFrame(config[base]).to_csv(base + '/params.csv')


if __name__ == '__main__':

    # TODO: argparse instead of / in addition to exp configs?
    for exp in experiments.all_exps:
        run_experiment(exp)
