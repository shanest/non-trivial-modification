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

exp1_base = {'num_trials': 3,
             'num_preds': 2,
             'num_strengths': 2,
             'pos_reward': 1.0,
             'neg_reward': 0.3,
             'pred_cost': 0.0,
             'm2cost': 0.2,
             'strength_weights': [1.0, 2.0],
             'num_iters': 5000}
exp1 = {
    'exp1/s1pred-correct_id': dict(
        exp1_base, **{'s1pred': True, 'correct_id': True}),
    'exp1/s1pred-no_correct_id': dict(
        exp1_base, **{'s1pred': True, 'correct_id': False}),
    'exp1/no_s1pred-correct_id': dict(
        exp1_base, **{'s1pred': False, 'correct_id': True}),
    'exp1/no_s1pred-no_correct_id': dict(
        exp1_base, **{'s1pred': False, 'correct_id': False}),
}

all_exps = [exp1]
