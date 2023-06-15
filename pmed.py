import mip
from pprint import pprint
import pandas as pd
import numpy as np
from mip import Model, xsum
from typing import Dict
from collections import defaultdict as ddict


CRITERIAS = {
    "high_visibility": {
        "weight": 0.18,
        "attr": 'more'
    },
    "security": {
        "weight": 0.3,
        "attr": 'less'
    },
    "competitor_ext": {
        "weight": 0.1,
        "attr": 'more'
    },
    "economic_soc_level": {
        "weight": 0.04,
        "attr": 'more'
    },
    "ease_of_access": {
        "weight": 0.2,
        "attr": 'more'
    },
    "cost_of_installing_atm": {
        "weight": 0.16,
        "attr": 'less'
    },
    "company_agr": {
        "weight": 0.02,
        "attr": 'more'
    },
}


class PMedianProblem:
    def __init__(self,
                 name: str,
                 p: int,
                 distances: Dict[str, Dict[str, float]],
                 locations):

        data_dict = self.preprocess(p, distances, locations)
        self.D = data_dict['distances']
        self.K = data_dict['limit']
        self.I = data_dict['demandSet']
        self.J = data_dict['atmSet']
        self.m = data_dict['atm']
        self.n = data_dict['area']
        self.location_score = data_dict['location_score']

        self.model = Model(name=name, sense=mip.MINIMIZE, solver_name='CBC')

        self.X = [[self.model.add_var(name=f'X{i}{j}', var_type=mip.BINARY) for j in range(len(self.J))] for i in range(len(self.I))]
        self.Y = [self.model.add_var(name=f'Y{i}', var_type=mip.BINARY) for i in range(len(self.I))]

        self.model.objective = xsum(xsum(self.location_score[j] * self.D[i][j] * self.X[i][j] for j in range(len(self.J))) for i in range(len(self.I)))

        self.model.add_constr(xsum(self.Y[i] for i in range(len(self.J))) == self.m, name=f'(1)')
        for j in range(len(self.J)):
            self.model.add_constr(xsum(self.X[i][j] for i in range(len(self.I))) == 1, name=f'(2)-{j}')

        for i in range(len(self.I)):
            self.model.add_constr(xsum(self.X[i][j] for j in range(len(self.J))) <= len(self.J) * self.Y[i], name=f'(3)-{i}')

        for i in range(len(self.I)):
            for j in range(len(self.J)):
                self.model.add_constr(self.X[i][j] - self.Y[i] <= 0, name=f'(4)-{i}{j}')

        for i in range(len(self.I)):
            for j in range(len(self.J)):
                self.model.add_constr(self.D[i][j] * self.X[i][j] <= self.K, name=f'(5)-{i}{j}')

    def solve(self,
              verbose: bool = False):
        self.model.verbose = verbose
        self.model.optimize()
        if self.model.num_solutions:
            selected = set([i for i in range(len(self.Y)) if float(self.Y[i].x) >= 0.99])

            atm2demand = ddict(list)
            for i in self.I.difference(selected):
                assign = None
                smallest_distance = float('inf')
                for j in selected:
                    if self.D[i][j] < smallest_distance:
                        smallest_distance = self.D[i][j]
                        assign = j
                assert assign is not None, "assign to None"
                atm2demand[assign].append(i)
            result = atm2demand
        else:
            result = "No Solution Found!"
        return result

    @staticmethod
    def preprocess(p: int,
                   distances,
                   locations: dict):

        data_dict = dict()

        matrix = list()
        for source in sorted(distances.keys(), key=int):
            tmp = []
            for dest in sorted(distances[source].keys(), key=int):
                tmp.append(distances[source][dest])
            matrix.append(tmp)

        data_dict['distances'] = matrix
        data_dict['atmSet']    = set([int(atm) for atm in distances.keys()])
        data_dict['demandSet'] = set([int(demand) for demand in distances.keys()])
        data_dict['atm']   = int(p)
        data_dict['area']  = len(matrix)
        data_dict['limit'] = 100.0

        df = pd.DataFrame(columns=list(CRITERIAS.keys()))
        for location in locations.keys():
            tmp = [{crit: val for crit, val in locations[location].items()}]
            df = pd.concat([df, pd.DataFrame(tmp)], ignore_index=True)

        for crit_name in df:
            upper = df[crit_name].max()
            lower = df[crit_name].min()

            if CRITERIAS[crit_name]['attr'] == 'more':
                df[crit_name] = 9 - 8 * (upper-df[crit_name])/(upper-lower)
            elif CRITERIAS[crit_name]['attr'] == 'less':
                df[crit_name] = 9 - 8 * (df[crit_name]-lower)/(upper-lower)
            else:
                raise ValueError

        df['norm_score'] = 0
        for criteria in CRITERIAS.keys():
            crit_weight = CRITERIAS[criteria]['weight']
            df['norm_score'] += df[criteria] * crit_weight
        df['norm_score'] /= df['norm_score'].sum()
        data_dict['location_score'] = df['norm_score'].to_list()

        return data_dict


# if __name__ == "__main__":
#     with open('data.json', 'r') as f:
#         data = json.load(f)
#
#     pmed = PMedianProblem('ATM', data['p'], data['distances'], LOCATIONS)
#     pmed.solve()
