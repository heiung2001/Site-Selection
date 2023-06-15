import mip
from pprint import pprint
import pandas as pd
from mip import Model, xsum
from typing import Dict


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
        pprint(self.D)

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
            result = [i for i in range(len(self.Y)) if float(self.Y[i].x) >= 0.99]
            # print("Number of Solutions:", self.model.num_solutions)
            # print(selected)

            # other_areas = self.J - set(selected)
            # for area in other_areas:
            #     pass
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
#     # data = preprocess(data['p'], data['distances'])
#     LOCATIONS = {
#              "0": {"company_agr": 2,
#                  "competitor_ext": 4,
#                  "cost_of_installing_atm": 12000,
#                  "ease_of_access": 3,
#                  "economic_soc_level": 5000,
#                  "high_visibility": 10,
#                  "security": 10},
#              "1": {"company_agr": 5,
#                  "competitor_ext": 7,
#                  "cost_of_installing_atm": 30000,
#                  "ease_of_access": 4,
#                  "economic_soc_level": 7000,
#                  "high_visibility": 2,
#                  "security": 5},
#              "2": {"company_agr": 3,
#                  "competitor_ext": 11,
#                  "cost_of_installing_atm": 70000,
#                  "ease_of_access": 7,
#                  "economic_soc_level": 12000,
#                  "high_visibility": 6,
#                  "security": 40},
#              "3": {"company_agr": 2,
#                  "competitor_ext": 13,
#                  "cost_of_installing_atm": 80000,
#                  "ease_of_access": 1,
#                  "economic_soc_level": 3500,
#                  "high_visibility": 5,
#                  "security": 70},
#              "4": {"company_agr": 1,
#                  "competitor_ext": 20,
#                  "cost_of_installing_atm": 20000,
#                  "ease_of_access": 2,
#                  "economic_soc_level": 1500,
#                  "high_visibility": 8,
#                  "security": 14},
#              "5": {"company_agr": 3,
#                  "competitor_ext": 9,
#                  "cost_of_installing_atm": 45000,
#                  "ease_of_access": 1,
#                  "economic_soc_level": 700,
#                  "high_visibility": 30,
#                  "security": 8},
#              "6": {"company_agr": 5,
#                  "competitor_ext": 20,
#                  "cost_of_installing_atm": 60000,
#                  "ease_of_access": 4,
#                  "economic_soc_level": 12000,
#                  "high_visibility": 1,
#                  "security": 2},
#              "7": {"company_agr": 0,
#                  "competitor_ext": 10,
#                  "cost_of_installing_atm": 100000,
#                  "ease_of_access": 1,
#                  "economic_soc_level": 3500,
#                  "high_visibility": 40,
#                  "security": 9},
#              "8": {"company_agr": 1,
#                  "competitor_ext": 12,
#                  "cost_of_installing_atm": 23000,
#                  "ease_of_access": 3,
#                  "economic_soc_level": 8400,
#                  "high_visibility": 25,
#                  "security": 11},
#              "9": {"company_agr": 5,
#                  "competitor_ext": 0,
#                  "cost_of_installing_atm": 50000,
#                  "ease_of_access": 1,
#                  "economic_soc_level": 1000,
#                  "high_visibility": 11,
#                  "security": 23},
#              "10": {"company_agr": 3,
#                   "competitor_ext": 3,
#                   "cost_of_installing_atm": 40091,
#                   "ease_of_access": 1,
#                   "economic_soc_level": 3012,
#                   "high_visibility": 3,
#                   "security": 3},
#              "11": {"company_agr": 2,
#                   "competitor_ext": 2,
#                   "cost_of_installing_atm": 18696,
#                   "ease_of_access": 1,
#                   "economic_soc_level": 4240,
#                   "high_visibility": 10,
#                   "security": 5},
#              "12": {"company_agr": 4,
#                   "competitor_ext": 5,
#                   "cost_of_installing_atm": 41866,
#                   "ease_of_access": 1,
#                   "economic_soc_level": 10802,
#                   "high_visibility": 16,
#                   "security": 8},
#              "13": {"company_agr": 0,
#                   "competitor_ext": 3,
#                   "cost_of_installing_atm": 42192,
#                   "ease_of_access": 1,
#                   "economic_soc_level": 10854,
#                   "high_visibility": 9,
#                   "security": 2},
#              "14": {"company_agr": 3,
#                   "competitor_ext": 8,
#                   "cost_of_installing_atm": 35671,
#                   "ease_of_access": 1,
#                   "economic_soc_level": 7290,
#                   "high_visibility": 1,
#                   "security": 16},
#              "15": {"company_agr": 5,
#                   "competitor_ext": 7,
#                   "cost_of_installing_atm": 34292,
#                   "ease_of_access": 1,
#                   "economic_soc_level": 1889,
#                   "high_visibility": 2,
#                   "security": 9},
#              "16": {"company_agr": 3,
#                   "competitor_ext": 10,
#                   "cost_of_installing_atm": 12671,
#                   "ease_of_access": 2,
#                   "economic_soc_level": 2402,
#                   "high_visibility": 13,
#                   "security": 4},
#              "17": {"company_agr": 1,
#                   "competitor_ext": 1,
#                   "cost_of_installing_atm": 19616,
#                   "ease_of_access": 1,
#                   "economic_soc_level": 10527,
#                   "high_visibility": 6,
#                   "security": 2},
#              "18": {"company_agr": 5,
#                   "competitor_ext": 2,
#                   "cost_of_installing_atm": 48230,
#                   "ease_of_access": 5,
#                   "economic_soc_level": 10524,
#                   "high_visibility": 5,
#                   "security": 18},
#              "19": {"company_agr": 4,
#                   "competitor_ext": 5,
#                   "cost_of_installing_atm": 45296,
#                   "ease_of_access": 8,
#                   "economic_soc_level": 6599,
#                   "high_visibility": 8,
#                   "security": 8}
#             }
#     pmed = PMedianProblem('ATM', data['p'], data['distances'], LOCATIONS)
#     pmed.solve()
