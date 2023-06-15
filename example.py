import mip
import json
from mip import Model, xsum


class PMedianProblem:
    def __init__(self,
                 name: str,
                 input_file: str):

        data_dict = self.preprocess(input_file)
        self.W = data_dict['weight']
        self.D = data_dict['distance']
        self.K = data_dict['limit']
        self.I = data_dict['demandSet']
        self.J = data_dict['atmSet']
        self.m = data_dict['atm']
        self.n = data_dict['area']
        self.criterias = data_dict['criterias']
        self.location_score = data_dict['locations']

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
            selected = [i for i in range(len(self.Y)) if float(self.Y[i].x) >= 0.99]
            print("Number of Solutions:", self.model.num_solutions)
            print(selected)

            other_areas = self.J - set(selected)
            for area in other_areas:
                pass
        else:
            print("No Solution Found!")

    @staticmethod
    def preprocess(file):
        with open(file, 'r') as f:
            data = json.load(f)

        I = set(data['atmSet'])
        J = set(data['demandSet'])

        distance = [[0 if i == j
                     else data['distance'][i][j - i - 1] if j > i
                     else data['distance'][j][i - j - 1]
                     for j in J] for i in I]

        data['atmSet'] = I
        data['demandSet'] = J
        data['distance'] = distance

        criterias = data['criterias']
        for k in criterias.keys():
            criteria_attr = criterias[k]['attr']
            criteria_data = criterias[k]['data']

            lower = min(criteria_data)
            upper = max(criteria_data)
            if criteria_attr == 'more':
                criteria_data = list(map(lambda x: 9 - 8 * (upper-x)/(upper-lower), criteria_data))
            elif criteria_attr == 'less':
                criteria_data = list(map(lambda x: 9 - 8 * (x-lower)/(upper-lower), criteria_data))
            else:
                raise ValueError

            criterias[k]['data'] = criteria_data
        data['criterias'] = criterias

        data['locations'] = list()
        weights = data['weight']
        total = 0
        for i in I:
            total_in_location = 0
            for w, score in zip(weights, [criterias[k]['data'][i] for k in criterias.keys()]):
                total_in_location += w * score
            data['locations'].append(total_in_location)
            total += total_in_location
        data['locations'] = list(map(lambda x: x/total, data['locations']))

        return data


if __name__ == "__main__":
    problem = PMedianProblem('ATM-Solver', 'data_old.json')
    problem.solve()
