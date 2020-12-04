# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:41:10 2020

@authors:
Mateusz Woźniak s18182
Jakub Włoch s16912

reference: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# New Antecedent/Consequent objects hold universe variables and membership
# functions

standard = ctrl.Antecedent(np.arange(0, 11, 1), 'standard')
location = ctrl.Antecedent(np.arange(0, 11, 1), 'location')
year_of_construction = ctrl.Antecedent(np.arange(1945, 2021, 1), 'year_of_construction')

cost_per_square_meter = ctrl.Consequent(np.arange(0, 14000, 1), 'cost_per_square_meter')


# Auto-membership function population

standard.automf(3)
location.automf(3)
year_of_construction.automf(3)


# Custom membership functions

cost_per_square_meter['low'] = fuzz.trimf(cost_per_square_meter.universe, [0, 0, 7500])
cost_per_square_meter['medium'] = fuzz.trimf(cost_per_square_meter.universe, [0, 7500, 14000])
cost_per_square_meter['high'] = fuzz.trimf(cost_per_square_meter.universe, [7500, 14000, 14000])


# Fuzzy relationship between input and output variables

rule1 = ctrl.Rule(antecedent=((standard['poor'] & location['poor'] & year_of_construction['poor']) |
                              (standard['poor'] & location['poor']) |
                              (standard['average'] & location['poor'] & year_of_construction['poor'])),
                  consequent=cost_per_square_meter['low'])

rule2 = ctrl.Rule(antecedent=((standard['good'] & location['poor'] & year_of_construction['poor']) |
                              (standard['poor'] & location['poor'] & year_of_construction['good']) |
                              (standard['poor'] & location['average']) |
                              (standard['good'] & location['poor'] & year_of_construction['good']) |
                              (standard['average'] & location['average'] & year_of_construction['average']) |
                              (standard['poor'] & location['good'] & year_of_construction['poor'])),
                  consequent=cost_per_square_meter['medium'])

rule3 = ctrl.Rule(antecedent=((standard['good'] & location['good'] & year_of_construction['good']) |
                              (standard['poor'] & location['good'] & year_of_construction['good']) |
                              (standard['average'] & location['good'] & year_of_construction['average']) |
                              (standard['good'] & location['good']) |
                              (standard['average'] & location['good'] & year_of_construction['good'])),
                  consequent=cost_per_square_meter['high'])

apartment_pricing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
apartment_pricing = ctrl.ControlSystemSimulation(apartment_pricing_ctrl)


# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API

apartment_pricing.input['standard'] = 1
apartment_pricing.input['location'] = 6
apartment_pricing.input['year_of_construction'] = 2015


# Crunch the numbers

apartment_pricing.compute()
print(apartment_pricing.output['cost_per_square_meter'])

year_of_construction.view()
