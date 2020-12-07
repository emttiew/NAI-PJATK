# Simple apartment pricing with fuzzy logic

## Problem desription

Fuzzy Logic is a methodology predicated on the idea that the “truthiness” of something can be expressed over a continuum. This is to say that something isn’t true or false but instead partially true or partially false.

Using fuzzy logic with python skfuzzy toolkit:

Apartment pricing script creates a controller which estimates the cost per square meter.
### Input
* standard [1-10]
Standard / quality of the apartment

* year of construction [1945 - 2020]
* location [1-10]
How good is the location of the apartment eg. its close to the sea with safe area. 
This could be another fuzzy logic problem, but in this example we consider that it is allready estimated. 

### Output
* cost per square meter [0 - 14000] (PLN)

## Membership functions:
### standard.view()
<img src="https://github.com/emttiew/NAI-PJATK/blob/master/FuzzyLogicApartmentPricing/resources/standard_view.png">

### year_of_construction.view()
<img src="https://github.com/emttiew/NAI-PJATK/blob/master/FuzzyLogicApartmentPricing/resources/year_view.png">

### location.view()
<img src="https://github.com/emttiew/NAI-PJATK/blob/master/FuzzyLogicApartmentPricing/resources/location_view.png">

### cost_per_square_meter.view()
<img src="https://github.com/emttiew/NAI-PJATK/blob/master/FuzzyLogicApartmentPricing/resources/output_view.png">

## Output examples
### Low
#### input:
[standard] = 0

[location] = 0

[year_of_construction] = 1980

<img src="https://github.com/emttiew/NAI-PJATK/blob/master/FuzzyLogicApartmentPricing/resources/low_output.png">

### Medium
#### input:
[standard] = 3

[location] = 6

[year_of_construction] = 2015

<img src="https://github.com/emttiew/NAI-PJATK/blob/master/FuzzyLogicApartmentPricing/resources/medium_output.png">

### High
#### input:
[standard] = 9 

[location] = 10 

[year_of_construction] = 2019

<img src="https://github.com/emttiew/NAI-PJATK/blob/master/FuzzyLogicApartmentPricing/resources/high_output.png">
