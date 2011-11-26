"""
This module contains exact two-sample tests.

TODO: use hypothesized values, report variances, report confidence intervals,
allow fisher test to work if more than 2 outcomes
"""

import csv, new
from itertools import combinations
from scipy import stats
import statistical_tests
from statistical_tests.exact import ExactTest

class ExactTwoSampleDistribution(object):
    """
    An exact distribution for a matched-pair test
    """
    def __init__(self, replacement_values, smaller_group_size):
        # validate replacement_values
        if not isinstance(replacement_values, list) and not isinstance(replacement_values, tuple):
            raise TypeError('replacement_values must be a list or tuple')
        # create the exact sampling distribution of all possible combinations
        n = len(replacement_values)
        self.__sampling_distribution = list()
        for vals in combinations(replacement_values, smaller_group_size):
            self.__sampling_distribution.append(sum(vals))
        self.__sampling_distribution.sort()
        self.__individual_probability_mass = 1/float(len(self.__sampling_distribution))
        # retain an ordered list of possible scores in the sampling distribution
        self.__possible_scores = list(set(self.__sampling_distribution))
        self.__possible_scores.sort()
        # store frequency and probability mass tables for possible scores
        self.__score_frequencies = dict((x, self.__sampling_distribution.count(x)) for x in self.__possible_scores)
        self.__score_probabilities = dict((x, self.__score_frequencies[x]*self.__individual_probability_mass) for x in self.__possible_scores)
        # store the expected value of the distribution
        self.__expected_value = sum(replacement_values)/float(n) * smaller_group_size
        # retain an ordered list of delta values for possible scores
        self.__delta_values = [x-self.__expected_value for x in self.__possible_scores]
        # retain a dictionary of distance:score pairs
        self.__delta_lookup = dict()
        for i in range(len(self.__possible_scores)):
            self.__delta_lookup.setdefault(abs(self.__delta_values[i]), []).append(self.__possible_scores[i])
        # retain a reverse ordered list of delta values by absolute value
        self.__delta_by_abs = self.__delta_lookup.keys()
        self.__delta_by_abs.sort(reverse=True)
        # store the variance of the distribution
        self.__variance = None
        # store the mean of the distribution
        self.__mean = sum(x for x in self.__sampling_distribution)/float(len(self.__sampling_distribution))
    def possible_scores(self):
        """
        Return the list of possible scores in the distribution
        """
        return self.__possible_scores
    def var(self):
        """
        Return the variance of this distribution
        """
        return self.__variance
    def std(self):
        """
        Return the standard deviation of this distribution
        """
        return math.sqrt(self.__variance) if self.__variance else None
    def median(self):
        """
        The expected value of the sampling distribution
        """
        return self.__expected_value
    def mean(self):
        """
        The mean of the sampling distribution
        """
        return self.__mean
    def frequency_of(self, outcome):
        """
        Return the number of occurrences of outcome in the sampling
        distribution
        """
        return self.__score_frequencies.setdefault(outcome, 0)
    def pmf(self, outcome):
        """
        Return the probability mass of a specified outcome
        """
        return self.__score_probabilities.setdefault(outcome, 0)
    def cvs1(self, alpha=0.05):
        """
        Get (LCV, UCV) for one-tailed tests
        """
        num_errors = int(alpha*len(self.__sampling_distribution))
        f = 0; i = -1
        while f+self.__score_frequencies[self.__possible_scores[i+1]] <= num_errors:
            i += 1
            f += self.__score_frequencies[self.__possible_scores[i]]
        lcv = self.__possible_scores[i] if i >= 0 else None
        f = 0; i = len(self.__possible_scores)
        while f+self.__score_frequencies[self.__possible_scores[i-1]] <= num_errors:
            i -= 1
            f += self.__score_frequencies[self.__possible_scores[i]]
        ucv = self.__possible_scores[i] if i < len(self.__possible_scores) else None
        return lcv, ucv
    def cvs2(self, alpha=0.05):
        """
        Get (LCV, UCV) for two-tailed tests
        """
        num_errors = int(alpha*len(self.__sampling_distribution))
        error_scores = list()
        f = 0
        for x in self.__delta_by_abs:
            f += sum(self.__score_frequencies[y] for y in self.__delta_lookup[x])
            if f > num_errors: break
            error_scores += self.__delta_lookup[x]
        lcv, ucv = None, None
        try: lcv = max([x for x in error_scores if x < self.__expected_value])
        except: pass
        try: ucv = min([x for x in error_scores if x > self.__expected_value])
        except: pass
        return lcv, ucv
    def p_lower(self, outcome):
        """
        Return probability of getting a value less than or equal to outcome
        """
        return sum(self.pmf(self.__possible_scores[i]) for i in range(len(self.__possible_scores)) if self.__delta_values[i] <= outcome-self.__expected_value)
    def p_upper(self, outcome):
        """
        Return probability of getting a value greater than or equal to outcome
        """
        return sum(self.pmf(self.__possible_scores[i]) for i in range(len(self.__possible_scores)) if self.__delta_values[i] >= outcome-self.__expected_value)
    def p_two(self, outcome):
        """
        Return probability of getting a value whose distance to the expected
        value is greater or equal to that of outcome
        """
        return sum(self.pmf(self.__possible_scores[i]) for i in range(len(self.__possible_scores)) if abs(self.__delta_values[i]) >= abs(outcome-self.__expected_value))

class ExactTwoSampleTest(ExactTest):
    """
    A base class for exact two-sample tests
    """
    def __init__(self,
                 group_key='Group',
                 outcome_key='Outcome',
                 groups = {'1':[], '2':[]},
                 population_of_interest=None,
                 *args,
                 **kwargs
                 ):
        ExactTest.__init__(self, *args, **kwargs)
        
        # validate input parameters
        if self.input_file:
            if not group_key or not outcome_key:
                raise ValueError('Keys must be defined for group and outcome columns.')
            groups = list(set(entry[group_key] for entry in csv.DictReader(open(self.input_file))))
            group_of_interest = 0
            try: group_of_interest = groups.index(population_of_interest)
            except: population_of_interest = groups[0]
            g1 = list()
            g2 = list()
            for row in csv.DictReader(open(self.input_file)):
                if row[group_key] == groups[group_of_interest]: g1.append(row[outcome_key])
                else: g2.append(row[outcome_key])
                if not row[group_key]: sys.stderr.write('Warning: row contains no group assignment in file %s: %s.\n'%(self.input_file, row))
            groups = { groups[group_of_interest]:g1, 'Not %s'%groups[group_of_interest]:g2 }
        if not isinstance(groups, dict):
            raise TypeError('Groups must be supplied as a dictionary of pairs Group Label : Outcome Values.')
        if len(groups.keys()) != 2:
            raise ValueError('There must be exactly 2 groups; found %s'%len(groups.keys()))
        for k in groups.keys():
            if not groups[k]:
                raise ValueError('No scores found for Group: %s.'%k)
            if not self.__class__.is_array_like(groups[k]):
                raise TypeError('Scores for Group: %s must be a list or tuple.'%k)
            groups[k] = list(groups[k])
        if not population_of_interest: population_of_interest = groups.keys()[0]
        self.population_of_interest = population_of_interest
        
        # store group scores as a data attribute
        self.groups = groups
        # set the test parameter to median of the population of interest
        self.test_parameter = 'M %s'%population_of_interest
        # store the outcome key as a data attribute
        self.outcome_key = outcome_key
        # store the key of the smaller group
        self.smaller_group_key = groups.keys()[0] if len(groups[groups.keys()[0]]) < len(groups[groups.keys()[1]]) else groups.keys()[1]
        #self.hypothesized_value = 'M %s'%(groups.keys())[(groups.keys()).index(self.smaller_group_key)-1] # TODO: need to adjust when taking actual hypothesized value into account
        self.hypothesized_value = 'M %s'%groups.keys()[groups.keys().index(population_of_interest)-1] 
        # order all of the scores based on rank
        try: self.combined_scores = [float(x) for k in groups.keys() for x in groups[k]]
        except: self.combined_scores = [x for k in groups.keys() for x in groups[k]]
        self.combined_scores.sort()
        # store sample size
        self.n = len(self.combined_scores)
    
    def create_distribution(self):
        """
        Create an exact distribution from the values in the smaller group
        """
        self.distribution = ExactTwoSampleDistribution(self.replacement_values, len(self.groups[self.smaller_group_key]))
    
    def validate_numericality(self):
        """
        Validate the numericality of the data
        """
        try:
            for k in self.groups.keys(): self.groups[k] = [float(x) for x in self.groups[k]]
        except:
            raise ValueError('Non-numeric data encountered in category %s.'%self.outcome_key)
    
    def perform_test(self):
        """
        Perform a two-sample test using all of the information stored in the
        data attributes
        """
        # gather test results
        if self.results.test_statistic is None: # e.g., FisherExactTest sets test_statistic manually
            self.results.test_statistic = sum(self.smaller_group_replacement_values)
            self.results.var = self.distribution.var()
            self.results.std = self.distribution.std()
        
        # get the p-values
        self.results.p_lower = self.distribution.p_lower(self.results.test_statistic)
        self.results.p_upper = self.distribution.p_upper(self.results.test_statistic)
        self.results.p_two = self.distribution.p_two(self.results.test_statistic)
        
        # get the critical values
        self.results.lcv1, self.results.ucv1 = self.distribution.cvs1(self.alpha)
        self.results.lcv2, self.results.ucv2 = self.distribution.cvs2(self.alpha)
        
        # report actual type i error rates
        self.results.exact_alpha_lower  = self.distribution.p_lower(self.results.lcv1) if self.results.lcv1 is not None else 0
        self.results.exact_alpha_upper = self.distribution.p_upper(self.results.ucv1) if self.results.ucv1 is not None else 0
        self.results.exact_alpha_two = (
                                        (self.distribution.p_lower(self.results.lcv2) if self.results.lcv2 is not None else 0) + 
                                        (self.distribution.p_upper(self.results.ucv2) if self.results.ucv2 is not None else 0)
                                        )
        
        # print results if not in silent mode
        if not self.is_silent: print self.printable_test_results()

class FisherExactTest(ExactTwoSampleTest):
    """
    Fisher's exact test, which compares a population of interest to its
    complement across groups specified by group_key
    """
    def __init__(self, *args, **kwargs):
        ExactTwoSampleTest.__init__(self, *args, **kwargs)
        self.possible_outcomes = list(set(self.combined_scores))
        if len(self.possible_outcomes) > 2:
            raise KeyError('Fisher Exact Test requires a dichotomous variable. %s possible outcomes were found.'%len(self.possible_outcomes))
        # manually override some data attributes for output
        self.test_parameter = 'Population of %s'%self.population_of_interest
        self.hypothesized_value = 'Population of %s'%self.groups.keys()[self.groups.keys().index(self.population_of_interest)-1]
        # create the distribution as usual
        self.create_distribution()
        # manually set some results output to override ordinary statistical test
        self.results.test_statistic = self.distribution.__delta_values[self.groups[self.population_of_interest].count(self.possible_outcomes[0])]
        self.results.var = abs(self.results.test_statistic)*(1.0-abs(self.results.test_statistic)) # TODO: is this right?
        self.results.std = math.sqrt(self.results.var)
        # perform the test as usual
        self.perform_test()
        # generate a probability table if requested
        self.create_output_file_path_if_needed()
        if len(self.output_file) > 0:
            headers = ['frequency of %s in population %s'%(self.possible_outcomes[0], self.population_of_interest),
                       'population of %s'%self.population_of_interest,
                       'population of %s'%self.groups.keys()[self.groups.keys().index(self.population_of_interest)-1],
                       'delta'] + self.get_probability_table_probability_headers()
            funcs = [lambda x: x,
                     lambda x: x/float(self.distribution.col1_sum),
                     lambda x: (self.distribution.row1_sum-x)/float(self.n-self.distribution.col1_sum),
                     lambda x: self.distribution.__delta_values[x],
                     lambda x: self.distribution.pmf(x),
                     lambda x: self.distribution.p_lower(self.distribution.__delta_values[x]),
                     lambda x: self.distribution.p_upper(self.distribution.__delta_values[x]),
                     lambda x: self.rejection_region_symbols[self.is_in_rejection_two(self.distribution.__delta_values[x])],
                     lambda x: self.rejection_region_symbols[self.is_in_rejection_low(self.distribution.__delta_values[x])],
                     lambda x: self.rejection_region_symbols[self.is_in_rejection_upper(self.distribution.__delta_values[x])]
                     ]
            entries = list()
            for x in range(len(self.distribution.__delta_values)):
                entries.append(dict((headers[i],funcs[i](x)) for i in range(len(headers))))
            self.save_probability_table(headers, entries)
        
    def create_distribution(self):
        """
        Create the exact hypergeometric distribution for Fisher's exact test
        """
        # organize the data into a 2x2 contingency table; column 0 is the population of interest
        parameter_scores = self.groups[self.population_of_interest]
        other_scores = self.groups[self.groups.keys()[self.groups.keys().index(self.population_of_interest)-1]]
        table = [[parameter_scores.count(self.possible_outcomes[0]), other_scores.count(self.possible_outcomes[0])],
                 [parameter_scores.count(self.possible_outcomes[1]), other_scores.count(self.possible_outcomes[1])]
                 ]
        row1_sum = sum(table[0])
        col1_sum = table[0][0]+table[1][0]
        
        # create the distribution to report probabilities
        self.distribution = stats.hypergeom(self.n, row1_sum, col1_sum)
        self.distribution.row1_sum = row1_sum
        self.distribution.col1_sum = col1_sum
        
        # add necessary attributes to the distribution
        self.distribution.__delta_values = [x/float(col1_sum) - (row1_sum-x)/float(self.n-col1_sum) for x in range(min(row1_sum, col1_sum)+1)]
        self.distribution.__delta_by_abs = [abs(x) for x in self.distribution.__delta_values]
        self.distribution.__delta_by_abs.sort(reverse=True)
        self.distribution.__delta_lookup = dict()
        for x in self.distribution.__delta_values:
            self.distribution.__delta_lookup.setdefault(abs(x), []).append(x)
        self.distribution.__delta_lookup
        self.distribution.__expected_value = 0.0
        def cvs1(self, alpha=0.05):
            """
            Get (LCV, UCV) for one-tailed tests
            """
            i = -1
            while self.cdf(i)+self.pmf(i+1) <= alpha:
                i += 1
            lcv = self.__delta_values[i] if i >= 0 else None
            i = len(self.__delta_values)
            while (1.0-self.cdf(i))+self.pmf(i)+self.pmf(i-1) <= alpha:
                i -= 1
            ucv = self.__delta_values[i] if i < len(self.__delta_values) else None
            return lcv, ucv
        self.distribution.cvs1 = new.instancemethod(cvs1, self.distribution, self.distribution.__class__)
        def cvs2(self, alpha=0.05):
            """
            Get (LCV, UCV) for two-tailed tests
            """
            error_scores = list()
            for x in self.__delta_by_abs:
                if sum([self.pmf(self.__delta_values.index(y)) for y in error_scores]) + sum([self.pmf(self.__delta_values.index(y)) for y in self.__delta_lookup[x]]) >= alpha: break
                error_scores += self.__delta_lookup[x]
            lcv, ucv = None, None
            try: lcv = max([x for x in error_scores if x < self.__expected_value])
            except: pass
            try: ucv = min([x for x in error_scores if x > self.__expected_value])
            except: pass
            return lcv, ucv
        self.distribution.cvs2 = new.instancemethod(cvs2, self.distribution, self.distribution.__class__)
        def p_lower(self, outcome):
            """
            Return probability of getting a value less than or equal to outcome
            """
            return sum(self.pmf(i) for i in range(len(self.__delta_values)) if self.__delta_values[i] <= outcome-self.__expected_value)
        self.distribution.p_lower = new.instancemethod(p_lower, self.distribution, self.distribution.__class__)
        def p_upper(self, outcome):
            """
            Return probability of getting a value greater than or equal to
            outcome
            """
            return sum(self.pmf(i) for i in range(len(self.__delta_values)) if self.__delta_values[i] >= outcome-self.__expected_value)
        self.distribution.p_upper = new.instancemethod(p_upper, self.distribution, self.distribution.__class__)
        def p_two(self, outcome):
            """
            Return probability of getting a value whose distance to the
            expected value is greater or equal to that of outcome
            """
            return sum(self.pmf(i) for i in range(len(self.__delta_values)) if abs(self.__delta_values[i]) >= abs(outcome-self.__expected_value))
        self.distribution.p_two = new.instancemethod(p_two, self.distribution, self.distribution.__class__)
    
    @classmethod
    def perform_unit_test(cls, **kwargs):
        """
        Perform base implementation, passing arguments for categories
        """
        super(FisherExactTest, cls).perform_unit_test(population_of_interest='Jury', **kwargs)

class MoodMedianTest(FisherExactTest):
    """
    Mood's median test
    """
    def __init__(self, *args, **kwargs):
        ExactTwoSampleTest.__init__(self, *args, **kwargs)
        self.validate_numericality()
        # find the grand median
        median = self.combined_scores[len(self.combined_scores)/2] if len(self.combined_scores)%2 == 1 else (self.combined_scores[len(self.combined_scores)/2]+self.combined_scores[len(self.combined_scores)/2-1])*0.5
        # create groups to perform Fisher exact test
        less_key = 'n < M'
        greater_key = u'n \u2265 M'
        other_population = self.groups.keys()[(self.groups.keys()).index(self.population_of_interest)-1]
        # first assign certain occurrences
        v1 = list()
        for x in self.groups[self.population_of_interest]:
            if x < median: v1.append(less_key)
            elif x > median: v1.append(greater_key)
            else: pass
        v2 = list()
        for x in self.groups[other_population]:
            if x < median: v2.append(less_key)
            elif x > median: v2.append(greater_key)
            else: pass
        # if there is an odd number of scores, it is possible for a value to fall on the median, so it should be assigned conservatively (e.g., less likely to reject)
        for i in range(len(self.groups[self.population_of_interest])-len(v1)):
            v1.append(less_key if v1.count(less_key) < v1.count(greater_key) else greater_key)
        for i in range(len(self.groups[other_population])-len(v2)):
            v2.append(less_key if v2.count(less_key) < v2.count(greater_key) else greater_key)
        # create new groups with dichotomized data
        new_groups = { self.population_of_interest:v1, other_population:v2 }
        # store an output file path as necessary
        self.create_output_file_path_if_needed()
        # perform Fisher's exact test
        FisherExactTest.__init__(
                                 self,
                                 population_of_interest=self.population_of_interest,
                                 groups=new_groups,
                                 hypothesized_value=self.hypothesized_value,
                                 alpha=self.alpha,
                                 output_file=self.output_file,
                                 is_silent=self.is_silent
                                 )

class TwoSampleWilcoxonTest(ExactTwoSampleTest):
    """
    A two-sample Wilcoxon rank test
    """
    def __init__(self, *args, **kwargs):
        ExactTwoSampleTest.__init__(self, *args, **kwargs)
        self.validate_numericality()
        # get the replacement scores associated with the smaller group
        self.replacement_values = self.__class__.get_ranks(self.combined_scores)
        self.smaller_group_replacement_values = [self.replacement_values[self.combined_scores.index(i)] for i in self.groups[self.smaller_group_key]]
        self.create_distribution()
        self.perform_test()
        # generate a probability table if requested
        self.create_output_file_path_if_needed()
        if len(self.output_file) > 0:
            headers = ['Tw', 'frequency', 'delta'] + self.get_probability_table_probability_headers()
            funcs = [lambda x: x, lambda x: self.distribution.frequency_of(x), lambda x: self.distribution.median()-x,] + self.get_probability_table_probability_functions()
            entries = list()
            for x in self.distribution.possible_scores():
                entries.append(dict((headers[i],funcs[i](x)) for i in range(len(headers))))
            self.save_probability_table(headers, entries)

class TwoSampleNormalScoresTest(ExactTwoSampleTest):
    """
    A two-sample Van der Waerden normal scores test
    """
    def __init__(self, *args, **kwargs):
        ExactTwoSampleTest.__init__(self, *args, **kwargs)
        self.validate_numericality()
        # get the replacement scores associated with the smaller group
        self.replacement_values = self.__class__.get_normal_deviates(self.combined_scores)
        self.smaller_group_replacement_values = [self.replacement_values[self.combined_scores.index(i)] for i in self.groups[self.smaller_group_key]]
        self.create_distribution()
        self.perform_test()
        # generate a probability table if requested
        self.create_output_file_path_if_needed()
        if len(self.output_file) > 0:
            headers = ['Tz', 'frequency', 'delta'] + self.get_probability_table_probability_headers()
            funcs = [lambda x: x, lambda x: self.distribution.frequency_of(x), lambda x: self.distribution.median()-x,] + self.get_probability_table_probability_functions()
            entries = list()
            for x in self.distribution.possible_scores():
                entries.append(dict((headers[i],funcs[i](x)) for i in range(len(headers))))
            self.save_probability_table(headers, entries)