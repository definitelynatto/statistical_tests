"""
This module contains exact matched-pair tests.
TODO: use hypothesized values, report confidence intervals
"""

import csv, math
from statistical_tests.exact import ExactTest

class ExactMatchedPairDistribution(object):
    """An exact distribution for a matched-pair test"""
    def __init__(self, replacement_values):
        # validate replacement_values
        if not isinstance(replacement_values, list) and not isinstance(replacement_values, tuple):
            raise TypeError('replacement_values must be a list or tuple')
        # create the exact sampling distribution of all possible combinations
        n = len(replacement_values)
        self.__sampling_distribution = list()
        for i in range(2**n):
            weights = bin(i)[2:].zfill(n)
            self.__sampling_distribution.append(sum([replacement_values[x]*int(weights[x]) for x in range(n)]))
        self.__sampling_distribution.sort()
        self.__individual_probability_mass = 1/float(len(self.__sampling_distribution))
        # retain an ordered list of possible scores in the sampling distribution
        self.__possible_scores = list(set(self.__sampling_distribution))
        self.__possible_scores.sort()
        # store frequency and probability mass tables for possible scores
        self.__score_frequencies = dict((x, self.__sampling_distribution.count(x)) for x in self.__possible_scores)
        self.__score_probabilities = dict((x, self.__score_frequencies[x]*self.__individual_probability_mass) for x in self.__possible_scores)
        # store the expected value of the distribution
        self.__expected_value = 0.5*sum(replacement_values)
        # retain an ordered list of delta values for possible scores
        self.__delta_values = [x-self.__expected_value for x in self.__possible_scores]
        # store the variance of the distribution
        self.__variance = 0.25*sum(x**2 for x in replacement_values)
    def possible_scores(self):
        """Return the list of possible scores in the distribution"""
        return self.__possible_scores
    def var(self):
        """Return the variance of this distribution"""
        return self.__variance
    def std(self):
        """Return the standard deviation of this distribution"""
        return math.sqrt(self.__variance)
    def median(self):
        """The expected value of the sampling distribution"""
        return self.__expected_value
    def mean(self):
        """The expected value of the sampling distribution (same as median since it is symmetrical)"""
        return self.median()
    def frequency_of(self, outcome):
        """Return the number of occurrences of outcome in the sampling distribution"""
        return self.__score_frequencies.setdefault(outcome, 0)
    def pmf(self, outcome):
        """Return the probability mass of a specified outcome"""
        return self.__score_probabilities.setdefault(outcome, 0)
    def cvs1(self, alpha=0.05):
        """Get (LCV, UCV) for one-tailed tests"""
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
        """Get (LCV, UCV) for two-tailed tests"""
        return self.cvs1(alpha*0.5)
    def p_lower(self, outcome):
        """Return probability of getting a value less than or equal to outcome"""
        '''
        delta = outcome - self.__expected_value
        limit = 0
        for i in range(len(self.__delta_values)):
            if self.__delta_values[i] > delta: break
            limit += 1
        return sum(self.pmf(self.__possible_scores[i]) for i in range(limit))
        '''
        return sum(self.pmf(self.__possible_scores[i]) for i in range(len(self.__possible_scores)) if self.__delta_values[i] <= outcome-self.__expected_value)
    def p_upper(self, outcome):
        """Return probability of getting a value greater than or equal to outcome"""
        '''
        delta = outcome - self.__expected_value
        limit = 0
        for i in range(len(self.__delta_values)-1,-1,-1):
            if self.__delta_values[i] < delta: break
            limit += 1
        return sum(self.pmf(self.__possible_scores[i]) for i in range(len(self.__possible_scores)-1, len(self.__possible_scores)-limit-1, -1))
        '''
        return sum(self.pmf(self.__possible_scores[i]) for i in range(len(self.__possible_scores)) if self.__delta_values[i] >= outcome-self.__expected_value)
    def p_two(self, outcome):
        """Return probability of getting a value whose distance to the expected value is greater or equal to that of outcome"""
        return sum(self.pmf(self.__possible_scores[i]) for i in range(len(self.__possible_scores)) if abs(self.__delta_values[i]) >= abs(outcome-self.__expected_value))

class ExactMatchedPairTest(ExactTest):
    """A base class for exact matched-pair tests"""
    def __init__(self,
                 pre_key='Pre', post_key='Post',    # keys for the two columns to compare in the csv file
                 pre=[], post=[],                   # optionally specify the pre and post scores manually
                 hypothesized_value=0,              # the value under test
                 *args,
                 **kwargs):
        ExactTest.__init__(self, *args, **kwargs)
        
        # validate input parameters
        if self.input_file:
            if not pre_key:
                raise ValueError('You must specify the name of the column containing pre-test scores.')
            if not post_key:
                raise ValueError('You must specify the name of the column containing post-test scores.')
            pre = list()
            post = list()
            try:
                for row in csv.DictReader(open(self.input_file)):
                    if not pre_key in row.keys(): raise ValueError('The pre-test key %s was not found.'%pre_key)
                    if not post_key in row.keys(): raise ValueError('The post-test key %s was not found.'%pre_key)
                    if row[pre_key] and row[post_key]:
                        pre.append(float(row[pre_key]))
                        post.append(float(row[post_key]))
                    elif (row[pre_key] and not row[post_key]) or (row[post_key] and not row[pre_key]):
                        sys.stderr.write('Row does not contain matched pairs (%s). Data is being skipped.\n'%row)
            except KeyError as e:
                raise KeyError('Unable to locate key %s in file %s.'%(e, self.input_file))
        if not pre or not post:
            raise ValueError('You must supply scores for both pre and post.')
        if (pre and not isinstance(pre, list) and not isinstance(pre, tuple)):
            raise TypeError('pre must be a list or tuple.')
        if (post and not isinstance(post, list) and not isinstance(post, tuple)):
            raise TypeError('post must be a list or tuple.')
        if pre and post and not (len(pre)==len(post)):
            raise ValueError('pre and post must be sequences of equal length.')
        
        # set the test parameter to median difference
        self.test_parameter = 'Md'
        # store difference scores as data attribute
        self.difference_scores = [post[i]-pre[i] for i in range(len(post))]
        # sort difference scores by absolute value
        self.difference_scores.sort(cmp=lambda x,y: cmp(abs(x), abs(y)))
        # store sample size
        self.n = len(self.difference_scores)
        # generate replacement values and weights
        self.replacement_values = self.difference_scores
        self.weights = [1.0 if x > 0.0 else (0.0 if x < 0.0 else 0.5) for x in self.difference_scores]
    
    def create_distribution(self):
        """Create an exact distribution from the replacement values"""
        self.distribution = ExactMatchedPairDistribution(self.replacement_values)
    
    def perform_test(self):
        """Perform a matched-pair test using all of the information stored in the data attributes"""
        # gather test results
        self.results[self.kTestStatKey] = sum(self.replacement_values[i]*self.weights[i] for i in range(self.n))
        self.results[self.kVarianceKey] = self.distribution.var()
        self.results[self.kStandardDeviationKey] = self.distribution.std()
        
        # get the p-values
        self.results[self.kPLowerKey] = self.distribution.p_lower(self.results[self.kTestStatKey])
        self.results[self.kPUpperKey] = self.distribution.p_upper(self.results[self.kTestStatKey])
        self.results[self.kP2Key] = self.distribution.p_two(self.results[self.kTestStatKey])
        
        # get the critical values
        self.results[self.kLCV1Key], self.results[self.kUCV1Key] = self.distribution.cvs1(self.alpha)
        self.results[self.kLCV2Key], self.results[self.kUCV2Key] = self.distribution.cvs2(self.alpha)
        
        # report actual type i error rates
        self.results[self.kTypeIErrorLKey] = self.distribution.p_lower(self.results[self.kLCV1Key]) if self.results[self.kLCV1Key] is not None else 0
        self.results[self.kTypeIErrorUKey] = self.distribution.p_upper(self.results[self.kUCV1Key]) if self.results[self.kUCV1Key] is not None else 0
        self.results[self.kTypeIError2Key] = (self.distribution.p_lower(self.results[self.kLCV2Key]) if self.results[self.kLCV2Key] is not None else 0) + (self.distribution.p_upper(self.results[self.kUCV2Key]) if self.results[self.kUCV2Key] is not None else 0)
        
        # print results if not in silent mode
        if not self.is_silent: self.print_test_results()
    
class SignTest(ExactMatchedPairTest):
    """A sign test, which properly retains 0s"""
    def __init__(self, *args, **kwargs):
        ExactMatchedPairTest.__init__(self, *args, **kwargs)
        self.replacement_values = [1]*len(self.difference_scores)
        self.create_distribution()
        self.perform_test()
        # generate a probability table if requested
        self.create_output_file_path_if_needed()
        if len(self.output_file) > 0:
            headers = ['x', 'n-x'] + self.get_probability_table_probability_headers()
            funcs = [lambda x: x, lambda x: self.n-x] + self.get_probability_table_probability_functions()
            entries = list()
            for x in range(self.n+1): entries.append(dict((headers[i],funcs[i](x)) for i in range(len(headers))))
            self.save_probability_table(headers, entries)

class MatchedPairWilcoxonTest(ExactMatchedPairTest):
    """A matched-pair Wilcoxon test, which properly retains 0s"""
    def __init__(self, *args, **kwargs):
        ExactMatchedPairTest.__init__(self, *args, **kwargs)
        self.replacement_values = self.__class__.get_ranks(self.difference_scores)
        self.create_distribution()
        self.perform_test()
        # generate a probability table if requested
        self.create_output_file_path_if_needed()
        if len(self.output_file) > 0:
            headers = ['T+', 'frequency'] + self.get_probability_table_probability_headers()
            funcs = [lambda x: x, lambda x: self.distribution.frequency_of(x)] + self.get_probability_table_probability_functions()
            entries = list()
            for x in self.distribution.possible_scores():
                entries.append(dict((headers[i],funcs[i](x)) for i in range(len(headers))))
            self.save_probability_table(headers, entries)

class MatchedPairNormalScoresTest(ExactMatchedPairTest):
    """A matched-pair Van der Waerden Normal Scores test, which properly retains 0s"""
    def __init__(self, *args, **kwargs):
        ExactMatchedPairTest.__init__(self, *args, **kwargs)
        self.replacement_values = self.__class__.get_positive_normal_deviates(self.difference_scores)
        self.create_distribution()
        self.perform_test()
        # generate a probability table if requested
        self.create_output_file_path_if_needed()
        if len(self.output_file) > 0:
            headers = ['Tz+', 'frequency'] + self.get_probability_table_probability_headers()
            funcs = [lambda x: x, lambda x: self.distribution.frequency_of(x)] + self.get_probability_table_probability_functions()
            entries = list()
            for x in self.distribution.possible_scores():
                entries.append(dict((headers[i],funcs[i](x)) for i in range(len(headers))))
            self.save_probability_table(headers, entries)