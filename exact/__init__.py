"""
This module's extensions contain a variety of nonparametric exact tests that,
contrary to popular fashion, do not eliminate data in order to increase power,
and consequently do not inflate Type I error rate. For information on the
tests, refer to e.g.,

Marascuilo, L.A., & Serlin, R.C. (1987). Statistical Methods for the Social and
    Behavioral Sciences. New York, NY: W.H. Freeman & Company.
Marascuilo, L.A., & McSweeney, M. (1977). Nonparametric and Distribution-Free
    Methods for the Social Sciences. Monterey, California: Brooks/Cole
    Publishing Co.
    
When selecting a test, consider the following table of relative efficiencies
for different types of populations:
                                          DISTRIBUTION
Test     Compared to  Normal   Uniform  Logistic Dbl Exp  Exponent Cauchy
-------- ------------ -------- -------- -------- -------- -------- --------
Sign     t            0.637    0.333    0.822    2.000    0.541    inf.
         Norm Sc      0.637    0.000    0.785    1.570    0.000    1.883
         Wilcoxon     0.667    0.333    0.750    1.333    0.180    1.333
-------- ------------ -------- -------- -------- -------- -------- --------
Wilcoxon t            0.955    1.000    1.097    1.500    3.000    inf.
         Norm Sc      0.955    0.000    1.047    1.178    0.000    1.412
         Sign         1.570    3.000    1.217    0.500    5.556    0.750
-------- ------------ -------- -------- -------- -------- -------- --------
Norm Sc  t            1.000    inf.     1.047    1.274    inf.     inf.
         Wilcoxon     1.047    inf.     0.955    0.849    inf.     0.708
         Sign         1.571    inf.     1.274    0.637    inf.     0.531
-------- ------------ -------- -------- -------- -------- -------- --------
t        Sign         1.570    3.000    1.217    0.500    1.848    0.000
         Wilcoxon     1.047    1.000    0.912    0.667    0.333    0.000
         Norm Sc      1.000    0.000    0.955    0.785    0.000    0.000
"""

import codecs, csv, os, re, sys
from scipy import stats
from statistical_tests import *

class ExactTestResults(StatisticalTestResults):
    """
    A class to store the results of an exact test
    """
    ## actual exact two-tailed alpha
    exact_alpha_two = 0.0
    ## actual exact lower tail alpha
    exact_alpha_lower = 0.0
    ## actual exact upper tail alpha
    exact_alpha_upper = 0.0

class ExactTest(StatisticalTest):
    def __init__(self, *args, **kwargs):
        """
        Initialize a new ExactTest instance
        """
        StatisticalTest.__init__(self, *args, **kwargs)
        self.results = ExactTestResults()
    
    ## a dict to store symbols to use for the output table if the supplied score is in the distribution's rejection region
    rejection_region_symbols = { True:'X', False:'' }
    
    def is_in_rejection_two(self, x):
        """
        Return whether or not x is in the 2-tailed rejection region
        """
        return (x>=self.results.ucv2 if self.results.ucv2 is not None else False) or (x<=self.results.lcv2 if self.results.lcv2 is not None else False)
    
    def is_in_rejection_low(self, x):
        """
        Return whether or not x is in the lower-tail rejection region
        """
        return x<=self.results.lcv1 if self.results.lcv1 is not None else False
    
    def is_in_rejection_upper(self, x):
        """
        Return whether or not x is in the upper tail rejection region
        """
        return x>=self.results.ucv1 if self.results.ucv1 is not None else False
    
    @classmethod
    def get_ranks_by_abs(cls, scores):
        """
        Get a list of ranks and midranks for the supplied scores based on
        absolute value
        """
        if not cls.is_array_like(scores):
            raise TypeError('Scores must be supplied as a list or tuple.')
        abs_scores = sorted([abs(x) for x in scores])
        reverse_abs_scores = abs_scores[::-1]
        replacement_values = list()
        n = len(scores)
        for i in range(n):
            start = abs_scores.index(abs_scores[i])
            end = n-reverse_abs_scores.index(abs_scores[i])
            replacement_values.append(sum(range(start+1,end+1))/float(end-start))
        return replacement_values
    
    @classmethod
    def get_positive_normal_deviates(cls, scores):
        """
        Get a list of positive normal deviates for the supplied scores based on
        absolute value
        """
        if not cls.is_array_like(scores):
            raise TypeError('Scores must be supplied as a list or tuple.')
        # generate the list of positive normal deviates
        n = len(scores)
        normal_distribution = stats.norm()
        positive_normal_deviates = [normal_distribution.ppf((i+n+2)/(2.0*n+2.0)) for i in range(n)]
        # sort the scores by absolute value
        abs_scores = sorted([abs(x) for x in scores])
        reverse_abs_scores = abs_scores[::-1]
        # get the positive normal deviates associated with the ranks
        replacement_values = [positive_normal_deviates[x-1] for x in range(1, n+1)]
        # average replacement value of ties
        keys = set(abs_scores)
        for k in keys:
            start = abs_scores.index(k)
            end = n-reverse_abs_scores.index(k)
            avg_score = sum([x for x in replacement_values[start:end]])/float(end-start)
            for i in range(start, end): replacement_values[i] = avg_score
        return replacement_values
    
    @classmethod
    def get_ranks(cls, scores):
        """
        Get a list of ranks and midanks for the supplied scores
        """
        if not cls.is_array_like(scores):
            raise TypeError('Scores must be supplied as a list or tuple.')
        sorted_scores = sorted(scores)
        reverse_scores = sorted_scores[::-1]
        replacement_values = list()
        n = len(scores)
        for i in range(n):
            start = sorted_scores.index(sorted_scores[i])
            end = n-reverse_scores.index(sorted_scores[i])
            replacement_values.append(sum(range(start+1,end+1))/float(end-start))
        return replacement_values
    
    @classmethod
    def get_normal_deviates(cls, scores):
        """
        Get a list of normal deviates for the supplied scores based on absolute
        value
        """
        if not cls.is_array_like(scores):
            raise TypeError('Scores must be supplied as a list or tuple.')
        # generate the list of positive normal deviates
        n = len(scores)
        normal_distribution = stats.norm()
        normal_deviates = [normal_distribution.ppf((i+1)/float(n+1)) for i in range(n)]
        # sort the scores
        sorted_scores = scores
        sorted_scores.sort()
        reverse_scores = sorted_scores[::-1]
        # average ties
        replacement_values = list()
        for i in range(n):
            start = sorted_scores.index(sorted_scores[i])
            end = n-reverse_scores.index(sorted_scores[i])
            replacement_values.append(sum(normal_deviates[start:end])/float(end-start))
        return replacement_values
    
    def create_output_file_path_if_needed(self):
        """
        Create a path to an output file for a probability table
        """
        if self.output_file == True and os.path.isfile(self.input_file):
            self.create_output_file_path_from_input_file_path(
                '%s probability table.csv'%
                self.__class__.create_output_file_name_from_class_name()
            )
    
    def get_probability_table_probability_headers(self):
        """
        Get a list of headers for the probability and rejection region columns
        in the probability table.
        """
        return ['probability',
                'cumulative left tail probability',
                'cumulative right tail probability',
                'two-tailed rejection region (alpha=%s)'%self.alpha,
                'lower tail rejection region (alpha=%s)'%self.alpha,
                'upper tail rejection region (alpha=%s)'%self.alpha
                ]
    
    def get_probability_table_probability_functions(self):
        """
        Get a list of functions for the probability and rejection region
        columns in the probability table.
        """
        return [lambda x: self.distribution.pmf(x),
                lambda x: self.distribution.p_lower(x),
                lambda x: self.distribution.p_upper(x),
                lambda x: self.rejection_region_symbols[self.is_in_rejection_two(x)],
                lambda x: self.rejection_region_symbols[self.is_in_rejection_low(x)],
                lambda x: self.rejection_region_symbols[self.is_in_rejection_upper(x)]
                ]
    
    def save_probability_table(self, headers, entries):
        """
        Save a probability table with the specified headers and dictionary
        entries to the specified location as a CSV.
        """
        # validate input parameters
        if not self.__class__.is_array_like(headers):
            raise TypeError('Headers must be specified as a list or tuple.')
        if not self.__class__.is_dict_array_like(entries):
            raise TypeError('Entries must be specified as a list or tuple of dictionaries.')
        # create a new csv writer
        try: writer = csv.DictWriter(open(self.output_file, 'w'), headers)
        except: raise
        # create a line of headers using the first entry
        writer.writerow(dict((k,k) for k in headers))
        # write each line
        try:
            for entry in entries: writer.writerow(entry)
            # status message
            sys.stdout.write('Saved probability table: %s\n'%self.output_file)
        except: raise
    
    def printable_test_results(self):
        """
        Nicely format the results of a test for console output
        """
        decision = { True:u'Reject H\u2080', False:u'Fail to Reject H\u2080' }
        x = self.results.test_statistic
        summary =           u'Test Results:\n' + \
                            u'--------------------------------\n' + \
                            u'Test Statistic: %s\n'%x + \
                            u'\u03C3\u00b2:             %s\n'%self.results.var + \
                            u'\u03C3:              %s\n'%self.results.std
        two_tailed = ''
        try:
            two_tailed =    u'Two-Tailed Test of %s = %s\n'%(self.test_parameter, self.hypothesized_value) + \
                            u'--------------------------------\n' + \
                            u'p:              %s\n'%self.results.p_two + \
                            u'Upper CV:       %s\n'%self.results.ucv2 + \
                            u'Lower CV:       %s\n'%self.results.lcv2 + \
                            u'Actual \u03b1:       %s\n'%self.results.exact_alpha_two + \
                            u'Decision:       %s\n'%decision[
                                                             (x>=self.results.ucv2 if self.results.ucv2 is not None else False) or 
                                                             (x<=self.results.lcv2 if self.results.lcv2 is not None else False)
                                                             ]
        except: pass
        lower_tailed = ''
        try:
            lower_tailed =  u'Lower Tail Test of %s \u2265 %s\n'%(self.test_parameter, self.hypothesized_value) + \
                            u'--------------------------------\n' + \
                            u'p:              %s\n'%self.results.p_lower + \
                            u'Lower CV:       %s\n'%self.results.lcv1 + \
                            u'Actual \u03b1:       %s\n'%self.results.exact_alpha_lower + \
                            u'Decision:       %s\n'%decision[x<=self.results.lcv1 if self.results.lcv1 is not None else False]
        except: pass
        upper_tailed = ''
        try:
            upper_tailed =  u'Upper Tail Test of %s \u2264 %s\n'%(self.test_parameter, self.hypothesized_value) + \
                            u'--------------------------------\n' + \
                            u'p:              %s\n'%self.results.p_upper + \
                            u'Upper CV:       %s\n'%self.results.ucv1 + \
                            u'Actual \u03b1:       %s\n'%self.results.exact_alpha_upper + \
                            u'Decision:       %s\n'%decision[x>=self.results.ucv1 if self.results.ucv1 is not None else False]
        except: pass
        return ('\n'.join([summary, two_tailed, lower_tailed, upper_tailed])).rstrip()
    
    @classmethod
    def perform_unit_test(cls, auto_delete=True, **kwargs):
        """
        Perform the base implementation and then confirm probability table.
        Data comes from Marascuilo & Serlin (1987) and Marascuilo & McSweeney
        (1977).
        """
        test_probability_table = os.path.join(
                                              os.getenv('HOME'),
                                              'unit test - %s.csv'%cls.create_output_file_name_from_class_name()
                                              )
        kwargs['output_file'] = test_probability_table
        st = super(ExactTest, cls).perform_unit_test(**kwargs)
        expected = None
        with codecs.open(cls.get_unit_test_probability_file_name(), encoding='utf-8') as f:
            expected = f.read()
        test_output = None
        with codecs.open(test_probability_table, encoding='utf-8') as f:
            test_output = f.read()
        print '%s Probability table: %s'%(cls.__name__,
                                          ('PASS' if expected == test_output
                                           else 'FAIL')
                                          )
        if auto_delete: os.remove(test_probability_table)
        # return the test instance in case this is called from a subclass that needs to do more with it
        return st
    
    @classmethod
    def get_unit_test_probability_file_name(cls):
        """
        Get the file name for expected probability table results for the unit
        test
        """
        return os.path.join(
                            cls.get_unit_test_dir(),
                            'expected output - %s probabilities.csv'%
                            cls.create_output_file_name_from_class_name()
                            )