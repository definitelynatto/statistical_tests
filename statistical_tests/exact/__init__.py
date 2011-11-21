"""
This module's extensions contain a variety of nonparametric exact tests that,
contrary to popular fashion, do not eliminate data in order to increase power,
and consequently do not inflate Type I error rate. For information on the
tests, refer to e.g.,

Marascuilo, L. A., & Serlin, R. C. (1987). Statistical Methods for the Social
    and Behavioral Sciences. New York, New York: W.H. Freeman & Company.
    
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

import csv, os, re, sys
from scipy import stats
from statistical_tests import StatisticalTest

class ExactTest(StatisticalTest):
    ## common key for test results to designate actual 2-tailed type i error rate
    kTypeIError2Key = 'actual 2-tailed type i error rate'
    ## common key for test results to designate actual 2-tailed type i error rate
    kTypeIErrorLKey = 'actual lower tail type i error rate'
    ## common key for test results to designate actual 2-tailed type i error rate
    kTypeIErrorUKey = 'actual upper tail type i error rate'
    
    def __init__(self, *args, **kwargs):
        """Initialize a new ExactTest instance"""
        StatisticalTest.__init__(self, *args, **kwargs)
    
    ## a dict to store symbols to use for the output table if the supplied score is in the distribution's rejection region
    isInRejectionRegionSymbol = { True:'X', False:'' }
    
    def isInRejection2(self, x):
        """Return whether or not x is in the 2-tailed rejection region"""
        return (x>=self.results[self.kUCV2Key] if self.results.setdefault(self.kUCV2Key, None) is not None else False) or (x<=self.results[self.kLCV2Key] if self.results.setdefault(self.kLCV2Key, None) is not None else False)
    
    def isInRejectionLow(self, x):
        """Return whether or not x is in the lower-tail rejection region"""
        return x<=self.results[self.kLCV1Key] if self.results.setdefault(self.kLCV1Key, None) is not None else False
    
    def isInRejectionUpper(self, x):
        """Return whether or not x is in the upper tail rejection region"""
        return x>=self.results[self.kUCV1Key] if self.results.setdefault(self.kUCV1Key, None) is not None else False
    
    @classmethod
    def get_ranks(cls, scores):
        """Get a list of ranks and midranks for the supplied scores based on absolute value"""
        if not isinstance(scores, list) and not isinstance(scores, tuple):
            raise TypeError('Scores must be supplied as a list or tuple.')
        abs_scores = [abs(x) for x in scores]
        abs_scores.sort()
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
        """Get a list of positive normal deviates for the supplied scores based on absolute value"""
        if not isinstance(scores, list) and not isinstance(scores, tuple):
            raise TypeError('Scores must be supplied as a list or tuple.')
        # generate the list of positive normal deviates
        n = len(scores)
        normal_distribution = stats.norm()
        positive_normal_deviates = [normal_distribution.ppf((i+n+2)/(2.0*n+2.0)) for i in range(n)]
        # sort the scores by absolute value
        abs_scores = [abs(x) for x in scores]
        abs_scores.sort()
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
    def get_normal_deviates(cls, scores):
        """Get a list of normal deviates for the supplied scores based on absolute value"""
        if not isinstance(scores, list) and not isinstance(scores, tuple):
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
        """Create a path to an output file for a probability table"""
        if self.output_file == True and os.path.isfile(self.input_file):
            self.create_output_file_path_from_input_file_path(
                '%s probability table.csv'%(
                    re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ', self.__class__.__name__).lstrip().lower()
                )
            )
    
    def get_probability_table_probability_headers(self):
        """Get a list of headers for the probability and rejection region columns in the probability table."""
        return ['probability',
                'cumulative left tail probability',
                'cumulative right tail probability',
                'two-tailed rejection region (alpha=%s)'%self.alpha,
                'lower tail rejection region (alpha=%s)'%self.alpha,
                'upper tail rejection region (alpha=%s)'%self.alpha
                ]
    
    def get_probability_table_probability_functions(self):
        """Get a list of functions for the probability and rejection region columns in the probability table."""
        return [lambda x: self.distribution.pmf(x),
                lambda x: self.distribution.p_lower(x),
                lambda x: self.distribution.p_upper(x),
                lambda x: self.isInRejectionRegionSymbol[self.isInRejection2(x)],
                lambda x: self.isInRejectionRegionSymbol[self.isInRejectionLow(x)],
                lambda x: self.isInRejectionRegionSymbol[self.isInRejectionUpper(x)]
                ]
    
    def save_probability_table(self, headers, entries):
        """Save a probability table with the specified headers and dictionary entries to the specified location as a CSV."""
        # validate input parameters
        if not isinstance(headers, list) and not isinstance(headers, tuple):
            raise TypeError("Headers must be specified as a list or tuple.")
        if not isinstance(entries, list) and not isinstance(entries, tuple):
            raise TypeError("Entries must be specified as a list or tuple of dictionaries.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("Each entry must be a dictionary whose keys are specified by the headers.")
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
    
    def print_test_results(self):
        """Nicely format the results of a test and print to the console"""
        decision = { True:u'Reject H\u2080', False:u'Fail to Reject H\u2080' }
        x = self.results[self.kTestStatKey]
        summary = u'''
        Test Results:
        --------------------------------
        Test Statistic: %s
        \u03C3\u00b2:             %s
        \u03C3:              %s
        '''%(
             self.results[self.kTestStatKey],
             self.results[self.kVarianceKey],
             self.results[self.kStandardDeviationKey])
        two_tailed = ''
        try: two_tailed = u'''
        Two-Tailed Test of %s = %s
        --------------------------------
        p:              %s
        Upper CV:       %s
        Lower CV:       %s
        Actual \u03b1:       %s
        Decision:       %s
        '''%(
             self.test_parameter,
             self.hypothesized_value,
             self.results[self.kP2Key],
             self.results[self.kUCV2Key],
             self.results[self.kLCV2Key],
             self.results[self.kTypeIError2Key],
             decision[(x>=self.results[self.kUCV2Key] if self.results[self.kUCV2Key] is not None else False) or (x<=self.results[self.kLCV2Key] if self.results[self.kLCV2Key] is not None else False)])
        except: raise
        lower_tailed = ''
        try: lower_tailed = u'''
        Lower Tail Test of %s \u2265 %s
        --------------------------------
        p:              %s
        Lower CV:       %s
        Actual \u03b1:       %s
        Decision:       %s
        '''%(
             self.test_parameter,
             self.hypothesized_value,
             self.results[self.kPLowerKey],
             self.results[self.kLCV1Key],
             self.results[self.kTypeIErrorLKey],
             decision[x<=self.results[self.kLCV1Key] if self.results[self.kLCV1Key] is not None else False])
        except: pass
        upper_tailed = ''
        try: upper_tailed = u'''
        Upper Tail Test of %s \u2264 %s
        --------------------------------
        p:              %s
        Upper CV:       %s
        Actual \u03b1:       %s
        Decision:       %s
        '''%(
             self.test_parameter,
             self.hypothesized_value,
             self.results[self.kPUpperKey],
             self.results[self.kUCV1Key],
             self.results[self.kTypeIErrorUKey],
             decision[x>=self.results[self.kUCV1Key] if self.results[self.kUCV1Key] is not None else False])
        except: pass
        print summary + two_tailed + lower_tailed + upper_tailed