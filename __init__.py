"""
A module and a set of extensions that contain classes and utilities
for performing statistical tests
"""

import codecs, os
from itertools import chain
from scipy import stats

class StatisticalTestResults(object):
    """
    A class for storing the results of a statistical test
    """
    ## the a priori type i error rate
    alpha = None
    ## the test statistic
    test_statistic = None
    ## variance
    var = None
    ## standard deviation
    std = None
    ## hypothesized value being tested
    test_value = None
    ## two-tailed p-value
    p_two = None
    ## one-tailed lower p-value
    p_lower = None
    ## one-tailed upper p-value
    p_upper = None
    ## two-tailed upper critical value
    ucv2 = None
    ## two-tailed lower critical value
    lcv2 = None
    ## one-tailed upper critical value
    ucv1 = None
    ## one-tailed lower critical value
    lcv1 = None
    ## two-tailed confidence interval
    ci_two = (None, None)
    ## lower-tail confidence interval upper limit
    ci_lower = None
    ## upper-tail confidence interval lower limit
    ci_upper = None
    ## effect size
    effect_size = None
    ## partial effect size
    partial_effect_size = None
    
class StatisticalTest(object):
    """
    A base class for statistical tests.
    """
    def __init__(self,
                 input_file='',         # path to a csv file with input data
                 output_file='',        # specify path for an output file if the test supports it; optionally just specify True if using input_file
                 alpha=0.05,            # the desired type i error rate
                 hypothesized_value=0,  # the value being tested
                 test_parameter='',     # the parameter under test
                 is_silent=False,       # should the test print results to the console?
                 *args,
                 **kwargs
                 ):
        """
        Initialize a new StatisticalTest instance
        """
        self.input_file = input_file
        if self.input_file and not os.path.isfile(input_file):
            raise IOError('Unable to load specified file: %s'%input_file)
        self.output_file = output_file
        self.alpha = alpha
        self.is_silent = is_silent
        self.test_parameter = test_parameter
        self.hypothesized_value = hypothesized_value
        # an object to store the results of the test
        self.results = StatisticalTestResults()
    
    @classmethod
    def is_array_like(cls, data):
        """
        Returns true if data is a tuple or list; else false
        """
        return any((isinstance(data, list), isinstance(data, tuple)))
    
    @classmethod
    def is_numeric(cls, x):
        """
        Returns true if x is an int or float; else false
        """
        return any((isinstance(x, float), isinstance(x, int)))
    
    @classmethod
    def is_numeric_array_like(cls, data):
        """
        Returns true if data is a tuple or list of numeric values; else false
        """
        return cls.is_array_like(data) and all((cls.is_numeric(x) for x in data))
    
    @classmethod
    def is_dict_array_like(cls, data):
        """
        Returns true i data is a tuple or list of dicts; else false
        """
        return cls.is_array_like(data) and all((isinstance(x, dict) for x in data))
    
    @classmethod
    def create_output_file_name_from_class_name(cls):
        """
        Return a name for an output file base on the class name
        """
        return re.sub(
                      '((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))',
                      ' ',
                      cls.__name__
                      ).lstrip().lower()
    
    def perform_test(self):
        """
        Perform the test. Override in subclass.
        """
        pass
    
    def create_output_file_path_from_input_file_path(self, output_file_name):
        """
        Create a path to an output file with the specified name that is
        parallel to that of input_file
        """
        if not self.input_file: raise ValueError('No input file specified.')
        self.output_file = os.path.join(os.path.split(self.input_file)[0], output_file_name)
    
    def printable_test_results(self):
        """
        Return printable test results for console output. Override in subclass.
        """
        pass
    
    @classmethod
    def combine_samples(cls, *args):
        """
        Combine several lists/tuples into a single list
        """
        try: return list(chain.from_iterable([a for a in args]))
        except: raise TypeError('Expected only lists or tuples')
    
    @classmethod
    def get_grand_mean(cls, *args):
        """
        Get the grand mean of several lists/tuples of data
        """
        return stats.tmean(cls.combine_samples(*args))
    
    @classmethod
    def get_pooled_standard_error(cls, *args):
        """
        Get the pooled standard error of the groups
        """
        try: return sum(len(a)*stats.tvar(a) for a in args)/float(sum(len(a)-1 for a in args))
        except: raise TypeError('Expected only lists or tuples')
    
    @classmethod
    def ss_between(cls, grand_mean=None, *args, **kwargs):
        """
        Get the sum of squared deviations of each group's mean compared to the
        grand mean of all groups
        """
        if grand_mean is None: grand_mean = cls.get_grand_mean(*args)
        return sum([len(a)*(grand_mean-stats.tmean(a))**2 for a in args])
    
    @classmethod
    def ss_total(cls, grand_mean=None, combined_sample=[], *args, **kwargs):
        """
        Get the sum of squared deviations of each value compared to the grand
        mean
        """
        if grand_mean is None: grand_mean = cls.get_grand_mean(*args)
        if not combined_sample: combined_sample = cls.combine_samples(*args)
        return sum([(x-grand_mean)**2 for x in combined_sample])
    
    @classmethod
    def ss_within(cls, *args):
        """
        Get the sum of square deviations of each value compared to its group
        mean value
        """
        try: return sum((len(a)-1)*stats.tvar(a) for a in args)
        except: raise TypeError('Expected only lists or tuples')
    
    @classmethod
    def perform_unit_test(cls, **kwargs):
        """
        Perform the unit test for the class.
        """
        kwargs.setdefault('is_silent', True)
        kwargs.setdefault('input_file', cls.get_unit_test_results_file_name())
        st = cls(**kwargs)
        expected = None
        with codecs.open(cls.get_unit_test_results_file_name(suffix=kwargs.setdefault('suffix', '')), encoding='utf-8') as f:
            expected = f.read()
        kwargs.setdefault('test_name', '')
        print '%s %s Printed results: %s'%(cls.__name__,
                                           kwargs['test_name'],
                                           ('PASS' if expected == st.printable_test_results()
                                            else 'FAIL'
                                            )
                                           )
        # return the test instance in case this is called from a subclass that needs to do more with it
        return st
    
    @classmethod
    def get_unit_test_dir(cls):
        """
        Return the location of the unit test data directory
        """
        return os.path.join(os.path.split(__file__)[0], 'unit test data')
    
    @classmethod
    def get_unit_test_results_file_name(cls, suffix=''):
        """
        Get the name for a file containing the expected unit test results
        """
        return os.path.join(
                            cls.get_unit_test_dir(),
                            'expected results - %s%s'%(
                                                       cls.create_output_file_name_from_class_name(),
                                                       suffix
                                                       )
                            )
    
    @classmethod
    def get_unit_test_input_file_name(cls):
        """
        Get the name for a file containing input unit test data
        """
        return os.path.join(
                            cls.get_unit_test_dir(),
                            'test - %s.csv'%cls.create_output_file_name_from_class_name()
                            )

class MultipleComparisons(object):
    """
    A class containing assorted multiple comparison procedures
    """
    @classmethod
    def __validate_alpha(cls, alpha):
        """
        Validate alpha value
        """
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a decimal number')
        if alpha > 1 or alpha < 0:
            raise ValueError('alpha must be in the range [0, 1]')
    
    @classmethod
    def __validate_args(cls, alpha, p_values):
        """
        Validate input parameters
        """
        cls.__validate_alpha(alpha)
        if not StatisticalTest.is_numeric_array_like(p_values):
            raise TypeError('p-values must be supplied as a list or tuple')
    
    @classmethod
    def bonferroni_dunn(cls, alpha, p_values):
        """
        Perform the so-called Bonferonni-Dunn procedure (which should actually
        be called the Fisher-Bool procedure). Return True if all p-values are
        significant; else False
        """
        cls.__validate_args(alpha, p_values)
        pc = alpha/float(len(p_values))
        return all((p <= pc for p in p_values))
    
    @classmethod
    def sidak_bonferroni(cls, alpha, p_values):
        """
        Perform the Sidak-Bonferroni procedure, which is marginally more
        powerful than the Bonferonni-Dunn. Return True if all p-values are
        significant; else False
        """
        cls.__validate_args(alpha, p_values)
        pc = 1-(1-alpha)**(1/float(len(p_values)))
        return all((p <= pc for p in p_values))
    
    @classmethod
    def holm_test(cls, alpha, p_values):
        """
        Perform the Holm sequentially rejective procedure using the supplied
        alpha and p-values. Return True if all p-values are significant; else
        False
        """
        cls.__validate_args(alpha, p_values)
        p_values = sorted(p_values)
        return all((p_values[i] <= alpha/float(len(p_values)-i) for i in range(len(p_values))))
    
    @classmethod
    def scheffe_cv(cls, alpha, num_levels, num_subjects):
        """
        Obtain a Scheffe critical value using the specified alpha, number of
        levels for the factor, and number of subjects in the entire sample.
        """
        cls.__validate_alpha(alpha)
        if not isinstance(num_levels, int):
            raise TypeError('num_levels must be an integer describing the number of levels for the factor in question')
        if not num_levels > 1:
            raise ValueError('There must be more than one level for the factor')
        if not num_subjects > 0:
            raise ValueError('There must be at least 1 (ideally many more) subjects')
        return (num_levels-1)*stats.f.ppf(1.0-alpha)

import exact
import anova

def __unit_test_all():
    """
    Automated unit test for all statistical tests in the package
    """
    exact.matched_pair.MatchedPairNormalScoresTest.perform_unit_test()
    exact.matched_pair.MatchedPairWilcoxonTest.perform_unit_test()
    exact.matched_pair.SignTest.perform_unit_test()
    exact.two_sample.FisherExactTest.perform_unit_test()
    exact.two_sample.MoodMedianTest.perform_unit_test()
    exact.two_sample.TwoSampleNormalScoresTest.perform_unit_test()
    exact.two_sample.TwoSampleWilcoxonTest.perform_unit_test()