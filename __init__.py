import os
from itertools import chain
from scipy import stats

class StatisticalTest(object):
    """A base class for statistical tests."""
    ## common key for test results to designate the hypothesized value 
    kTestValue = 'test value'
    
    ## common key for test results to designate the test statistic
    kTestStatKey = 'test statistic'
    ## common key for test results to designate the variance
    kVarianceKey = 'variance'
    ## common key for test results to designate the standard deviation
    kStandardDeviationKey = 'standard deviation'
    
    ## common key for test results to designate the 2-tailed p-value
    kP2Key = '2-tailed p-value'
    ## common key for test results to designate the lower tail p-value
    kPLowerKey = 'lower tail p-value'
    ## common key for test results to designate the upper tail p-value
    kPUpperKey = 'upper tail p-value'
    
    ## common key for test results to designate the 2-tailed ucv
    kUCV2Key = '2-tailed upper critical value'
    ## common key for test results to designate the 2-tailed lcv
    kLCV2Key = '2-tailed lower critical value'
    ## common key for test results to designate the 1-tailed ucv
    kUCV1Key = '1-tailed upper critical value'
    ## common key for test results to designate the 1-tailed lcv
    kLCV1Key = '1-tailed lower critical value'
    
    ## common key for test results to designate the lower limit of the confidence interval for a 2-tailed test
    kCI2LowerKey = '2-tailed confidence interval lower limit'
    ## common key for test results to designate the upper limit of the confidence interval for a 2-tailed test
    kCI2UpperKey = '2-tailed confidence interval upper limit'
    ## common key for test results to designate the lower limit of the confidence interval for a 1-tailed test
    kCI1LowerKey = '1-tailed confidence interval lower limit'
    ## common key for test results to designate the upper limit of the confidence interval for a 1-tailed test
    kCI1UpperKey = '1-tailed confidence interval upper limit'
    
    ## common key for test results to designate the effect size
    kEffectSizeKey = 'effect size'
    ## common key for test results to designate the partial effect size
    kPartialEffectSizeKey = 'partial effect size'
    
    def __init__(self,
                 input_file='',         # path to a csv file with input data
                 output_file='',        # specify path for an output file if the test supports it; optionally just specify True if using input_file
                 alpha=0.05,            # the desired type i error rate
                 hypothesized_value=0,  # the value being tested
                 test_parameter='',     # the parameter under test
                 is_silent=False,       # should the test print results to the console?):
                 *args,
                 **kwargs):
        """Initialize a new StatisticalTest instance"""
        self.input_file = input_file
        if self.input_file and not os.path.isfile(input_file):
            raise IOError('Unable to load specified file: %s'%input_file)
        self.output_file = output_file
        self.alpha = alpha
        self.is_silent = is_silent
        self.test_parameter = test_parameter
        self.hypothesized_value = hypothesized_value
        # a dictionary to store the results of the test
        self.results = dict()
        
    def perform_test(self):
        """Override in subclass"""
        pass
    
    def create_output_file_path_from_input_file_path(self, output_file_name):
        """Create a path to an output file with the specified name that is parallel to that of input_file"""
        if not self.input_file: raise ValueError('No input file specified.')
        self.output_file = os.path.join(os.path.split(self.input_file)[0], output_file_name)
    
    def print_test_results(self):
        """Override in subclass"""
        pass
    
    @classmethod
    def combine_samples(cls, *args):
        """Combine several lists/tuples into a single list"""
        try: return list(chain.from_iterable([a for a in args]))
        except: raise TypeError('Expected only lists or tuples')
    
    @classmethod
    def get_grand_mean(cls, *args):
        """Get the grand mean of several lists/tuples of data"""
        return stats.tmean(cls.combine_samples(*args))
    
    @classmethod
    def get_pooled_standard_error(cls, *args):
        """Get the pooled standard error of the groups"""
        try: return sum(len(a)*stats.tvar(a) for a in args)/float(sum(len(a)-1 for a in args))
        except: raise TypeError('Expected only lists or tuples')
    
    @classmethod
    def ss_between(cls, grand_mean=None, *args, **kwargs):
        """Get the sum of squared deviations of each group's mean compared to the grand mean of all groups"""
        if grand_mean is None: grand_mean = cls.get_grand_mean(*args)
        return sum([len(a)*(grand_mean-stats.tmean(a))**2 for a in args])
    
    @classmethod
    def ss_total(cls, grand_mean=None, combined_sample=[], *args, **kwargs):
        """Get the sum of squared deviations of each value compared to the grand mean"""
        if grand_mean is None: grand_mean = cls.get_grand_mean(*args)
        if not combined_sample: combined_sample = cls.combine_samples(*args)
        return sum([(x-grand_mean)**2 for x in combined_sample])
    
    @classmethod
    def ss_within(cls, *args):
        """Get the sum of square deviations of each value compared to its group mean value"""
        try: return sum((len(a)-1)*stats.tvar(a) for a in args)
        except: raise TypeError('Expected only lists or tuples')