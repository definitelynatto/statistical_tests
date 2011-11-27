"""
This module contains a variety of tests in the ANOVA family

TODO: support testing without the requirement of a csv file
"""

import copy, csv, itertools, os
from scipy import stats
from numpy.testing import assert_approx_equal
from statistical_tests import *

class ANOVAConstrastSource(object):
    """
    A class to describe a source in an ANOVA contrast
    """
    def __init__(self, source_factor, source_level, coefficient):
        """
        Initialze a new instance of a constrast source
        """
        if not all([
                    isinstance(source_factor, str),
                    isinstance(source_level, str)
                    ]):
            raise Exception('source_factor and source_level must be strings')
        if not StatisticalTest.is_numeric(coefficient):
            raise Exception('Contrast coefficient must be a number')
        self.source_factor = source_factor
        self.source_level = source_level
        self.coefficient = coefficient
    
class ANOVAConstrast(object):
    """
    A class for storing a constrast definition
    """
    def __init__(self, *args):
        """
        Initialize a new instance of a contrast definition
        """
        # TODO: ensure there are no overlapping sources
        if not all([isinstance(x, ANOVAConstrastSource) for x in args]):
            raise Exception('All arguments must be of type ANOVAContrastSource')
        self.sources = args
        if not self.is_valid():
            raise Exception('All contrast coefficients must sum to 0')
    
    def is_valid(self):
        """
        Return True if coefficients sum to 0; else False
        """
        try:
            assert_approx_equal(
                                sum(source.coefficient for source in self.sources),
                                0
                                )
            return True
        except: return False
    
    def is_orthogonal_to(self, other):
        """
        Return True if orthogonal to other; else False
        """
        # TODO: implement it
        return False
    
    @classmethod
    def are_orthogonal(cls, c1, c2):
        """
        Return True if c1 and c2 are orthogonal; else False
        """
        return c1.is_orthogonal_to(c2)

class ANOVAResults(StatisticalTestResults):
    """
    A class for storing the results of an analysis of variance
    """
    ## the name for the variability source
    source_name = None
    ## the marginal mean associated with the source
    mean = None
    ## the sum of squares for the source
    ss = None
    ## the degrees of freedom for the source
    df = None
    ## the mean squared for the source
    ms = None

class ANOVA(StatisticalTest):
    """
    An analysis of variance on any number of terms, assuming a balanced design.
    """
    '''
    ## key for storing the grand mean in the test results
    kGrandMeanKey = 'grand mean'
    ## key for storing the degrees of freedom in the test results
    kDFKey = 'df'
    ## key for storing the sum of squares in the test results
    kSSKey = 'ss'
    ## key for storing the mean square in the test results
    kMSKey = 'ms'
    ## key for storing the source of data in the test results
    kSourceName = 'source name'
    ## key for storing main effect test results
    kMainEffectsKey = 'main effects'
    ## key for storing contrast results
    kContrastsKey = 'contrasts'
    ## key for storing names of contrast factors
    kContrastFactorNames = 'contrast factors'
    '''
    def __init__(
                 self,
                 factor_keys=['Group'], # keys for columns specifying factors of interest
                 outcome_key='Outcome', # key for column specifying the outcome variable
                 contrasts=(),          # an array_like of ANOVAContrast definitions
                 *args,
                 **kwargs
                 ):
        StatisticalTest.__init__(self, *args, **kwargs)
        self.results = list() # a list of ANOVAResults
        if not self.input_file and not os.path.isfile(self.input_file):
            raise ValueError('You must specify an input csv file')
        if not isinstance(factor_keys, list) and not isinstance(factor_keys, tuple):
            raise TypeError('factor_keys must be a list or tuple of column headers')
        if not outcome_key:
            raise ValueError('You must specify a key for the outcome variable')
        # store necessary arguments as data attributes
        self.outcome_key = str(outcome_key)
        self.factor_keys = list(factor_keys)
        # store levels for each factor in a dictionary of factor : [levels] pairs
        self.factor_levels = dict((f, list(set(entry[f] for entry in csv.DictReader(open(self.input_file)) if entry[f]))) for f in self.factor_keys)
        # sort the levels if there is a natural order
        for f in self.factor_levels.keys():
            self.factor_levels[f].sort(cmp=lambda x,y: cmp(str(x).lower(), str(y).lower()))
        self.data = list()
        row_num = 1
        for row in csv.DictReader(open(self.input_file)):
            if not row[self.outcome_key]: raise ValueError('Missing outcome data in row %i. Design must be balanced.'%row_num)
            try: row[self.outcome_key] = float(row[self.outcome_key])
            except: raise ValueError('Error converting outcome value to a decimal in row %i.'%row_num)
            for f in self.factor_keys:
                if not row[f]: raise ValueError('Missing data for factor %s in row %i. Design must be balanced.'%(f, row_num))
            self.data.append(row)
            row_num += 1
        # store the number of subjects
        self.n = len(self.data)
        # store the number of unique combinations of conditions (e.g., number of cells)
        self.num_conditions = reduce(lambda x,y: x*y, [len(self.factor_levels[f]) for f in self.factor_levels.keys()])
        # store the number of subjects per condition
        self.n_per_condition = self.n/self.num_conditions
        if self.n != self.n_per_condition*self.num_conditions:
            raise ValueError('Design must be balanced (fully crossed with equal number of subjects per condition).')
        # ensure all contrasts are valid
        if contrasts:
            if not ANOVA.is_array_like(constrasts) and all([isinstance(x, ANOVAConstrast) for x in constrasts]):
                raise TypeError('contrasts must be specified as a list or tuple of ANOVAContrast definitions.')
            # ensure all contrasts are valid
            if not all([x.is_valid() for x in constrasts]):
                raise ValueError('One or more contrast definitions are invalid. All coefficients must sum to 0.')
        # store contrasts as a data attribute
        self.contrasts = contrasts
        
        # perform the test
        self.perform_test()
    
    def __create_new_source_for_results(self, name):
        """
        Create a new source of data for the results given the specified name
        """
        results = ANOVAResults()
        results.source_name = name
        return results
    
    def get_marginal_mean(self, **kwargs):
        """
        Get the marginal mean of a subset of the data, supplied as factor:level
        arguments
        """
        try: return stats.tmean([d[self.outcome_key] for d in self.data if all([d[k]==kwargs[k] for k in kwargs])])
        except: raise TypeError('You must specify at least one factor with a list containing at least one level.')
    
    def __get_mean_bracket_term(self, combination):
        """
        Get the bracket term corresponding to the supplied combination of
        factor keys
        """
        return (
                self.n_per_condition * 
                reduce(lambda x,y:x*y, [len(self.factor_levels[f]) for f in self.factor_keys if not f in combination]+[1]) * 
                sum([self.get_marginal_mean(**dict(zip(combination, prod)))**2 for prod in itertools.product(*(self.factor_levels[f] for f in combination))])
                )
    
    def perform_test(self):
        """
        Perform requested tests
        """
        # store some stuff for reuse
        combined_sample = [d[self.outcome_key] for d in self.data]
        grand_mean = stats.tmean(combined_sample)
        sum_of_grand_mean_squared = self.n*grand_mean**2
        sum_of_observations_squared = sum(d[self.outcome_key]**2 for d in self.data)
        highest_order_interaction_bracket_term = self.__get_mean_bracket_term(self.factor_keys) 
        ss_error = sum_of_observations_squared - highest_order_interaction_bracket_term
        df_error = float((self.n_per_condition-1)*self.num_conditions)
        ms_error = ss_error / df_error
        
        # store each main effect and interaction as a dictionary in a list
        self.results = list()
        bracket_terms = dict()
        for i in range(len(self.factor_keys)):
            combinations = itertools.combinations(self.factor_keys, i+1)
            for combination in combinations:
                bracket_terms[combination] = self.__get_mean_bracket_term(combination)
                ss_explained = (
                                bracket_terms[combination] + 
                                sum([bracket_terms[b]*(1 if len(b)%2==len(combination)%2 else -1) for b in bracket_terms.keys() if len(b) < len(combination) and all([t in combination for t in b])]) +
                                sum_of_grand_mean_squared*(1 if len(combination)%2==0 else -1)
                                )
                self.results.append(self.__create_new_source_for_results('x'.join(combination)))
                r = ANOVAResults
                
                self.results[-1].ss = ss_explained
                self.results[-1].df = reduce(lambda x, y: x*y, [len(self.factor_levels[f])-1.0 for f in combination])
                self.results[-1].ms = self.results[-1].ss / self.results[-1].df
                self.results[-1].test_statistic = self.results[-1].ms / ms_error
                self.results[-1].p_two = stats.f.sf(self.results[-1].test_statistic, self.results[-1].df, df_error)
                self.results[-1].effect_size = 0.0 # TODO
                self.results[-1].partial_effect_size = ANOVA.estimate_partial_effect_size(self.results[-1].test_statistic, self.results[-1].df, self.n)
        # fill in complete effects
        f_df_pairs = [(effect.test_statistic, effect.df) for effect in self.results]
        for i in range(len(self.results)):
            self.results[i].effect_size = ANOVA.estimate_complete_effect_size(
                                                                              self.results[i].test_statistic,
                                                                              self.results[i].df,
                                                                              self.n,
                                                                              *f_df_pairs
                                                                              )
        # add entry for error
        self.results.append(self.__create_new_source_for_results('Error'))
        self.results[-1].ss = ss_error
        self.results[-1].df = df_error
        self.results[-1].ms = ms_error
        # add entry for total
        self.results.append(self.__create_new_source_for_results('Total'))
        self.results[-1].ss = self.ss_total(grand_mean, combined_sample)
        self.results[-1].df = float(self.n-1)
        
        # TODO: simple effects
        # TODO: contrasts
        
        # print results if not in silent mode
        if not self.is_silent: print self.printable_test_results()
    
    @classmethod
    def estimate_contrast_complete_effect(cls, f_psi, f_omnibus, num_groups, group_size):
        """
        Estimate the complete effect size for a contrast, or the proportion of
        total variability that the contrast captures, assuming a balanced
        design
        """
        if not isinstance(f_psi, float) and not isinstance(f_psi, int):
            raise TypeError('F for contrast must be a decimal number')
        if not isinstance(f_omnibus, float) and not isinstance(f_omnibus, int):
            raise TypeError('Omnibus F must be a decimal number')
        if not isinstance(num_groups, int):
            raise TypeError('Number of groups must be an integer')
        if not isinstance(group_size, int):
            raise TypeError('Group size must be an integer')
        return (f_psi-1)/float((num_groups-1)*(f_omnibus-1)+num_groups*group_size)
    
    @classmethod
    def estimate_contrast_partial_effect(cls, f_psi, group_size):
        """
        Estimate the partial effect size for a contrast, or the variability of
        the contrast relative to itself and the error, assuming a balanced
        design
        """
        if not isinstance(f_psi, float) and not isinstance(f_psi, int):
            raise TypeError('F for contrast must be a decimal number')
        if not isinstance(group_size, int):
            raise TypeError('Group size must be an integer')
        print f_psi, group_size
        return (f_psi-1)/float(f_psi-1+2*group_size)
    
    @classmethod
    def estimate_partial_effect_size(cls, f, df, n):
        """
        Estimate the partial effect size of an effect based on its f, df, and
        total sample size
        """
        if not isinstance(f, float) and not isinstance(f, int):
            raise TypeError('F statistic must be a decimal number')
        if not isinstance(df, float) and not isinstance(df, int):
            raise TypeError('Degrees of freedom must be a decimal number')
        if not isinstance(n, float) and not isinstance(n, int):
            raise TypeError('Total sample size N must be a decimal number')
        return (df*(f-1)) / float(df*(f-1)+n)
    
    @classmethod
    def estimate_complete_effect_size(cls, f, df, n, *args):
        """
        Estimate the complete effect size of an effect based on its f, df,
        total sample size, and iterables of (F, df) for all effects
        """
        if not isinstance(f, float) and not isinstance(f, int):
            raise TypeError('F statistic must be a decimal number')
        if not isinstance(df, float) and not isinstance(df, int):
            raise TypeError('Degrees of freedom must be a decimal number')
        if not isinstance(n, float) and not isinstance(n, int):
            raise TypeError('Total sample size N must be a decimal number')
        if len(args) == 0 or not all(isinstance(x, list) or isinstance(x, tuple) and len(x)==2 and all(isinstance(y, float) or isinstance(y, int) for y in x) for x in args):
            raise ValueError('You must supply a variable-length positional argument list containing pairs of (F, df) for all effects')
        return (df*(f-1)) / float(sum((x[1]*(x[0]-1)) for x in args)+n)
    
    def printable_test_results(self):
        """
        Nicely format the results of a test for console output
        """
        col_wid = 16
        summary = u'Main Effects and Interactions:\n' + \
            u'---------------------------------------------------------------------------------------------------------------------------------------\n' + \
            u'Source                         SS               df               MS                F                p       partial \u03c9\u00b2      complete \u03c9\u00b2\n' + \
            u'---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ----------------'
        for r in self.results:
            summary += u'\n%s %s %s %s %s %s %s %s'%(
                                                     str(r.source_name[:col_wid-1]).ljust(col_wid),
                                                     str('%.5f'%r.ss if r.ss is not None else '').rjust(col_wid),
                                                     str('%.5f'%r.df if r.df is not None else '').rjust(col_wid),
                                                     str('%.5f'%r.ms if r.ms is not None else '').rjust(col_wid),
                                                     str('%.5f'%r.test_statistic if r.test_statistic is not None else '').rjust(col_wid),
                                                     str('%.5f'%r.p_two if r.p_two is not None else '').rjust(col_wid),
                                                     str('%.5f'%r.partial_effect_size if r.partial_effect_size is not None else '').rjust(col_wid),
                                                     str('%.5f'%r.effect_size if r.effect_size is not None else '').rjust(col_wid)
                                                     )
        '''
        if self.results[self.kContrastsKey]:
            summary += u'Contrasts:\n' + \
            u'---------------------------------------------------------------------------------------------------------------------------------------\n' + \
            u'Source                         SS               df               MS                F                p       partial \u03c9\u00b2      complete \u03c9\u00b2\n' + \
            u'---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ----------------'
            for r in self.results[self.kContrastsKey]:
                summary += u'\n%s %s %s %s %s %s %s %s'%(
                                                         str(r[self.kSourceName][:col_wid-1]).ljust(col_wid),
                                                         str('%.5f'%r[self.kSSKey] if r[self.kSSKey] is not None else '').rjust(col_wid),
                                                         str('%.5f'%r[self.kDFKey] if r[self.kDFKey] is not None else '').rjust(col_wid),
                                                         str('%.5f'%r[self.kMSKey] if r[self.kMSKey] is not None else '').rjust(col_wid),
                                                         str('%.5f'%r[self.kTestStatKey] if r[self.kTestStatKey] is not None else '').rjust(col_wid),
                                                         str('%.5f'%r[self.kP2Key] if r[self.kP2Key] is not None else '').rjust(col_wid),
                                                         str('%.5f'%r[self.kPartialEffectSizeKey] if r[self.kPartialEffectSizeKey] is not None else '').rjust(col_wid),
                                                         str('%.5f'%r[self.kEffectSizeKey] if r[self.kEffectSizeKey] is not None else '').rjust(col_wid)
                                                         )
        '''
        return summary.rstrip()
    
    @classmethod
    def perform_unit_test(cls, **kwargs):
        """
        Perform unit tests for different conditions. Most data are from:
        Keppel, G. & Wickens, T.D. (2004). Design and Analysis: A Researcher's
            Handbook, 4th ed. Pearson Prentice Hall: Upper Saddle River, NJ.
        """
        results = list()
        kwargs['test_name'] = 'One-Way'
        kwargs['input_file'] = '%s - one-way.csv'%cls.get_unit_test_input_file_name()[:-4]
        kwargs['suffix'] = ' - one-way'
        results.append(super(ANOVA, cls).perform_unit_test(**kwargs))
        kwargs['test_name'] = 'Two-Way'
        kwargs['input_file'] = '%s - two-way.csv'%cls.get_unit_test_input_file_name()[:-4]
        kwargs['suffix'] = ' - two-way'
        kwargs['factor_keys'] = ['Drug', 'Deprivation']
        results.append(super(ANOVA, cls).perform_unit_test(**kwargs))
        kwargs['test_name'] = 'Three-Way'
        kwargs['input_file'] = '%s - three-way.csv'%cls.get_unit_test_input_file_name()[:-4]
        kwargs['suffix'] = ' - three-way'
        kwargs['factor_keys'] = ['Feedback', 'Word Type', 'Grade']
        results.append(super(ANOVA, cls).perform_unit_test(**kwargs))
        return results

'''
kGrandMean = 'Grand Mean'
kDFWithin = 'dfWithin'
kDFBetween = 'dfBetween'
kDFTotal = 'dfTotal'
kSSWithin = 'SSWithin'
kSSBetween = 'SSBetween'
kSSTotal = 'SSTotal'
kMSWithin = 'MSWithin'
kMSBetween = 'MSBetween'
kTestStat = 'F'
kContrastTestStat = 'Psi-hat'
kPValue = 'p'
kContrasts = 'Contrasts'
kFactorNames = 'Factors'
kFactorLevelNames = 'Factor Levels'
kEffectSize = 'Effect Size'
kPartialEffectSize = 'Partial Effect Size'

kwGrandMean = 'grand_mean'
kwCombinedSample = 'combined_sample'

def one_way(input_file='',                              # path to a csv file
            factor_key='Group', outcome_key='Outcome',  # keys for the columns specifying the factor and the outcome variable
            alpha=0.05,                                 # values for the hypothesis test
            contrasts=[],                               # should be a list of dictionaries, where each dictionary specifies factor_level : contrast_coefficient pairs for at least the factor levels involved
            is_silent=False):                           # should the test print results to the console? 
    """Perform a one-way ANOVA on the supplied data across the supplied factor"""
    if not os.path.isfile(input_file):
        raise IOError('Unable to load specified file: %s'%input_file)
    if not factor_key or not outcome_key:
        raise ValueError('Keys must be specified for factor and outcome categories')
    factor_levels = dict((k, []) for k in set(entry[factor_key] for entry in csv.DictReader(open(input_file)) if entry[factor_key]))
    for row in csv.DictReader(open(input_file)):
        if not row[factor_key]:
            sys.stderr.write('Row contains no factor assignment in file %s: %s. Data is is being skipped.\n'%(input_file, row))
            continue
        if not row[outcome_key]:
            sys.stderr.write('Row contains no outcome assignment in file %s: %s. Data is is being skipped.\n'%(input_file, row))
            continue
        try: factor_levels[row[factor_key]].append(float(row[outcome_key]))
        except: raise ValueError('Error converting value to a decimal: %s'%row[outcome_key])
    
    # a dictionary of results for the test
    results = dict()
    
    # store some stuff for reuse
    list_of_group_data = [factor_levels[f] for f in factor_levels.keys()]
    combined_sample = combine_samples(*list_of_group_data)
    group_means = dict((f, stats.tmean(factor_levels[f])) for f in factor_levels.keys())
    
    # determine df values
    n = len(combined_sample)
    results[kDFTotal] = n-1
    results[kDFBetween] = len(factor_levels.keys())-1
    results[kDFWithin] = n-len(factor_levels.keys())
    
    # compute grand mean
    results[kGrandMean] = stats.tmean(combined_sample)
    
    # compute SS values
    results[kSSBetween] = ss_between(*list_of_group_data)
    results[kSSTotal] = ss_total(*list_of_group_data)
    results[kSSWithin] = ss_within(*list_of_group_data)
    
    # compute MS values
    results[kMSBetween] = results[kSSBetween] / float(results[kDFBetween])
    results[kMSWithin] = results[kSSWithin] / float(results[kDFWithin])
    
    # compute test statistic
    results[kTestStat], results[kPValue] = stats.f_oneway(*[factor_levels[f] for f in factor_levels.keys()])
    results[kEffectSize] = estimate_omnibus_effect_size_from_summary(results[kSSBetween], results[kSSTotal], results[kMSWithin], len(factor_levels.keys()))
    
    # perform contrasts
    results[kContrasts] = list()
    for contrast in contrasts:
        contrast_results = dict()
        contrast_results[kContrastTestStat] = sum(contrast.setdefault(f, 0)*group_means[f] for f in factor_levels.keys())
        contrast_results[kSSBetween] = contrast_results[kContrastTestStat]**2 / float(sum(contrast[f]**2/float(len(factor_levels[f])) for f in factor_levels.keys()))
        contrast_results[kDFBetween] = 1
        contrast_results[kTestStat] = contrast_results[kSSBetween] / results[kMSWithin]
        contrast_results[kEffectSize] = estimate_contrast_complete_effect(contrast_results[kTestStat], results[kTestStat], len(list_of_group_data), len(list_of_group_data[0]))
        contrast_results[kPartialEffectSize] = estimate_contrast_partial_effect(contrast_results[kTestStat], len(list_of_group_data[0]))
        sorted_names = [f for f in factor_levels.keys()]
        sorted_names.sort()
        contrast_results[kFactorLevelNames] = ['(%s)%s'%(('%.3f'%contrast[f]).rjust(6), f) for f in sorted_names]
        results[kContrasts].append(contrast_results)
    
    # configure other output
    results[kFactorNames] = [factor_key]
        
    # print the results if requested
    if not is_silent: print_oneway_test_results(results)
    
    # return the results
    return results

def estimate_contrast_complete_effect(f_psi, f_omnibus, num_groups, group_size):
    """Estimate the complete effect size for a contrast, or the proportion of total variability that the contrast captures"""
    if not isinstance(f_psi, float) and not isinstance(f_psi, int): raise TypeError('F for contrast must be a decimal number')
    if not isinstance(f_omnibus, float) and not isinstance(f_omnibus, int): raise TypeError('Omnibus F must be a decimal number')
    if not isinstance(num_groups, int): raise TypeError('Number of groups must be an integer')
    if not isinstance(group_size, int): raise TypeError('Group size must be an integer')
    return (f_psi-1)/float((num_groups-1)*(f_omnibus-1)+num_groups*group_size)

def estimate_contrast_partial_effect(f_psi, group_size):
    """Estimate the partial effect size for a contrast, or the variability of the contrast relative to itself and the error"""
    if not isinstance(f_psi, float) and not isinstance(f_psi, int): raise TypeError('F for contrast must be a decimal number')
    if not isinstance(group_size, int): raise TypeError('Group size must be an integer')
    print f_psi, group_size
    return (f_psi-1)/float(f_psi-1+2*group_size)

def estimate_omnibus_effect_size_from_f(f_omnibus, number_of_groups, group_size):
    """Return an estimate of omega squared using the supplied parameters. Note: assumes equal group size. I don't know how to do it for unequal groups."""
    if not isinstance(f_omnibus, float) and not isinstance(f_omnibus, int): raise TypeError('Omnibus F must be a decimal number')
    if not isinstance(number_of_groups, int): raise TypeError('Number of groups must be an integer')
    if not isinstance(group_size, int): raise TypeError('Group size must be an integer')
    f_omnibus = float(f_omnibus)
    return (number_of_groups-1)*(f_omnibus-1) / ((number_of_groups-1)*(f_omnibus-1)+number_of_groups*group_size)

def estimate_omnibus_effect_size_from_summary(ssb, sst, msw, number_of_groups):
    """Return an estimate of omega squared using the supplied parameters."""
    if not isinstance(ssb, float) and not isinstance(ssb, int): raise TypeError('Sum of squares between must be a decimal number')
    if not isinstance(sst, float) and not isinstance(sst, int): raise TypeError('Sum of squares total must be a decimal number')
    if not isinstance(msw, float) and not isinstance(msw, int): raise TypeError('Sum of squares total must be a decimal number')
    if not isinstance(number_of_groups, int): raise TypeError('Number of groups must be an integer')
    ssb = float(ssb)
    sst = float(sst)
    msw = float(msw)
    return (ssb-(number_of_groups-1)*msw) / (sst+msw)

def combine_samples(*args):
    """Combine several lists/tuples into a single list"""
    try: return list(itertools.chain.from_iterable([a for a in args]))
    except: raise TypeError('Expected only lists or tuples')

def get_grand_mean(*args):
    """Get the grand mean of several lists/tuples of data"""
    return stats.tmean(combine_samples(*args))

def ss_between(*args, **kwargs):
    """Get the sum of squared deviations of each group's mean compared to the grand mean of all groups"""
    grand_mean = kwargs[kwGrandMean] if kwGrandMean in kwargs else get_grand_mean(*args)
    return sum([len(a)*(grand_mean-stats.tmean(a))**2 for a in args])

def __get_variability_of_group_means_around_grand_mean(*args, **kwargs):
    """Get the variability of the group means around the grand mean"""
    grand_mean = kwargs[kwGrandMean] if kwGrandMean in kwargs else get_grand_mean(args)
    return sum((grand_mean-stats.tmean(g))**2 for g in args)/float(len(args))

def ss_total(*args, **kwargs):
    """Get the sum of squared deviations of each value compared to the grand mean"""
    grand_mean = kwargs[kwGrandMean] if kwGrandMean in kwargs else get_grand_mean(*args)
    combined_sample = kwargs[kwCombinedSample] if kwCombinedSample in kwargs else combine_samples(*args)
    return sum([(x-grand_mean)**2 for x in combined_sample])

def pooled_standard_error(*args):
    """Get the pooled standard error of the groups"""
    try: return sum(len(a)*stats.tvar(a) for a in args)/float(sum(len(a)-1 for a in args))
    except: raise TypeError('Expected only lists or tuples')

def ss_within(*args):
    """Get the sum of square deviations of each value compared to its group mean value"""
    try: return sum((len(a)-1)*stats.tvar(a) for a in args)
    except: raise TypeError('Expected only lists or tuples')

'''
"""
def print_oneway_test_results(results):
    x = results[kTestStat]
    col_wid= 16
    summary = u'''
Omnibus Test Results:
----------------------------------------------------------------------------------------------------------------------
Source                         SS               df               MS                F                p               \u03c9\u00b2
---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ----------------
%s %s %s %s %s %s %s
Within           %s %s %s
Total            %s %s
'''%(
     (results[kFactorNames][0]).ljust(col_wid),
     ('%.5f'%results[kSSBetween]).rjust(col_wid), ('%.5f'%results[kDFBetween]).rjust(col_wid), ('%.5f'%results[kMSBetween]).rjust(col_wid), ('%.5f'%x).rjust(col_wid), ('%.5f'%(1.0-stats.f.cdf(x, results[kDFBetween], results[kDFWithin]))).rjust(col_wid), ('%.5f'%results[kEffectSize]).rjust(col_wid), 
     ('%.5f'%results[kSSWithin]).rjust(col_wid), ('%.5f'%results[kDFWithin]).rjust(col_wid), ('%.5f'%results[kMSWithin]).rjust(col_wid),
     ('%.5f'%results[kSSTotal]).rjust(col_wid), ('%.5f'%results[kDFTotal]).rjust(col_wid)
     )
    contrasts = ''
    if results[kContrasts]:
        contrasts = u'''
Contrast Results:
---------------------------------------------------------------------------------------------------------------------------------------
Source                         SS               df               MS                F                p       partial \u03c9\u00b2      complete \u03c9\u00b2
---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ----------------'''
        for c in results[kContrasts]:
            contrasts += u'''
%s %s %s %s %s %s %s %s\n'''%(
                      ('\n'.join(('%s'%f).ljust(col_wid) for f in c[kFactorLevelNames])),
                      ('%.5f'%c[kSSBetween]).rjust(col_wid),
                      ('%.5f'%c[kDFBetween]).rjust(col_wid),
                      ('%.5f'%c[kSSBetween]).rjust(col_wid),
                      ('%.5f'%c[kTestStat]).rjust(col_wid),
                      ('%.7f'%(1.0-stats.f.cdf(c[kTestStat], 1, results[kDFWithin]))).rjust(col_wid),
                      ('%.7f'%c[kPartialEffectSize]).rjust(col_wid),
                      ('%.7f'%c[kEffectSize]).rjust(col_wid)
                      )
        contrasts += u'''
Within           %s %s %s
'''%(
     str(results[kSSWithin]).rjust(col_wid), str(results[kDFWithin]).rjust(col_wid), str(results[kMSWithin]).rjust(col_wid)[:col_wid])
    print summary+contrasts
"""