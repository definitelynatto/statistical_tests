"""
A module containing assorted multiple comparison procedures
"""

from scipy import stats

def __validate_alpha(alpha):
    """Validate alpha value"""
    if not isinstance(alpha, float):
        raise TypeError('alpha must be a decimal number')
    if alpha > 1 or alpha < 0: raise ValueError('alpha must be in the range [0, 1]')
    
def __validate_args(alpha, p_values):
    """Validate input parameters"""
    __validate_alpha(alpha)
    if not isinstance(p_values, list) and not isinstance(p_values, tuple):
        raise TypeError('p-values must be supplied as a list or tuple')

def bonferroni_dunn(alpha, p_values):
    """Perform the so-called Bonferonni-Dunn procedure, which should actually be called the Fisher-Bool procedure. Return True if all p-values are significant."""
    __validate_args(alpha, p_values)
    pc = alpha/float(len(p_values))
    for p in p_values:
        if p > pc: return False
    return True

def sidak_bonferroni(alpha, p_values):
    """Perform the Sidak-Bonferroni procedure, which is marginally more powerful than the Bonferonni-Dunn. Return True if all p-values are significant"""
    __validate_args(alpha, p_values)
    pc = 1-(1-alpha)**(1/float(len(p_values)))
    for p in p_values:
        if p > pc: return False
    return True

def holm_test(alpha, p_values):
    """Perform the Holm sequentially rejective procedure using the supplied alpha and p-values. Return True if all p-values are significant."""
    __validate_args(alpha, p_values)
    p_values.sort()
    for i in range(len(p_values)):
        if p_values[i] > alpha/float(len(p_values)-i): return False
    return True

def scheffe_cv(alpha, num_levels, num_subjects):
    """Obtain a Scheffe critical value using the specified alpha, number of levels for the factor, and number of subjects in the entire sample."""
    __validate_alpha(alpha)
    if not isinstance(num_levels, int): raise TypeError('num_levels must be an integer describing the number of levels for the factor in question')
    if not num_levels > 1: raise ValueError('There must be more than one level for the factor')
    if not num_subjects > 0: raise ValueError('There must be at least 1 (ideally many more) subjects')
    return (num_levels-1)*stats.f.ppf(1.0-alpha)
    