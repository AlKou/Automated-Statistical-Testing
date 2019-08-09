#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import statsmodels.api as sm

class AutoTesting():  
 
    def __init__(self):
        pass
        
    def data_ver(self, *args):
        '''Verify and process input data into a list of n x 1 ndarrays.
        
        Parameters:
        ----------
        *args : numpy.ndarrays with shape (n, 1), or (n, m)
        
        Returns: 
        -------
        x : list
            Of n x 1 numpy.ndarrays. 
        
        Notes:
        -----
        This allows flexible data input and n = len(args)
        e.g.: With two ndarrays shaped 300 x 2 and 100 x 3, the returned
        list length would be 5.'''
           
        x = []
        for arg in args: 
            if type(arg) is np.ndarray:
                if arg.shape != (len(arg), ): # Split 2D data into 1D sets
                    for i in range(arg.shape[1]):
                        x.append(arg[:, i].reshape(-1, 1))
                else: # Convert 1D data into 2D
                    x.append(arg.reshape(-1, 1))
            else:
                raise ValueError('Only n x 1 or n x m np.ndarrays allowed!')
                
        return x
        
        
    def norm_cal(self, x):
        
        '''Calculate the normality of a single variable x. 
        
        Parameters:
        ----------
        x : numpy.ndarray
        
        Returns:
        -------
        x_res : dict
            'Statistic': statistic value calculated by the test
            'Pvalue':  p-value calculated by the test
            'Critical': critical value if Anderson-Darling is used
            'Test': name of the test used
            'Sample size': sample size of the variable 
            'Result': bool, True if p-value < .5, False otherwise
        
        Notes:
        -----
        More conservative cutoff numbers of 3500 and 50 are chosen 
        based on below test conventions:
        Jarque_bera requires 2000+ samples;
        Shapiro-Wilk is accurate under 5000;
        And common difinition of small sample size is 30'''
        
        x_res = {}
        if len(x) >= 3500: # Use Jarque_bera for samples larger 3500
            x_res['Statistic'] = ss.jarque_bera(x)[0]
            x_res['Pvalue'] = ss.jarque_bera(x)[1]
            x_res['Test'] = 'Jarque Bera Test'
            x_res['Sample Size'] = x.shape
        elif len(x) >= 50: # Use Shapiro-Wilk for samples  [50  3500)
            x_res['Statistic'] = ss.shapiro(x)[0]
            x_res['Pvalue'] = ss.shapiro(x)[1]
            x_res['Test'] = 'Shapiro-Wilk Test'
            x_res['Sample Size'] = x.shape
        else: # Use Anderson-Darling for samples less than 50
            x_res['Statistic'] = ss.anderson(x)[0][2]
            x_res['Critical'] = ss.anderson(x)[1][2]
            x_res['Test'] = 'Aderson-Darling Test'
            x_res['Sample Size'] = x.shape
            
        if x_res['Test'] != 'Aderson-Darling Test':
            if x_res['Pvalue'] < .05: # Fixed significance level
                x_res['Result'] = False
            else:
                x_res['Result'] = True
        else: # Anderson-Darling result has to be specially handled
            if x_res['Critical'] < x_res['Statistic']:
                x_res['Result'] = False
            else:
                x_res['Result'] = True
                
        return x_res
    
    def norm_report(self, x_res, i):
        
        '''Display statistic values of the normality test for ith variable x. 
        
        Parameters:
        ----------
        x_res : dict 
            Results of the variable's normality test.
        i : int
            The ith variable, passed from norm_test()
            
        Returns:
        -------
        None
        
        Notes:
        -----
        None'''
        
        if x_res['Sample Size'][0] < 50: # Alert if sample size is small
            print('\n>>> Warning: Small sample size! Low power!\n')
        print('\nVariable:\t', i)
        print('Testing Method:\t', x_res['Test'])
        print('Sample Size: \t', x_res['Sample Size'][0])
        print('Statistic: \t', x_res['Statistic'])
        
        if x_res['Test'] != 'Aderson-Darling Test':
            print('Pvalue: \t', x_res['Pvalue'])
        else:
            print('Critical:\t', x_res['Critical'])
            
        if x_res['Result']:
            print('Test Result: \t Failed to reject! Data is normal! ')
        else:
            print('Test Result:\t Rejected! Data is not normal!')
            
    def norm_plot(self, x):
        
        '''Generate subplots of QQPlot and histgram to visualize
        the normality of a variable.
           
        Parameters:
        ----------
        x : numpy.ndarray
            The variable to plot
            
        Returns:
        -------
        ax1 : matplotlib.axes
            To plot the QQplot of variable x
        ax2 : matplotlib.axes
            To plot histogram of variable x
        
        Notes:
        -----
        None'''
        
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))
        
        qlt = sm.ProbPlot(x.reshape(-1), fit=True)
        qq = qlt.qqplot(marker='o', color='coral', ax=ax1)
        sm.qqline(qq.axes[0], line='45', fmt='g--')
        
        ax2.hist(x, color='orange', alpha=.6)
        
        return ax1, ax2
    
    def norm_test(self, x, report=True):
        
        '''Test the normality of the variable list x.
        
        Parameters:
        ----------
        x : numpy.ndarrays
            The variables to test normality on
        report : bool
            True to display report, False not.
            
        Returns:
        -------
        res : dict
            'Variable 1': True if failed to reject H0, False if not.
            'Variable 2': True if failed to reject H0, False if not.
                                        ...
            'Variable n': True if failed to reject H0, False if not.
            
        Notes:
        -----
        None'''
        
        if report:
            print('\n>>> Testing Normality...')
            
        res = {}
        for i in range(len(x)): # Iterate through the variables
            x_res= self.norm_cal(x[i])
            res['Variable {}'.format(i+1)] = x_res['Result']
            
            if report: # Plots
                self.norm_report(x_res, i+1)
                ax1, ax2 = self.norm_plot(x[i])
                ax1.set_title('QQPlot of Variable {}'.format(i+1))
                ax2.set_title('Histgram of Variable {}'.format(i+1))
                plt.show()
                print('-'*60)
                
        if report:
            print('\n>>> Normality Test Done!')
            print('='*60)
            
        return res
    
    def get_center(self, skews):
        
        '''Use skewness to decide the center parameter for Levene test.
        
        Parameters:
        ----------
        skews : list
            Skewness values used to dicide the center parameter.
            
        Returns:
        -------
        center : str
            'mean': if relatively symmetry;
            'median' if skewed;
            'trimmed' if heavily-skewed.
        
        Notes:
        -----
        The rule of thumb:
            -Not skewed:      (-.5, .5)
            -Skewed:          (-1, -.5] U [.5, 1)
            -Heavily-Skewed:  (small, -1] U [1, large)'''
        
        skews = np.array(skews)
        
        if any(abs(skews) >= 1): # Heavily skewed
            center = 'trimmed'
        elif any(abs(skews) >= .5): # Skewed
            center = 'median'
        else: # Not skewed
            center = 'mean'
            
        return center
    
    def homo_cal(self, x, norm_res, skews):
        
        '''Calculate and return the variance homogeneity test result.
        
        Parameters:
        ----------
        x : numpy.ndarrays
            The variables to test on
        norm_res : dict 
            Results of normality test
        skews : list
            Skewness values of variables of x
        
        Returns:
        -------
        homo_res : dict
            'Variables': number of variables tested
            'Statistic': statistic value calculated by the test
            'Pvalue': p-value value calculated by the test
            'Test': name of the test used
            'Result': True if homogeneous, False otherwise
            
        Notes:
        ------
        Both Barlett Test and Levene Test don't require equal
        sample sizes'''
        
        homo_res = {}
        if sum(norm_res.values()) == len(x) and all(abs(np.array(skews)) < .5): 
        # All normal, use Barlett Test
            homo_res['Variables'] = len(x)
            homo_res['Statistic'] = ss.bartlett(*x)[0]
            homo_res['Pvalue'] = ss.bartlett(*x)[1]
            homo_res['Test'] = 'Bartlett Test'
        else:  # Not all normal, use unparametric Levene Test
            c = self.get_center(skews)
            homo_res['Variables'] = len(x)
            homo_res['Statistic'] = ss.levene(*x, center=c)[0]
            homo_res['Pvalue'] = ss.levene(*x, center=c)[1]
            homo_res['Test'] = 'Levene Test'
            # Also Fligner-Killeen Test is an option
        if homo_res['Pvalue'] >= .05:
            homo_res['Result'] = True
        else:
            homo_res['Result'] =  False
            
        return homo_res
    
    def homo_report(self, homo_res):
        
        '''Display statistic values of the homogeneity test. 
        
        Parameters:
        ----------
        homo_res : dict
            Test results from variance homogeneity test.
        
        Returns:
        -------
        None
        
        Notes:
        ------
        None'''
        
        print('\nVariables:\t', homo_res['Variables'])
        print('Testing Method:\t', homo_res['Test'])
        print('Statistic:\t', homo_res['Statistic'])
        print('Pvalue: \t', homo_res['Pvalue'])
        
        if homo_res['Result']:
            print('Test Result:\t', 'Failed to reject! Variances equal!')
        else:
            print('Test Result:\t', 'Rejected! Variances unequal!')
    
    def homo_test(self, x, report=True):
        
        '''Homogeneity of Variance testing. 
        
        Parameters:
        ----------
        x : numpy.ndarrays
            Variables to test on
        report : bool
            True to display report, False not.
            
        Returns:
        -------
        res : dict
            'Variables': number of variables tested on
            'Result': True if variances equal, False if not.
            
        Notes:
        -----
        Use Bartlett Test when normality test shows all variables are 
        normal and all skewnesses below .5. Otherwise, use Levene Test.'''
        
        if len(x) == 1: # Verify if there are at least two variables
            print('\n>>> Error: Input dadta with at least two variables!\n')
        else:
            skews = [ss.skew(i) for i in x]
            norm_res = self.norm_test(x, report)
            homo_res = {}
            
            if report:
                print('\n>>> Testing Homogeneity of Variance...\n')
                for i in range(len(x)):
                    print('Variance {}:\t'.format(i+1), np.var(x[i], ddof=1))
                
            homo_res = self.homo_cal(x, norm_res, skews)
            res = {}
            res['Result'] = homo_res['Result']
            res['Variables'] = len(x)
            
            if report:
                self.homo_report(homo_res)
                print('\n>>> Homogeneity of Variance Test Done!')
                print('='*60)
                
            return res
                
    def two_cal(self, x, norm_res, homo_res, skews):
        
        '''Calculate and return two sample comparison tests results.
        
        Parameters:
        ----------
        x : numpy.ndarrays
            Variables to test on
        norm_res : dict
            Normality test results
        homo_res : dict
            Homogeneity test rerults
        skews : list
            Skewness values of all x variables
            
        Returns:
        -------
        res_ind : dict
            Test results for independent samples
            True if failed to reject hypothesis, False if not.
        res_prd : dict
            Test results for paired samples. 
            True if failed to reject hypothesis, False if not.
            
        Notes:
        -----
        None'''
        
        res_ind = {}
        res_prd = {}
        
        if sum(norm_res.values()) == len(x) and all(abs(np.array(skews)) < .5): 
        # All normal                

            # Independent variables
            if homo_res: # Variance equal
                res_ind['Statistic'] = ss.ttest_ind(x[0], x[1])[0]
                res_ind['Pvalue'] = ss.ttest_ind(x[0], x[1])[1]
                res_ind['Test'] = 'T-test with independent samples'
            else: # Variance unequal
                res_ind['Statistic'] = ss.ttest_ind(x[0], x[1], equal_var=False)[0]
                res_ind['Pvalue'] = ss.ttest_ind(x[0], x[1], equal_var=False)[1]
                res_ind['Test'] = 'Welch\'s T-test'

            # Paired variables. Results from independent samples test are overrid
            res_prd['Statistic'] = ss.ttest_rel(x[0], x[1])[0]
            res_prd['Pvalue'] = ss.ttest_rel(x[0], x[1])[1]
            res_prd['Test'] = 'T-test with paired samples'

        else: # If not all normal, use unparametric tests.
              # Don't have to assumme variance equality, 
              # but do have to consider sample size                

            # Independent variables
            if all([len(x[0]), len(x[1])]) >= 20: # Sample size > 20
                res_ind['Statistic'] = ss.mannwhitneyu(x[0], x[1])[0]
                res_ind['Pvalue'] = ss.mannwhitneyu(x[0], x[1])[1]
                res_ind['Test'] = 'Mann-Whitney U Test with Indepedent Samples'
            else: # Sample size < 20
                res_ind['Statistic'] = ss.ranksums(x[0], x[1])[0]
                res_ind['Pvalue'] = ss.ranksums(x[0], x[1])[1]
                res_ind['Test'] = 'Wilcoxon rank-sum Test with Indepedent Samples'

            # Paired variables. Results from independent samples test are overridden
            res_prd['Statistic'] = ss.wilcoxon(x[0].reshape(-1), x[1].reshape(-1))[0]
            res_prd['Pvalue'] = ss.wilcoxon(x[0].reshape(-1), x[1].reshape(-1))[1]
            res_prd['Test'] = 'Wilcoxon signed-rank Test with Paired Samples'
        
        # Get the results based on fixed significance level
        if res_ind['Pvalue'] >= .05:
            res_ind['Result'] = True
        else:
            res_ind['Result'] =False

        if res_prd['Pvalue'] >= .05:
            res_prd['Result'] = True
        else:
            res_prd['Result'] =False
            
        return res_ind, res_prd
            
    def two_report(self, x, res_ind, res_prd):
        
        '''Display statistic values of two samples comparison tests.
        
        Parameters:
        ----------
        x : numpy.ndarrays
            Variables tested on
        res_ind : dict
            Results for independent samples comparison test
        res_prd : dict
            Results for paired samples comparison test
            
        Returns:
        -------
        None
        
        Notes:
        -----
        None'''
        
        if 'T-test' in res_ind['Test']: # Parametric make hypothesis about means
            print('\nMean 1:\t', x[0].mean())
            print('Mean 2:\t', x[1].mean(),'\n')
        else: # Unparametric make hypothesis about medians
            print('\nMedian 1:\t', np.median(x[0]))
            print('Median 2:\t', np.median(x[1]),'\n')
        
        # Independent samples
        print('Testing Method:\t', res_ind['Test'])
        print('Statistic:\t', res_ind['Statistic'])
        print('Pvalue: \t', res_ind['Pvalue'])
        if res_ind['Result']:
            print('Test Result:\t', 'Failed to reject! Means equal!\n')
        else:
            print('Test Result:\t', 'Reject! Medians unequal!\n')
         
        # Paired samples
        print('Testing Method:\t', res_prd['Test'])
        print('Statistic:\t', res_prd['Statistic'])
        print('Pvalue: \t', res_prd['Pvalue'])
        if res_prd['Result']:
            print('Test Result:\t', 'Failed to reject! Means equal!')
        else:
            print('Test Result:\t', 'Reject! Medians unequal!')
                
    def compr_two(self, x, report=True):
        '''Conduct and return two samples comparision tests.
        
        Parameters:
        ----------
        x : numpy.ndarrays
            Variables to compare
        report : bool
            True to display report, False not.
            
        Returns:
        -------
        A dict of results for independent and paired tests. 
        True if equal, False if not.
            
        Notes:
        -----
            Both independent and paired samples comparison 
            results are calculated and returned. '''
        
        if len(x) == 1: # Data verification
            raise ValueError('\n>>> Error: Input dadta with two variables!\n')
        else:
            if report:
                print('\n>>> Comparing Two Variables...\n')
                
            skews = [ss.skew(i) for i in x]
            norm_res = self.norm_test(x, False)
            homo_res = self.homo_test(x, report)['Result']
            two_res = {}
            res_ind, res_prd = self.two_cal(x, norm_res, homo_res, skews)
            two_res['Independent Samples'] = res_ind['Result']
            two_res['Paired Samples'] = res_prd['Result']
            
            if report:  
                self.two_report(x, res_ind, res_prd)
                print('\n>>> Two-Variable Comparison Test Done!')
                print('='*60)
                
            return two_res
            
