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
            print('Test Result: \t Failed to reject! Normal! ')
        else:
            print('Test Result:\t Rejected! Not normal!')
            
    def norm_plot(self, x):
        
        '''Generate subplots of QQPlot and histgram to visualize
        the normality of a variable.
           
        Parameters:
        ----------
        x : list of numpy.ndarray
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
        x : list of numpy.ndarray
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
        
        '''Calculate and return the homoscedasticity test result.
        
        Parameters:
        ----------
        x : list of numpy.ndarray
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
            'Pvalue': p-value calculated by the test
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
        
        '''Display statistic values of the homoscedasticity test. 
        
        Parameters:
        ----------
        homo_res : dict
            Test results from homoscedasticity test.
        
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
        
        '''Homoscedasticity testing. 
        
        Parameters:
        ----------
        x : list of numpy.ndarray
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
                
    def two_cal(self, x, norm_res, homo_res, skews, paired=False):
        
        '''Calculate and return two samples comparison tests results.
        
        Parameters:
        ----------
        x : list of numpy.ndarray
            Variables to test on
        norm_res : dict
            Normality test results
        homo_res : dict
            Homogeneity test rerults
        skews : list
            Skewness values of all x variables
        paired : bool
            False for two independent variables. True Otherwise
            
        Returns:
        -------
        res : dict
            'Statistic': statistic value calculated by the test
            'Pvalue': p-value calculated by the test
            'Test': name of the test used
            'Result': True if failed to reject null hypothesis, False otherwise
            
        Notes:
        -----
        None'''
        
        res = {}
        
        if sum(norm_res.values()) == len(x) and all(abs(np.array(skews)) < .5): 
        # If all normal 
            if paired:# Paired samples
                res['Statistic'] = ss.ttest_rel(x[0], x[1])[0]
                res['Pvalue'] = ss.ttest_rel(x[0], x[1])[1]
                res['Test'] = 'T-test with paired samples'
            else: # Independent samples
                if homo_res: # Variances equal
                    res['Statistic'] = ss.ttest_ind(x[0], x[1])[0]
                    res['Pvalue'] = ss.ttest_ind(x[0], x[1])[1]
                    res['Test'] = 'T-test with independent samples'
                else: # Variances unequal
                    res['Statistic'] = ss.ttest_ind(x[0], x[1], equal_var=False)[0]
                    res['Pvalue'] = ss.ttest_ind(x[0], x[1], equal_var=False)[1]
                    res['Test'] = 'Welch\'s T-test'

        else: # If not all normal, use unparametric tests.  
            if paired: # Paired samples
                res['Statistic'] = ss.wilcoxon(x[0].reshape(-1), x[1].reshape(-1))[0]
                res['Pvalue'] = ss.wilcoxon(x[0].reshape(-1), x[1].reshape(-1))[1]
                res['Test'] = 'Wilcoxon signed-rank Test with Paired Samples'
            else:# Independent samples
                if all([len(x[0]), len(x[1])]) >= 20: # Sample size > 20
                    res['Statistic'] = ss.mannwhitneyu(x[0], x[1])[0]
                    res['Pvalue'] = ss.mannwhitneyu(x[0], x[1])[1]
                    res['Test'] = 'Mann-Whitney U Test with Indepedent Samples'
                else: # Sample size < 20
                    res['Statistic'] = ss.ranksums(x[0], x[1])[0]
                    res['Pvalue'] = ss.ranksums(x[0], x[1])[1]
                    res['Test'] = 'Wilcoxon rank-sum Test with Indepedent Samples'

        # Get the results based on fixed significance level
        if res['Pvalue'] >= .05:
            res['Result'] = True
        else:
            res['Result'] =False
            
        return res
            
    def two_report(self, x, res):
        
        '''Display statistic values of two samples comparison tests.
        
        Parameters:
        ----------
        x : list of numpy.ndarray
            Variables tested on
        res : dict
            Results of a two samples comparison test
            
        Returns:
        -------
        None
        
        Notes:
        -----
        None'''
        
        if 'T-test' in res['Test']: # Parametric makes hypothesis about means
            print('\nMean 1:\t', x[0].mean())
            print('Mean 2:\t', x[1].mean(),'\n')
            print('Testing Method:\t', res['Test'])
            print('Statistic:\t', res['Statistic'])
            print('Pvalue: \t', res['Pvalue'])
            if res['Result']:
                print('Test Result:\t', 'Failed to reject! Means equal!\n')
            else:
                print('Test Result:\t', 'Reject! Means unequal!\n')
        else: # Unparametric makes hypothesis about medians
            print('\nMedian 1:\t', np.median(x[0]))
            print('Median 2:\t', np.median(x[1]),'\n')
            print('Testing Method:\t', res['Test'])
            print('Statistic:\t', res['Statistic'])
            print('Pvalue: \t', res['Pvalue'])
            if res['Result']:
                print('Test Result:\t', 'Failed to reject! Medians equal!\n')
            else:
                print('Test Result:\t', 'Reject! Medians unequal!\n')
                
    def compr_two(self, x, paired=False, report=True):
        '''Conduct and return two samples comparision tests.
        
        Parameters:
        ----------
        x : list of numpy.ndarrays
            Variables to compare
        report : bool
            True to display report, False not.
        paired : bool
            False for two independent variables. True Otherwise
            
        Returns:
        -------
        res : dict
            'Statistic': statistic value calculated by the test
            'Pvalue': p-value value calculated by the test
            'Test': name of the test used
            'Result': True if failed to reject null hypothesis, False otherwise
            
        Notes:
        -----
            None '''
        
        if len(x) == 1: # Data verification
            raise ValueError('\n>>> Error: Input dadta with two variables!\n')
        else:
            if report:
                print('\n>>> Comparing Two Variables...\n')
                
            skews = [ss.skew(i) for i in x]
            norm_res = self.norm_test(x, False)
            homo_res = self.homo_test(x, report)['Result']
            res = self.two_cal(x, norm_res, homo_res, skews, paired)
            
            if report:  
                self.two_report(x, res)
                print('\n>>> Two-Variable Comparison Test Done!')
                print('='*60)
                
            return res
    
    def welch_anova(self, x):
        '''Conduct Welch's ANOVA and resturn its f statistic and p-value.
        
        Parameters:
        ----------
        x : list of numpy.ndarrays
            Groups to conduct Welch's ANOVA on

        Returns:
        -------
        A dict of results for Welch's ANOVA. 
            'f-statistic' : f statistic value computed from the test
            'Pvalue' : p-value of the test
            
        Notes:
        -----
            None '''
        
        sizes = [len(i) for i in x]
        means = [np.mean(i) for i in x]
        vrs = [np.var(i, ddof=1) for i in x]

        wghts = [sizes[i] / vrs[i] for i in range(len(x))]
        sum_w = sum(wghts)
        means_p = [wghts[i] * means[i] for i in range(len(x))]
        sum_means_p = sum(means_p)

        a = [wghts[i] * (means[i] - sum_means_p / sum_w)**2 for i in range(len(x))]
        b = [(1 - wghts[i] / sum_w)**2 / (sizes[i] - 1) for i in range(len(x))]
        k = len(x)
        sum_a = sum(a)
        sum_b = sum(b)

        F = sum_a / (k - 1) / (1 + 2 * sum_b * (k - 2) / (k**2 - 1))
        df1 = k - 1
        df2 = (k**2 - 1) / 3 / sum_b

        Pvalue = ss.f.sf(F, df1, df2)

        return {'F-statistic': F, 'Pvalue': Pvalue}
    
    def rm_anova(self, x):
        '''Conduct Repeated Measures One-way ANOVA and resturn its 
        f statistic and p-value.
        
        Parameters:
        ----------
        x : list of numpy.ndarrays
            Groups to conduct Repeated Measures One-way ANOVA on

        Returns:
        -------
        A dict of results for Repeated Measures One-way ANOVA.
            'f-statistic' : f statistic value computed from the test
            'Pvalue' : p-value of the test
            
        Notes:
        -----
            None '''
        
        k = len(x)
        n = len(x[0])
        N = k * n
        
        df_bt = k - 1
        df_wi = N - k
        df_sb = n - 1
                
        sum_row = [np.sum([x[i][j] for i in range(k)]) for j in range(n)]
        sum_col = [np.sum(i) for i in x]
        sum_x2 = np.sum([(x[i][j])**2 for i in range(k) for j in range(n)])
        G = np.sum(sum_col)
        
        ss_nm = np.sum([sum_col[i]**2 / n for i in range(k)]) - G**2 / N
        ss_dn = sum_x2 + G**2 / N \
                - np.sum([sum_col[i]**2 / n for i in range(k)]) \
                - np.sum([sum_row[i]**2 / k for i in range(n)])
        
        ms_nm = ss_nm / df_bt
        ms_dn = ss_dn / df_bt / df_sb
        
        F = ms_nm / ms_dn
        Pvalue = ss.f.sf(F, df_bt, df_bt * df_sb)
        
        return {'F-statistic': F, 'Pvalue': Pvalue}
        
        
        
            
    def multi_cal(self, x, norm_res, homo_res, skews, repeated=False):
        '''Calculate and return multiple samples comparison tests results.
        
        Parameters:
        ----------
        x : list of numpy.ndarray
            Variables to test on
        norm_res : dict
            Normality test results
        homo_res : dict
            Homogeneity test rerults
        skews : list
            Skewness values of all x variables
        repeated : bool
            False for two independent variables. True Otherwise
            
        Returns:
        -------
        res : dict
            'Statistic': statistic value calculated by the test
            'Pvalue': p-value value calculated by the test
            'Test': name of the test used
            'Result': True if failed to reject null hypothesis, False otherwise
            
            
        Notes:
        -----
        None'''
        
        res = {}
        
        if sum(norm_res.values()) == len(x) and all(abs(np.array(skews)) < .5): 
        # If all normal use ANOVA
            if repeated: # For repeated measures
                res['Statistic'] = self.rm_anova(x)['F-statistic']
                res['Pvalue'] = self.rm_anova(x)['Pvalue']
                res['Test'] = 'Repeated Measures One-way ANOVA'
            else: 
                if homo_res: # Equal variances
                    res['Statistic'] = ss.f_oneway(*x)[0]
                    res['Pvalue'] = ss.f_oneway(*x)[1]
                    res['Test'] = 'One-Way ANOVA'
                else: # Unequal variances
                    res['Statistic'] = self.welch_anova(x)['F-statistic']
                    res['Pvalue'] = self.welch_anova(x)['Pvalue']
                    res['Test'] = 'Welch\'s ANOVA'
        else: # Nonparametric
            if repeated: # For repeated measures
                res['Statistic'] = ss.friedmanchisquare(*x)[0]
                res['Pvalue'] = ss.friedmanchisquare(*x)[1]
                res['Test'] = 'Friedman Test'
            else:# For independent groups
                res['Statistic'] = ss.kruskal(*x)[0]
                res['Pvalue'] = ss.kruskal(*x)[1]
                res['Test'] = 'Kruskal Wallis H-test'  
            
        if res['Pvalue'] >= .05:
            res['Result'] = True
        else:
            res['Result'] = False
            
        return res
    
    def multi_report(self, x, res):
        
        '''Display statistic values of two samples comparison tests.
        
        Parameters:
        ----------
        x : list of numpy.ndarray
            Variables tested on
        res : dict
            Results of multiple comparison test
            
        Returns:
        -------
        None
        
        Notes:
        -----
        None'''
        
        if 'ANOVA' in res['Test']: # Parametric makes hypothesis about means
            for i in range(len(x)):
                print('Mean {}:\t'.format(i+1), x[i].mean())
            print('\nTesting Method:\t', res['Test'])
            print('Statistic:\t', res['Statistic'])
            print('Pvalue: \t', res['Pvalue'])
            if res['Result']:
                print('Test Result:\t', 'Failed to reject! Means equal!\n')
            else:
                print('Test Result:\t', 'Reject! Means unequal!\n')
        else: # Unparametric makes hypothesis about medians
            for i in range(len(x)):
                print('Median {}:\t'.format(i+1), x[i].mean())
            print('\nTesting Method:\t', res['Test'])
            print('Statistic:\t', res['Statistic'])
            print('Pvalue: \t', res['Pvalue'])
            if res['Result']:
                print('Test Result:\t', 'Failed to reject! Medians equal!')
            else:
                print('Test Result:\t', 'Reject! Medians unequal!')
    
    def compr_multi(self, x, repeated=False, report=True):
        '''Conduct and return multiple comparision tests.
        
        Parameters:
        ----------
        x : list of numpy.ndarrays
            Variables to compare
        repeated : bool
            False for independent groups. True Otherwise
        report : bool
            True to display report, False not.
            
        Returns:
        -------
        res : dict
            'Statistic': statistic value calculated by the test
            'Pvalue': p-value value calculated by the test
            'Test': name of the test used
            'Result': True if homogeneous, False otherwise
            
        Notes:
        -----
            None '''
        
        if len(x) < 3: # Data verification
            raise ValueError('\n>>> Error: Input dadta with at leat three groups!\n')
        else:
            if report:
                print('\n>>> Comparing groups...\n'.format(len(x)))
                
        norm_res = self.norm_test(x, False)
        homo_res = self.homo_test(x, report)['Result']
        skews = [ss.skew(i) for i in x]
        res = self.multi_cal(x, norm_res, homo_res, skews, repeated)
            
        if report:  
            self.multi_report(x, res)
            print('\n>>> Multiple Comparison Done!')
            print('='*60)
        
        return res
            
