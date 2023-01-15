import numpy as np
from river import sketch, stats

class ChiSquareTest(stats.base.Univariate):
    """Running Chi_Square Test.

    Uses multiple algorithms to compute the Chi square test in a running method.
    4 different algorithms are available to compute the Chi square test.
    Each of them compare a distribution to a uniform distribution.

    Parameters
    ----------
    K
        Number of bins used to approximate our distribution

    Examples
    --------

    >>> from stats.chi_square_test import ChiSquareTest
    >>> import random
    >>> import scipy.stats.chisquare as chisquare
    >>> import numpy as np

    >>> random.seed(0)
    >>> N = 1000
    >>> X = [random.uniform(0, 1) for _ in range(N)]
    >>> K = 10
    >>> numpy_histogram = []

    >>> chi_square_test_object = ChiSquareTest()
    >>> for x in X:
    ...    chi_square_test_object.update(x)
    >>> print(f'The estimated value of the chi square test is {chi_square_test_object.get():.4f}')
    The estimated value of the chi square test is 9.5211

    >>> numpy_histogram, _ = np.histogram(X, bins=K)
    >>> print(f'The real value of the chi square test is {scipy.stats.chisquare(numpy_histogram)[0]:.4f}')
    The real value of the chi square test is 9.8000

    References
    ----------
    [^1]: [Accessible Streaming Algorithms for the Chi-Square Test](http://personal.denison.edu/~lalla/papers/AccessibleStreamingAlgorithmsForTheChiSquareTest.pdf)
    
    """

    def __init__(self, K: int = 10):
        super().__init__()
        self._is_updated = False

        #K = 21 Paper says to use 20 bins for 1e7 data points
        self.K = K
        self.N = 0 #Number of sample seen

        self.quantiles1 = self.__define_quantile1__()
        self.quantiles2 = self.__define_quantile2__()
        self.river_histogram = sketch.Histogram(max_bins=self.K)


    def update(self, x):
        self._is_updated = True

        self.N +=1

        self.__update_quantile1__(x)
        self.__update_quantile2__(x)
        self.river_histogram.update(x)

        return self

    def get(self, methode_name: str = ''):
        """
        Retrieve the current value of the chi square test

        :param methode_name: Select from which algorithm we retrieve the value
        :return: The value of the chi square test
        """

        if not self._is_updated:
            return None

        match methode_name:
            case "quantile_1":
                return self.__get_quantile1__()
            case "quantile_2":
                return self.__get_quantile2__()
            case "paper":
                return self.__paper_method__()
            case "histogram":
                return self.__histogram_method__()
            case default:
                #By default, quantile 2 method
                return self.__get_quantile2__()


    #For quantile method 1
    def __define_quantile1__(self):
        quantiles = []
        for b in range(0,self.K):
            if ((b-1)/self.K < 1 and (b-1)/self.K > 0) and ((b)/self.K < 1 and (b)/self.K > 0):
                q_l = stats.Quantile((b-1)/self.K)
                q_u = stats.Quantile(b/self.K)

                quantiles.append([q_l, q_u])
        
        return quantiles

    #For quantile method 1
    def __update_quantile1__(self, x):
        for b in range(0,self.K):
            if ((b-1)/self.N < 1 and (b-1)/self.N > 0) and ((b)/self.N < 1 and (b)/self.N > 0) and (b>0) and (b<len(self.quantiles1)):
                self.quantiles1[b][0].update(x)
                self.quantiles1[b][1].update(x)

    #For quantile method 1
    def __get_quantile1__(self):
        # Initialize the chi2
        chi2 = 0

        # Calculate the degrees of freedom
        #expected frequency
        Ei = self.N/self.K
        
        # Iterate over the bins of the sample
        for b in range(2, self.K+2):

            if ((b-1)/self.N < 1 and (b-1)/self.N > 0) and ((b)/self.N < 1 and (b)/self.N > 0) and (b>0) and (b<len(self.quantiles1)) :

                p_l = self.quantiles1[b][0].get()
                p_u = self.quantiles1[b][1].get()

                i_l = self.river_histogram.cdf(p_l)
                i_u = self.river_histogram.cdf(p_u)

                Oi = (i_u-i_l)*self.N
                
                lambda_i = np.abs(Oi - Ei)

                chi2 += chi2 + ((lambda_i)**2) / Ei
    
        return chi2


    #For quantile method 2
    def __define_quantile2__(self):
        quantile_list = []

        for i in range(1, self.K):
            quantile_list.append(stats.Quantile(i/self.K))
        return quantile_list

    #For quantile method 2
    def __update_quantile2__(self, x):
        for quantile in self.quantiles2:
            quantile.update(x)

    #For quantile method 2
    def __get_quantile2__(self):
        x_min_quant = self.quantiles2[0].get()
        x_max_quant = self.quantiles2[-1].get()

        # Initialize the chi2
        chi2 = 0
        Ei = 0
        
        for i in range(0, len(self.quantiles2)-1):
            borne_inf = self.quantiles2[i].get()
            borne_sup = self.quantiles2[i+1].get()
            
            Oi = 1/(len(self.quantiles2)+2) #By definition, between 2 quantiles. We add 2 for Q0 and Q1
            
            if x_max_quant-x_min_quant !=0:
                
                ####
                # We are expecting to have an uniform distrib, that's why we can make the following assumptions:
                expected_quantile_size = (x_max_quant-x_min_quant)*(1/(len(self.quantiles2)-2))
                Q0 = x_min_quant - expected_quantile_size
                Q1 = x_max_quant + expected_quantile_size
                
                Ei = np.abs((borne_sup-borne_inf)/(Q1-Q0))
                ####

                if Ei !=0:
                    chi2 += ((((Oi - Ei)*self.N) ** 2) / (Ei*self.N))

        return chi2

    #For paper method:
    #[^1]: [Accessible Streaming Algorithms for the Chi-Square Test](http://personal.denison.edu/~lalla/papers/AccessibleStreamingAlgorithmsForTheChiSquareTest.pdf)

    def __paper_method__(self):
        # Initialize the chi2
        chi2 = 0

        # Iterate over the bins of the sample
        for i in range(1,self.K+1):
            
            Ei = self.N/self.K

            # Calculate the probability corresponding to the ith bin
            p_l = (i-1)/self.K
            #q_l = river.stats.Quantile(p_l)
            i_l = self.river_histogram.cdf(p_l) # Calculate the value x such that the CDF of the hypothesized distribution is equal to p
            
            p_u = (i)/self.K
            #q_u = river.stats.Quantile(p_u)
            i_u = self.river_histogram.cdf(p_u) # Calculate the value x such that the CDF of the hypothesized distribution is equal to p

            Oi = self.N*(i_u-i_l)
            lambda_i = np.abs(Oi - Ei)

            if lambda_i > 2*np.sqrt(self.N):
                chi2 += chi2 + ((lambda_i)**2) / Ei
                break

            chi2 += chi2 + ((lambda_i)**2) / Ei
    
        return chi2

    #For histogram method
    def __histogram_method__(self):
        histogram_width = self.river_histogram[-1].right - self.river_histogram[0].left

        chi2 = 0
        
        for i in range(len(self.river_histogram) - 1):
            bin_width = self.river_histogram[i + 1].left - self.river_histogram[i].left
            bin_relative_width = bin_width / histogram_width
            expected_count = bin_relative_width * self.river_histogram.n
            bin_count = self.river_histogram[i].count
            chi2 += ((bin_count - expected_count) ** 2) / expected_count
        
        return chi2