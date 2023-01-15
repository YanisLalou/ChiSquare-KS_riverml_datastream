import numpy as np
from river import sketch, stats
import random

class KSTest(stats.base.Univariate):
    """Running the Kolmogorov-Smirnov Test.

    Uses multiple algorithms to compute the Kolmogorov-Smirnov test in a running method.
    2 different algorithms are available to compute the Kolmogorov-Smirnov test.
    Each of them compare a distribution to a uniform distribution.

    Parameters
    ----------
    K
        Number of bins used to approximate our distribution

    Examples
    --------

    >>> from stats.Kolmogorov_Smirnov_test import KSTest
    >>> import random
    >>> import import scipy
    >>> from scipy.stats import ksone
    >>> import numpy as np

    >>> random.seed(0)
    >>> N = 1000
    >>> X = [random.uniform(0, 1) for _ in range(N)]
    >>> K = 30

    >>> cdf = [random.uniform(0, 1) for _ in range(N)] #The Distribution to compare to, here a uniform distrib
    >>> kolmogorov_Smirnov_test_object = KSTest(K, cdf) # The KSTest object implemented

    >>> for x in X:
    ...    kolmogorov_Smirnov_test_object.update(x)
    >>> print(f'The estimated value of the Kolmogorov-Smirnov test is {kolmogorov_Smirnov_test_object.get():.4f}')
    The estimated value of the Kolmogorov-Smirnov test is 0.0210

    >>> print(f'The real value of the chi square test is {scipy.stats.kstest(X[:i], cdf)[0]:.4f}')
    The real value of the chi square test is 0.0350

    References
    ----------
    [^1]: [Data Streaming Algorithms for the Kolmogorov-Smirnov Test](http://personal.denison.edu/~lalla/papers/ks-stream.pdf)
    
    """

    def __init__(self, K: int = 30, y: list = []):
        super().__init__()
        self._is_updated = False

        self.K = K
        self.N = 0 #Number of sample seen

        self.quantiles_KS = self.__define_quantile_ks__()
        self.river_histogram = sketch.Histogram(max_bins=self.K)

        #We know in advance the distribution to compare to, so we can make a sketch of it
        self.histogram_expected = sketch.Histogram(max_bins=self.K) #for methode2

        #By default we compare our distribution to the uniform one on [0, 1]
        if len(y) == 0:
            y = [random.uniform(0, 1) for _ in range(1000)] 


        for i, y_value in enumerate(y):
            self.histogram_expected.update(y_value)



    def update(self, x):
        self._is_updated = True

        self.N +=1

        self.__update_quantile_ks__(x)
        self.river_histogram.update(x)

        return self

    def get(self, methode_name: str = ''):
        """
        Retrieve the current value of the Kolmogorov-Smirnov test

        :param methode_name: Select from which algorithm we retrieve the value
        :return: The value of the Kolmogorov-Smirnov test
        """

        if not self._is_updated:
            return None

        match methode_name:
            case "quantile":
                return self.__get_quantile_ks__()
            case "histogram":
                #return self.__get_histogram_ks__()
                pass
            case default:
                #By default, quantile method
                return self.__get_quantile_ks__()


    #For quantile method KS
    def __define_quantile_ks__(self):
        quantile_list = []

        for i in range(1, self.K):
            quantile_list.append(stats.Quantile(i/self.K))

        return quantile_list

    #For quantile method KS
    def __update_quantile_ks__(self, x):
        for quantile in self.quantiles_KS:
            quantile.update(x)

    #For quantile method KS
    #[^1]: [Data Streaming Algorithms for the Kolmogorov-Smirnov Test](http://personal.denison.edu/~lalla/papers/ks-stream.pdf)
    def __get_quantile_ks__(self):
        # Initialize the ks
        ks = 0
        
        quantile_length = len(self.quantiles_KS) + 2 # + 2 for Q0 and Q1
        
        for i, quantile in enumerate(self.quantiles_KS, start=1):
            x = quantile.get()
            
            i_j = int(self.N*i/quantile_length)
            
            E_x = np.abs(i_j/self.N - self.histogram_expected.cdf(x)) #The Paper formula
            #E_x = np.maximum(self.histogram_expected.cdf(x) - ((i_j-1)/N), i_j/N - self.histogram_expected.cdf(x)) #The real Kolmogorov-Smirnov test formula
            
            ks = np.maximum(ks, E_x)
            
        return ks

    # TO MODIFY
    def __get_histogram_ks__(self):
        D_hat = 0
        nb_tot = np.sum([self.histogram_expected[i].count for i in range (len(self.histogram_expected))])

        for p in range(len(self.histogram_expected)):
            proba=np.sum([self.histogram_expected[i].count for i in range (p+1)])/nb_tot
            proba_exp=np.sum([self.river_histogram[i].count for i in range (p+1)])/nb_tot
            E_hat=np.abs(proba_exp-proba)
            D_hat=D_hat if D_hat > E_hat else E_hat
            
        return D_hat