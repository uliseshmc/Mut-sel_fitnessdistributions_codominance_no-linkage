#libraries
import numpy as np
import math
from scipy.stats import poisson

#List of functions
def get_min_max_poisson_sample(xmean, samplesize):
  """
  Generates a sample of a Poisson distribution. and spits the minimum value for this sample
  """
  sample = np.random.poisson(lam=xmean, size=samplesize)
  return np.min(sample), np.max(sample)

def get_expected_min_max_poisson_sample(xmean, samplesize, iterations):
  """
  Generates the expected minimum and standard error of iterations samples of Poisson distribution of size samplesize.
  """
  array_minx = []
  array_maxx = []
  for i in range(iterations):
    resultmin, resultmax = get_min_max_poisson_sample(xmean, samplesize)
    array_minx.append(resultmin)
    array_maxx.append(resultmax)

  mean_array_minx = np.mean(array_minx)
  std_dev_array_minx = np.std(array_minx)
  std_error_array_minx = std_dev_array_minx / np.sqrt(iterations)

  mean_array_maxx = np.mean(array_maxx)
  std_dev_array_maxx = np.std(array_maxx)
  std_error_array_maxx = std_dev_array_maxx / np.sqrt(iterations)

  return mean_array_minx, std_error_array_minx, mean_array_maxx, std_error_array_maxx

vget_median_min_max_poisson_sample = np.vectorize(get_expected_min_max_poisson_sample)

def get_median_min_max_poisson_sample(xmean, samplesize, iterations):
  """
  Generates the expected minimum and standard error of iterations samples of Poisson distribution of size samplesize.
  """
  array_minx = []
  array_maxx = []
  for i in range(iterations):
    resultmin, resultmax = get_min_max_poisson_sample(xmean, samplesize)
    array_minx.append(resultmin)
    array_maxx.append(resultmax)

  median_array_minx = np.median(array_minx)

  median_array_maxx = np.median(array_maxx)

  return median_array_minx, median_array_maxx

vget_median_min_max_poisson_sample = np.vectorize(get_median_min_max_poisson_sample)


def get_absolute_fitness(segregating_alleles, sd, Ud):
  """
  The model assumes that relative fitness is w=(1-s)^k. There is no epistasis and no linkage disequilibrium. 
  Then the distribution of number of mutations per individual follows a Poisson distribution with mean xmean.
  The best individual realtive fitness is the fitness of the individual with the expected minimum number of segregating mutations.
  We calculate this minimum number of segregating mutations numerically by using random samples from the Poisson distribution with size N.
  To calculate absolute fitness, we divide the best individual relative fitness by the mean relative fitness of the population.
  The mean relative fitness of the population is e^(-Ud)
  """
  relative_fitness = (1-sd)**(segregating_alleles)
  mean_relative_fitness = np.exp(-Ud)
  absolute_fitness = relative_fitness / mean_relative_fitness
  return absolute_fitness

vget_absolute_fitness = np.vectorize(get_absolute_fitness)

def get_interquantile_folddifference(xmean, sd):
  """
  The model assumes that relative fitness is w=(1-s)^k. There is no epistasis and no linkage disequilibrium. 
  Then the distribution of number of mutations per individual follows a Poisson distribution with mean xmean.
  The interquantile fold difference is the ratio of the 1st quantile individual over the 3rd quantile individual.
  """
  first_quantile_individual = poisson.ppf(0.25, xmean)
  fitness_first_quantile = (1-sd)**(first_quantile_individual)
  
  third_quantile_individual = poisson.ppf(0.75, xmean)
  fitness_third_quantile = (1-sd)**(third_quantile_individual)
  
  fold_difference_fitness = fitness_first_quantile / fitness_third_quantile
  return fold_difference_fitness

vget_interquantile_folddifference = np.vectorize(get_interquantile_folddifference)

