# Abstract

# Introduction
Genetic algorithms are an approach to optimization that takes its inspiration from biology. Rather than calculating a gradient and using that information to iteratively move towards a minimum value on an error surface, the fitness of several 

Genetic algorithms start by encoding a population of candidate solutions. Population size can vary based on the availability of computational resources (larger populations take longer to converge) vice desired accuracy (provided by larger populations) [Rylander 2002]. 

Genetic algorithms can be employed in two ways--either by directly calculating 

Ways to change the genes

Explanation of Elite Selection

Explanation of Crossover fraction


# Methodology

## Scale the data
The data set examined consisted of 5000 records of forty-three input values used to predict a single output value. First the data was imported and split into training (2000 records), testing (1500 records), and validation sets (1500 records). The data were scaled such that the training set had a zero mean and unit variance for each variable. 

Candidate alpha values were selected in the range [2.89, 164]. These values were selected based on a single value decomposition of the scaled training data set.

To provide a reference to benchmark the *Explanation *

Discrete or continuous variable selection

How many bits to include

How to scale continuous value to number of bits

Explanation of size of population

Explanation of size of elite fraction

Explanation of crossover
# Results

# Conclusions