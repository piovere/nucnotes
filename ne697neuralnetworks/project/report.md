# Abstract

# Introduction

- Introduction to LIBS

Laser Induced Breakdown Spectroscopy is a chemical analysis technique for 
examining samples in a gaseous, liquid, or solid state. A high-intensity 
laser pulse is used to ablate a small amount of a sample of interest, 
producing a plasma. After a short period, the constituents of the plasma 
cool enough to recombine into new compounds, emitting energy as they do so. 


- Explanation of handheld LIBS



- Spectral analysis
- Description of the RBF

\begin{equation}
    RBF \left(\vec{x}, \vec{\phi} \right) = \exp\left( -\beta \left| 
        \vec{\phi} - \vec{x} \right|^2 \right)
    \label{eq:rbf}
\end{equation}

- Description of PNN
- Description of GRNN
- Goals

# Methodology

11 samples of simulated meltglass were prepared. All but one sample consisted 
of a common base (consisting of SiO2, FeO3, and Na?????) plus a single 
additional analyte added (?????LIST OF ANALYTES?????). One sample was 
prepared without the addition of any analyte. These samples were melted in a 
furnace and rapidly cooled to produced melt glass beads. Each of the 11 beads 
was ablated with a benchtop LIBS instrument at 20 different sites on the 
surface of the bead, with a 11,725 channel spectrum collected from each site. 
This produced a dataset of 220 records of 11,725 values. These records were 
segmented into 154 training and 66 validation sets. The validation sets were 
selected by randomly choosing 6 records from each of the 11 samples in order 
to ensure that the validation set was representative. Each of the records was 
matched to two different output tags--one representing which analyte was 
added to the sample (or none), the other with the mass of each chemical 
constituent of the sample. The records were normalized to scale each of the 
input records in the range $\left[0,1\right]$, while the output values were 
converted into one-hot vectors (for the classification network) and 13 
element vectors with each value in the range range $\left[0,1\right]$ (one 
value for each of the 10 analytes plus 3 for the base recipe). 

## Classification

In order to determine the effectiveness of the RBF for classification, first 
the records from the test set were grouped by analyte and averaged. Each of 
these 11 average spectra were used as a reference vector $\phi$ for an RBF 
function. The value of this function was calcualted for each $\phi$ and 
record in the test and validation set, with the highest valued RBF considered 
the classification of the record. 

A probabilistic neural network was then generated. It was trained with one 
neuron for each of the 154 samples in the training set. In order to determine 
the optimal width for the neurons, 

# Results

# Conclusions

## Future work

- Per-neuron width?
- Convolutional layers
- Normalized output
- Multi-class classification/confidence