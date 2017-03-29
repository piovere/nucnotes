ne532 Project 3 Background
c ================================================================
c Cell Cards
c ----------------------------------------------------------------
c Material   Density      Surfaces        Data
c Soil
1 1          -1.82        -1              imp:p=1
c NaI
2 2          -3.67        -2              imp:p=1
c Aluminum casing on NaI detector
3 3          -2.70        2 -3            imp:p=1
c Air
4 4          -0.001225    -4 3 1          imp:p=1
c Void
5 0                       4               imp:p=0

c ================================================================
c Surface Cards
c ----------------------------------------------------------------
c Soil
c       xmin     xmax     ymin     ymax     zmin     zmax
1 rpp   -350     350      -30      0        -350     350
c ----------------------------------------------------------------
c NaI Detector, a cylinder with radius 3.81cm and height 7.62cm
c       vx       vy       vz       hx       hy       hz     r
2 rcc   0        110.1    0        0        7.62     0      3.81
c NaI Detector Casing, 1mm of Al around the detector
c     the bottom of which is 110cm above the ground
3 rcc   0        110.0    0        0        7.82     0      3.91
c ----------------------------------------------------------------
c The World, a sphere centered at (0,0,0) with radius 7m
c       x        y        z        r
4 sph   0        0        0        700

c ================================================================
c Material Cards
c ----------------------------------------------------------------
c name: Soil
c density = 1.82 g/cm^3
m1
     8000       -46.10
     14000      -28.20
     13000       -8.23
     26000       -5.63
     20000       -4.15
     11000       -2.36
     12000       -2.33
     19000       -2.09
     55137       -2.301541e-13
     90232       -8.626218e-04
     92235       -6.014194e-11
     92238       -8.531564e-09
c ----------------------------------------------------------------
c name: Sodium Iodide
c density = 3.67 g/cm^3
m2
     11000 1.000
     53000 1.000
c ----------------------------------------------------------------
c name: Aluminum casing
c density = 2.70 g/cm^3
m3
     13000 1.000
c ----------------------------------------------------------------
c name: Air (dry, near sea level)
c density = 0.001225 g/cm^3
m4
     6012 -1.2256e-04
     6013 -1.4365e-06
     7014 -7.5232e-01
     7015 -2.9442e-03
     8016 -2.3115e-01
     8017 -9.3580e-05
     8018 -5.3454e-04
     18036 -3.8527e-05
     18038 -7.6673e-06
     18040 -1.2781e-02
c ================================================================
c Source Definition
c ----------------------------------------------------------------
sdef erg=d1 x=d2 y=d3 z=d4 wgt=79994460 par=p cel=1
c soil energy distribution
si1
     l
     0.241
     0.24192
     0.2703
     0.27736
     0.29522
     0.30009
     0.328
     0.3384
     0.35199
     0.4094
     0.463
     0.51072
     0.5623
     0.58314
     0.60932
     0.66162
     0.66545
     0.727
     0.72717
     0.7552
     0.7633
     0.76836
     0.7721
     0.782
     0.78542
     0.78595
     0.7948
     0.80617
     0.8304
     0.8356
     0.8402
     0.86047
     0.89339
     0.9042
     0.91107
     0.93405
     0.9646
     0.9689
     1.07862
     1.12028
     1.15519
     1.23811
     1.2471
     1.28096
     1.37765
     1.38531
     1.4015
     1.40798
     1.4592
     1.46075
     1.4958
     1.5015
     1.50919
     1.51275
     1.5802
     1.5879
     1.62056
     1.6304
     1.66128
     1.7296
     1.76451
     1.84744
     2.11854
     2.20412
     2.44771
     2.61447
sp1
     D
     1.441
     4.086
     1.393
     0.8632
     10.502
     1.208
     1.243
     4.438
     20.3
     0.8251
     1.714
     2.988
     0.3643
     11.42
     25.22
     6.262
     0.855
     0.2947
     4.36
     0.4072
     0.2258
     2.672
     0.6001
     0.2015
     1.478
     0.5962
     1.789
     0.6718
     0.2282
     0.6729
     0.3643
     1.594
     0.2439
     0.3215
     10.72
     1.7312
     2.015
     6.451
     0.3584
     8.226
     0.9244
     3.236
     0.209
     0.8062
     2.198
     0.4246
     0.7582
     1.355
     0.7692
     39.24
     0.388
     0.2143
     1.199
     0.2106
     0.2625
     1.372
     1.016
     0.7202
     0.629
     1.6668
     8.708
     1.1612
     0.6614
     2.732
     0.8494
     13.28
c ----------------------------------------------------------------
c Sources inside of the soil block
c   -x       +x
si2 -351     351
sp2 0        1
c   -y       +y
si3 -31      1
sp3 0        1
c   -z       +z
si4 -351     351
sp4 0        1
c ================================================================
c Tally Ho!
c ----------------------------------------------------------------
f8:p 2
e8 0 1e-5 3
c Cutoff energies below 50keV
cut:p j 0.050 3j
c ================================================================
c Apocrypha
c ----------------------------------------------------------------
RAND SEED=238919401
nps 1e7
mode p
print
