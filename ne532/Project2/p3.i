ne532 project 2
c Cell Cards
c ================================================================
c Material   Density      Surfaces        Data
c Aluminum casing on PVT
1 2          -2.7         4 -5            imp:p=10
c Aluminum casing on NaI detector
2 2          -2.7         2 -3            imp:p=10
c PVT
3 4          -1.032       -4              imp:p=10
c NaI detector
4 3          -3.67        -2              imp:p=10
c Soil
5 1          -1.82        -1              imp:p=1
c Air
6 5          -0.001225    -6 5 3 1        imp:p=1
c The unending eternal void that awaits us all
7 0                       6               imp:p=0

c Surface Cards
c ================================================================
c Soil
c       xmin     xmax     ymin     ymax     zmin     zmax
1 rpp   -70      70       -30      0        -70      70
c ----------------------------------------------------------------
c NaI Detector
c       vx       vy       vz       hx       hy       hz     r
2 rcc   0        100.1    0        0        7.62     0      3.81
c NaI Detector Casing
3 rcc   0        100.0    0        0        7.82     0      3.91
c ----------------------------------------------------------------
c PVT
c       xmin     xmax     ymin     ymax     zmin     zmax
4 rpp   50       55.08    0.1      182.98   -30.48   30.48
c PVT Casing
c       xmin     xmax     ymin     ymax     zmin     zmax
5 rpp   49.9     55.18    0        183.08   -30.58   30.58
c ----------------------------------------------------------------
c The world
c       xmin     xmax     ymin     ymax     zmin     zmax
6 rpp   -75      75       -31      200      -75      75

c ================================================================
c Material Cards
c ----------------------------------------------------------------
c name: Soil
c density: 1.82 g/cm^3
m1
     8016       -46.10
     14028      -28.20
     13027       -8.23
     26056       -5.63
     20040       -4.15
     11023       -2.36
     12024       -2.33
     19039       -2.09
c ----------------------------------------------------------------
c name: Aluminum
c density: 2.80 g/cm^3
m2
     13027       1.00
c ----------------------------------------------------------------
c name: NaI
m3
     11023       1.000
     53127       1.000
c ----------------------------------------------------------------
c name: PVT
m4
     1001        0.085
     6012        0.915
c ----------------------------------------------------------------
c name: Air (dry, near sea level)
c density = 0.001225
m5
     6012 -1.2256e-04
     6013 -1.4365e-06
     7014 -7.5232e-01
     7015 -2.9442e-03
     8016 -2.3115e-01
     8017 -9.3580e-05
     18036 -3.8527e-05
     18038 -7.6673e-06
     18040 -1.2781e-02
c ================================================================
c Source Definition
c ----------------------------------------------------------------
c Volume source throughout soil block
sdef x=d2 y=d3 z=d4 wgt=1 erg=d1 par=p cel=5
si1
     L
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
c   -x       +x
si2 -71      71
sp2 0        1
c   -y       +y
si3 -31      1
sp3 0        1
c   -z       +z
si4 -71      71
sp4 0        1
c ================================================================
c Tally Ho!
c ----------------------------------------------------------------
c some sort of f8 tally?
c like, measuring pulse height?
f18:p 4 $ NaI detector
ft18 geb     0     0.080     0
e18 0 1e-05 0.02 148i 3.0
fq18 e f
f28:p 3 $ PVT
ft28 geb     0     0.114     0
e28 0 1e-05 0.02 148i 3.0
fq28 e f
c Tally to control for error/stop condition
f218:p 4 $ NaI detector
ft218 geb    0     0.080     0
e218         2.76            2.78
c ================================================================
c Physics!
c ----------------------------------------------------------------
mode p
c Cutoff energies below 10keV
cut:p j 0.010 3j
c ================================================================
c Stop condition
c ----------------------------------------------------------------
nps 1e14
c stop f18 0.10
c stop f28 0.10
stop f218 0.10
c ================================================================
c Print
c ----------------------------------------------------------------
print
