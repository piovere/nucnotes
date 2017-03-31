ne532 Project 3 Background + Source
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
sdef erg=d1 x=ferg=d2 y=ferg=d3 z=ferg=d4 wgt=80994460 par=p
c Photons are from source or soil
si1  s     5       6
sp1        1e6     79994460
ds2  s     11      21
ds3  s     12      22
ds4  s     13      23
c Cf-252 Source
si11 l     -350
sp11       1
si12 l     120
sp12       1
si13 l     400
sp13       1
c Energy distribution for Cf-252
SI5  L  0.01    0.05 0.1  0.15 0.2  0.25 0.3  0.35
        0.4     0.45 0.5  0.55 0.6  0.65    0.7    0.75
        0.8     0.85 0.9  0.95 1    1.05 1.1  1.15
        1.2     1.25 1.3  1.35   1.4     1.45 1.5  1.55
        1.6     1.65 1.7  1.75 1.8  1.85 1.9  1.95
        2       2.05     2.1  2.15 2.2  2.25 2.3  2.35
        2.4     2.45 2.5  2.55 2.6  2.65 2.7  2.75
        2.8     2.85 2.9  2.95 3    3.05 3.1  3.15
        3.2     3.25 3.3  3.35 3.4  3.45    3.5    3.55
        3.6     3.65 3.7  3.75 3.8  3.85 3.9  3.95
        4       4.05 4.1  4.15    4.2    4.25 4.3  4.35
        4.4     4.45 4.5  4.55 4.6  4.65 4.7  4.75
        4.8     4.85    4.9    4.95 5    5.05 5.1  5.15
        5.2     5.25 5.3  5.35 5.4  5.45 5.5  5.55
        5.6     5.65 5.7  5.75 5.8  5.85 5.9  5.95
        6       6.05 6.1  6.15 6.2  6.25    6.3    6.35
        6.4     6.45 6.5  6.55 6.6  6.65 6.7  6.75
        6.8     6.85 6.9  6.95      7    7.05 7.1  7.15
        7.2     7.25 7.3  7.35 7.4  7.45 7.5  7.55
        7.6     7.65    7.7    7.75 7.8  7.85 7.9  7.95
        8       8.05 8.1  8.15 8.2  8.25 8.3  8.35
        8.4     8.45 8.5  8.55 8.6  8.65 8.7  8.75
        8.8     8.85 8.9  8.95 9    9.05    9.1    9.15
        9.2     9.25 9.3  9.35 9.4  9.45 9.5  9.55
        9.6     9.65 9.7  9.75    9.8    9.85 9.9  9.95
        10     10.05     10.1 10.15     10.2 10.25
        10.3   10.35     10.4 10.45     10.5
SP5 D   6.6  6.6  6.6  6.6  6.6  6.6  6.6  6.6
        6.6    6.6 6.6   6.6
        6.942643463 6.351446641
        5.810592845 5.315795144
        4.863131658 4.449014471
        4.070161195 3.723568953
        3.406490575 3.116412824
        2.851036477 2.608258101
        2.386153379 2.182961856
        1.997072992 1.827013386
        1.671435109 1.529105011
        1.403657073 1.329205009
        1.258701993 1.191938563
        1.128716364 1.068847565
        1.012154296 0.958468123
        0.907629545 0.85948752
        0.813899021 0.770728603
        0.729848009 0.691135782
        0.654476909 0.619762478
        0.586889352 0.555759866
        0.526281534 0.498366777
        0.47193266  0.446900648
        0.423196371 0.400749404
        0.379493058 0.359364181
        0.340302969 0.322252792
        0.305160023 0.28897388
        0.273646275 0.259131668
        0.245386938 0.232371249
        0.220045932 0.208374367
        0.19732188  0.186855633
        0.176944532 0.16755913
        0.158671544 0.150255369
        0.1422856   0.13473856
        0.127591825 0.120824165
        0.114415471 0.108346704
        0.102599833 0.097157785
        0.092004391 0.087124341
        0.082503136 0.078127047
        0.073983072 0.070058899
        0.066342871 0.062823946
        0.05949167  0.056336143
        0.05334799  0.050518333
        0.047838766 0.045301326
        0.042898476 0.040623077
        0.038468368 0.036427949
        0.034495756 0.03266605
        0.030933394 0.02929264
        0.027738915 0.026267602
        0.024874329 0.023554957
        0.022305567 0.021122447
        0.020002081 0.018941141
        0.017936474 0.016985097
        0.016084182 0.015231053
        0.014423175 0.013658149
        0.0129337   0.012247677
        0.011598042 0.010982865
        0.010400317 0.009848669
        0.009326281 0.008831601
        0.00836316  0.007919565
        0.0074995   0.007101715
        0.006725029 0.006368324
        0.006030538 0.00571067
        0.005407767 0.005120931
        0.004849309 0.004592095
        0.004348523 0.004117871
        0.003899453 0.00369262
        0.003496758 0.003311285
        0.003135649 0.00296933
        0.002811832 0.002662688
        0.002521455 0.002387714
        0.002261066 0.002141136
        0.002027566 0.001920021
        0.001818181 0.001721742
        0.001630418 0.001543938
        0.001462045 0.001384496
        0.00131106  0.00124152
        0.001175668 0.001113309
        0.001054257 0.000998338
        0.000945384 0.00089524
        0.000847755 0.000802789
        0.000760208 0.000719885
        0.000681701 0.000645543
        0.000611302 0.000578878
        0.000548173 0.000519097
        0.000491564 0.00046549
        0.0004408   0.000417419
        0.000395279 0.000374313
        0.000354459 0.000335658
        0.000317854 0.000300994
        0.000285029 0.000269911
        0.000255594 0.000242037
        0.000229199 0.000217042
        0.00020553  0.000194628
        0.000184305 0.000174529
        0.000165272 0.000156506
        0.000148204 0.000140343
        0.000132899 0.00012585
        0.000119175 0.000112854
        0.000106868 0.000101199
        9.58315E-05 9.07485E-05
        8.5935E-05  8.13769E-05
        7.70606E-05
c soil energy distribution
si6
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
sp6
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
c    -x       +x
si21 -349.9   349.4
sp21 0        1
c    -y       +y
si22 -29.9    -0.1
sp22 0        1
c    -z       +z
si23 -349.9   349.9
sp23 0        1
c ================================================================
c Tally Ho!
c ----------------------------------------------------------------
f8:p 2
e8 0 1e-5 12
c Cutoff energies below 50keV
cut:p j 0.050 3j
c ================================================================
c Apocrypha
c ----------------------------------------------------------------
RAND SEED=19073486328125
nps 1e7
mode p
print