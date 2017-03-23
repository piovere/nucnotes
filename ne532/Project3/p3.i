ne532 Project 3
c ================================================================
c Cell Cards
c ----------------------------------------------------------------
c Material   Density      Surfaces        Data
c Soil
1 1          -1.82        -1              imp:p=1
c NaI
2 2          -3.67        -2              imp:p=10
c Aluminum casing on NaI detector
3 3          -2.70        2 -3            imp:p=10
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
c NaI Detector
c       vx       vy       vz       hx       hy       hz     r
2 rcc   0        110.1    0        0        7.62     0      3.81
c NaI Detector Casing
3 rcc   0        110.0    0        0        7.82     0      3.91
c ----------------------------------------------------------------
c The World
c       xmin     xmax     ymin     ymax     zmin     zmax
4 rpp   -351     351      -31      120      -351     351