ne532 project 2
c Cell Cards
c ================================
c Material   Density      Surfaces        Data
c Aluminum casing on PVT
1 1          -2.7
c Aluminum casing on NaI detector
2 1          -2.7
c PVT
3 
c NaI detector
4 
c Soil
5
c Air
6
c The unending eternal void that awaits us all
7

c Surface Cards
c ================================
c Soil
1 rpp
c NaI Detector
2 rcc
c NaI Detector Casing
3 rcc
c PVT
4 rpp
c PVT Casing
5 rpp
c The world
c     x       y       z       r
6 sph 0       0       0       300

c ================================
c Material Cards
c ================================
c name: Soil
c density: 1.82 g/cm^3
m1
     8016       46.10
     14028      28.20
     13026       8.23
     26056       5.63
     20040       4.15
     11023       2.36
     12024       2.33
     19039       2.09
c name: Aluminum
c density: 2.80 g/cm63
m2
     13027       1.00
c name: NaI
m3
     blah
c name: PVT
m4
     blah
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
c ================================
c Source Definition
c ================================
c Volume source throughout soil block