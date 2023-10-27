# CS 725 Course Project, Tiitle :
# Prediction of Ground Motion During a Seimsic Event using Neural Networks

## Members

1. 23D0292 / Anis M Vengasseri 
2. 22D1264 / Moh. Zaid
3. 23m2153 / Ashwin 
4. / Vishnu T.S.



## Abstract
Historically, seismic events have brought extensive destruction and distress to communities worldwide. These events have potential to generate economic losses of up to a staggering $200 billion (Japan, 1995). In the current year alone, over 50,000 lives were tragically lost due to earthquakes(Turkey Earthquake).The most devastating consequence of earthquakes is the ground vibration, which can inflict damage to infrastructure causing huge economic losses. Throughout history, the prediction of this ground motion has remained a challenge for seismology groups. Researchers have explored various empirical equations, commonly known as attenuation equations or Ground Motion Prediction Equations (GMPEs), to estimate ground motion to engineer communities towards more seismic resilience. These empirical relationships were always challenged by new events or information. Recent studies in this field have explored the use of neural networks for the prediction of ground motions. In this project, we intent on an exploration of the application of neural networks in predicting ground motions, aiming to assess their effectiveness in comparison to traditional empirical equations.

 Data Source https://ngawest2.berkeley.edu/

## Data

The earthquake ground motion data collected by Pacific Earthquake Engineering Research Center (PEER) for USA Western region is used for the present study.
The data can be downloaded from https://peer.berkeley.edu/research/data-sciences/databases. The sourced data are available in the data/ folder of the repository.

NGA-West2 Database Flatfile- The Updated NGA-West2 database “flatfiles” are posted. (January 17, 2015)

1. Updated NGA-West2 Flatfile of 5% damped spectra of vertical ground motion (.zip file, 48 MB)
2. Updated NGA-West2 Flatfile Part1 (.zip file, 290 MB)
3. Updated NGA-West2 Flatfile Part2 (.zip file, 240 MB

## Definition of Parameters	
1. Damping ratio	=  Viscous damping ratio (%) See Sanaz et al. (2012) PEER Report
2. PSA	=  Pseudo-absolute acceleration response spectrum (g)
3. PGA	=  Peak ground acceleration (g)
4. PGV	=  Peak ground velocity (cm/s)
5. Sd	=  Relative displacement response spectrum (cm)
6. Mw	=  Moment magnitude
7. RRUP	=  Closest distance to coseismic rupture (km), used in ASK13, CB13 and CY13. See Figures a, b and c for illustation
8. RJB	=  Closest distance to surface projection of coseismic rupture (km). See Figures a, b and c for illustation
9. RX	=  Horizontal distance from top of rupture measured perpendicular to fault strike (km). See Figures a, b and c for illustation
10. Ry0	=  The horizontal distance off the end of the rupture measured parallel to  strike (km)
11. VS30	= The average shear-wave velocity (m/s) over a subsurface depth of 30 m
12. U	=  Unspecified-mechanism factor:  1 for unspecified; 0 otherwise
13. FRV	=  Reverse-faulting factor:  0 for strike slip, normal, normal-oblique; 1 for reverse, reverse-oblique and thrust
14. FNM	=  Normal-faulting factor:  0 for strike slip, reverse, reverse-oblique, thrust and normal-oblique; 1 for normal
15. FHW	=  Hanging-wall factor:  1 for site on down-dip side of top of rupture; 0 otherwise
16. Dip	=  Average dip of rupture plane (degrees)
17. ZTOR	=  Depth to top of coseismic rupture (km)
18. ZHYP	=  Hypocentral depth from the earthquake
19. Z1.0 	= Depth to Vs=1 km/sec
20. Z2.5 	= Depth to Vs=2.5 km/sec
21. W	=  Fault rupture width (km)
22. Vs30flag	=  1 for measured, 0 for inferred Vs30
23. FAS	=   0 for mainshock; 1 for aftershock
24. Region	= Specific regions considered in the models, Click on Region to see codes
25. DDPP	=  Directivity term, direct point parameter; uses 0 for median predictions
26. PGAr (g) 	= Peak ground acceleration on rock (g), this specific cell is updated in the cell for BSSA14 and CB14, for others it is taken account for in the macros
27. ZBOT (km)	= The depth to the bottom of the seismogenic crust
28. ZBOR(km) 	= The depth to the bottom of the rupture plane
29. SS	=  1 for strike slip, automatically updated in the cell

The final file that will be used in the project is

data/Updated_NGA_West2_flatfiles_part1/Updated_NGA_West2_Flatfile_RotD50_d005_public_version.xlsx

## Final List of Paramters Considered for Study

[Parameter List 01](figures/param1.png)
[Parameter List 02](figures/param1.png)
[Parameter List 03](figures/param1.png)
