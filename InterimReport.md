# Ground Motion Prediction using Deep Neural Network

## Members

| Roll No.     | Name |
|----------|-------------------|
| 23D0292  | Anis M Vengasseri |
| 22D1264  | Moh. Zaid         |
| 23m2153  | Ashwin            |
| -------  | Vishnu T.S.       |


## Abstract
Historically, seismic events have brought extensive destruction and distress to communities worldwide.
These events have potential to generate economic losses of up to a staggering $200 billion (Japan, 1995).
In the current year alone, over 50,000 lives were tragically lost due to earthquakes(Turkey Earthquake).
The most devastating consequence of earthquakes is the ground vibration, which can inflict damage to infrastructure causing huge economic losses.
Throughout history, the prediction of this ground motion has remained a challenge for seismology groups. 
Researchers have explored various empirical equations, commonly known as attenuation equations or Ground Motion Prediction Equations (GMPEs), to estimate ground motion to engineer communities towards more seismic resilience. 
These empirical relationships were always challenged by new events or information. 
Recent studies in this field have explored the use of neural networks for the prediction of ground motions. 
In this project, we intent on an exploration of the application of neural networks in predicting ground motions, aiming to assess their effectiveness in comparison to traditional empirical equations.

## Brief Overview on Ground Motion Studies
The Earth's crust is comprised of tectonic plates that constantly move relative to one another.
This movement generates stress along the plate boundaries and faults.
When this stress exceeds a critical threshold, it triggers a sudden slip or movement at the boundary, releasing the accumulated energy in the form of waves.
The energy released propagates as waves across the plate boundary, causing ground vibrations. 
Seismographs are used to measure these vibrations, which scientists and engineers analyze to study and characterize future earthquakes and mitigate seismic vulnerabilities. 

Historically people has attempted to charcterise these ground motions/vibrations. 
The common parameters is peak ground accelartion (PGA), which is the highest ground accelaration measured at free ground.
Eventhough PGA provides intusion on ground shaking intensities, the behaviour of buildings/structures under ground motions are best described by Response Spectra.
It represents the maximum response of a structure to ground motion at various frequencies. 
In simpler terms, it shows how much a building or structure might shake at different levels of earthquake intensity, allowing engineers to design structures that can withstand these movements. 
By using this tool, they can create safer and more resilient buildings that can better withstand the forces of an earthquake. A typical process of estimating response spectra from ground motion is shown in figure below;

![Alt text](https://ars.els-cdn.com/content/image/1-s2.0-S1738573316300067-gr1.jpg)
Historically people has attempted to prepare database of such ground motion paramters recorded across the world.
Various attempts were made by individually various institutions.
The PEER research group had prepared more transperent version of such database.
The present studies consider this database for development of ground motion prediction model.
A brief overview of the database and preliminary discussion on data collected are discussed in the following section.

## PEER Database of Ground Motion
![Alt Text] ()
Tern of the centurey, PEER group initiated a large research program to develop Next Generation Attenuation equation, termed as project "NGA".
The project initially focussed on deelopment of common high-quality ground-motion database which can be accessed by all reserchers around the globe.
Earthquake ground motion records from various recording stations around the globe were collected and processed using commoon procedure to enure coerency.
A total of 335 earthquake events with multiple station records were collected and stored at common point. Locations of these ground motion are shown in figure eblow;

## Ground Motion Prediction Models
Variety of procedures were proposed for prediction of ground motion. 
The common procedure was empirical method, where functional form of ground motions are estimated through regresssion.
The functions forms is often expressed in terms of paramters characterising the  earthquake source, travel-path and recording site.
The common paramters are;
1. Magnitude of earthquake. This expresses quantity of energy released in log scale.
2. Distance to source from site.
3. Site Characteristics

A pictorial image of paramter is shown in figure.

## Development of Ground Motion Prediction Model
Multilayer Perceptrons (MLP) are often used in regression applications. The MLP's are very effcient in nonlinear regressions. 
The general regression is estimation of conditional density model of the form


$$ p(y | x; \theta) = N(y | f_{\mu}(x;\theta), f_{\sigma}(x;\theta)^2) $$

where $f_{\mu}(x;\theta)$ predicts the mean, and $f_{\sigma}(x;\theta)^2)$ predicts the variance.
In ground motion prediction equations, it is common to assume that the variance dependent of the input. 
This is called heteroskedastic regression.
The applications of Neural Network for the GMPEs, sofar has been limitted to homoscedastic regression.
This study would attempt an noval approach to develop an model using Bayesian MLP which capture mean and standard deviation of predictions correctly.
The advantage of the such model is the ability to estimate confidence interval of prediction, which are often critical in seismic mitigation studies and insurance studies.

### Heteroskedastic Regression
The traditional MLP by default don’t report the uncertainty of their estimates.
The uncertaintity build a Bayesian neural network which has two “heads” - that is, two endpoints of the network. One head will predict the value of the estimate, and the other to predict the uncertainty of that estimate. This dual-headed structure allows the model to dynamically adjust its uncertainty estimates, and because it’s a Bayesian network, also captures uncertainty as to what the network parameters should be, leading to more accurate uncertainty estimates.

## Preliminary Analysis

## Road Map

## Reference

1. Douglas, J., Aochi, H. A Survey of Techniques for Predicting Earthquake Ground Motions for Engineering Purposes. Surv Geophys 29, 187–220 (2008). https://doi.org/10.1007/s10712-008-9046-y
