--------------------------------------------------------------------
-------------------Boids Model by NIS+------------------------
--------------------------------------------------------------------
======================================
Original code：
general machinelearning framework called Neural Information Squeezer to automatically extract the effective coarsegraining strategy 
and the macro-level dynamics, as well as identify causal emergence directly from time series data.

Notebook:
1.NIS model: 
models.py

2. EI calculation:
EI_calculation.py

--------------------------------------------------------------------
Code for Training NIS+Based on Boids Data：
The framework maximizes effective information, resulting in a macro-dynamics model with enhanced causal effects. Experimental
results on simulated and real data demonstrate the effectiveness of the proposed framework

Notebook:
1.NIS+ with REWEIGHT：
NIS+_REWEIGHT_for_boids.ipynb

2.NIS+ with Bidriection：
NIS+_Bidriection_for_boids.ipynb

--------------------------------------------------------------------
Noise data training
To examine the impact of intrinsic and observational perturbations on CE, two types of noises are introduced. 
Intrinsic noise is incorporated into the rule by adding random turning angles to each boid at each time step.
Extrinsic noise is assumed to affect the observational micro-states. In this case, we assume that the micro-states 
of each boid cannot be directly observed, but instead, noisy data is obtained. 

Notebook:
1.Training with the change of observational noises
Boids_with_observational_noise.ipynb

2.Training with the change of intrinsic noises
Boids_with_intrinsic_noise.ipynb

======================================
Data Analysis:

The Integrated Gradient (IG) method，comparison of effective information under different noise levels and scales.

Notebook: analysis_of_model.ipynb

--------------------------------------------------------------------
To compare the learning and prediction effects of NIS+ and NIS, we assess their generalization abilities by testing 
their performances on initial conditions that differed from the training dataset.

Notebook: Compare_effects_of_NISplus_and_NIS.ipynb

--------------------------------------------------------------------

The 'Data' folder contains some of the required data
The 'Models_16birds_2groups' folder contains the required trained models
The 'Figure' folder contains PDF format images for data visualization


