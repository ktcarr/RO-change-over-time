# **ENSO Recharge Oscillator Practical: Simulations and Forecasting**  

This repository hosts the **Recharge Oscillator (RO) Practical** for the [ENSO Winter School 2025](https://sites.google.com/hawaii.edu/enso-winter-school-2025/). The practical covers theoretical and computational aspects of the RO framework, its applications in ENSO simulations, and forecasting.

## **Instructors**  
- [Sen Zhao](https://senzhao.netlify.app/), Assistant Researcher, University of Hawaiʻi at Mānoa
- [Soong-Ki Kim](https://sites.google.com/view/skkim1), Postdoctoral Researcher, Yale University
- Jérôme Vialard

## **Code Contributors**  
- [Sen Zhao](https://github.com/senclimate), lecture notebooks, XRO, revised CRO code in python
- [Soong-Ki Kim](https://github.com/Soong-Ki), origional CRO code in Matlab
- [Bastien Pagli](https://github.com/bpagli), origional CRO code in Python

## **Session Outline**  

## **1. Observed ENSO properties related to RO**
-  Time series, seasonal variance, asymmetry, auto-/cross-correlation, spectrum analysis.

### **2. Solving and Fitting RO using the Community RO (CRO) Model (30 min)**  
- Introduction to the CRO code and running basic simulations, see [CRO Technical Note](CRO/CRO_Code_Technical_Note_Winter_School_v1.0.pdf)
- `XRO` framework to fit and solve RO

### **3. RO stochastic simulations with different complexities (45 min)**  
3.1) Validating the fitted NRO model's ability to reproduce observed ENSO properties. (5 min)  
3.2) ENSO seasonal phase locking & sensitivity to parameters (R’s seasonal cycle) (10 min)  
3.3) ENSO asymmetry & sensitivity to parameters (*b, B*) (10 min)  
3.4) ENSO spectrum/auto-correlation, cross-correlation (WWV-SST) & sensitivity to parameters (*F₁, F₂, R, ε*) (10 min)  

### **4. RO Forecasting (10 min)**  
- ENSO predictability experiments using CRO code
- Real RO forecast from January 2025  

### **5. RO Extensions (Extended Recharge Oscillator, XRO) (5 min)** 
- Introduce [XRO](https://github.com/senclimate/XRO) framework [(Zhao et al. 2024, Nature)](https://doi.org/10.1038/s41586-024-07534-6) for improved ENSO modeling and forecasting  

