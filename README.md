# **ENSO Recharge Oscillator Practical: Simulations and Forecasting**  

This repository hosts the **Recharge Oscillator (RO) Practical** for the [ENSO Winter School 2025](https://sites.google.com/hawaii.edu/enso-winter-school-2025/). The practical covers theoretical and computational aspects of the RO framework, its applications in ENSO simulations, and forecasting.

## **Instructors**  
- [Sen Zhao](https://senzhao.netlify.app/), Assistant Researcher, University of Hawaiʻi at Mānoa
- [Soong-Ki Kim](https://sites.google.com/view/skkim1), Postdoctoral Researcher, Yale University
- [Jérôme Vialard](https://scholar.google.com/citations?user=gVtN-0sAAAAJ&hl=en), Institut de Recherche pour le Développement

## **Code Contributors**  
- [Sen Zhao](https://github.com/senclimate), lecture notebooks, XRO, revised CRO code in python
- [Soong-Ki Kim](https://github.com/Soong-Ki), origional CRO code in Matlab
- [Bastien Pagli](https://github.com/bpagli), origional CRO code in Python

---
## **Overview**
The RO practical lecture, we will demonstrate how to use the `XRO` framework for **Recharge-Oscillator (RO) model** fitting, simulations, and reforecasting.

***Extended Nonlinear Recharge Oscillator (XRO) framework***

The `XRO` framework was developed to investigate the role of climate mode interactions in ENSO dynamics and predictability ([Zhao et al. 2024](https://doi.org/10.1038/s41586-024-07534-6)). When other climate modes are not considered, it simplifies to the Recharge Oscillator (RO), making it well-suited for use in this practical context. We have designed `XRO` to be user-friendly, aiming to be a valuable tool not only for research but also for operational forecasting and as an educational resource in the classroom.

> Check out the updated version of XRO at https://github.com/senclimate/XRO


***Community Recharge Oscillator (CRO) model framework***

The `CRO` code package is an easy-to-use Python/MATLAB software for solving and fitting the ENSO RO model. The `CRO` code is currently under development and is planned for release in 2025. The distributed version for the ENSO Winter School 2025 is a light Python version that includes only the essential features. While we introduce the `CRO` framework in this practical, some of its functionalities are unavailable. Therefore, for consistency, we will primarily use the `XRO` framework.

For those interested in the `CRO` code, please refer to the Jupyter notebook:  
📂 *CRO_test/RO_Practical_with_CRO_Framework.ipynb*

Special thanks to **Bastien Pagli** for providing the original `CRO` code in Python.

---

### Run it at Google Colab
You can easily run this notebook on [Google Colab](https://colab.research.google.com/). 

Simply download [this notebook](RO_parctical_with_XRO_framework.ipynb) and upload it to Google Colab. 

Once uploaded, you can execute the notebook directly— all required data and Python libraries will be downloaded and installed automatically.

