# Patient-specific_fine-tuning_CNN

Code written with python 3.5, keras, and tensorflow

Fine-tuning a pre-trained CNN with data of one patient to obtain a patient-specific CNN, which performs better on follow-up data of that same patient.

The code is used for : M.J.A. Jansen, H.J. Kuijf, A.K. Dhara, N.A. Weaver, G.J. Biessels, R. Strand, and J.P.W. Pluim, “Patient-specific fine-tuning of CNNs for follow-up lesion quantification”

In the paper two approaches are described: liver metastases detection and white matter hyperintensities segmentation. The code on this repository is for liver metastases detection, but is similar to the white matter hyperintensities segmentation with small changes.

The code on https://github.com/MarielleJansen/Liver-metastases-detection is used to train a CNN. This pre-trained CNN is updated to obtain a patient-specific CNN that has a better performance on follow-up data of the same patient than the pre-trained CNN.

MR data is used: dynamic contrast enhanced MR and diffusion weighted MR images. A liver mask is used to exclude the regions outside the liver from the results. This liver mask is obtained using the code from: https://github.com/MarielleJansen/Liver-segmentation
