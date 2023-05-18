
# Structural Damage Detection

## Abstract
Prominent signs of wear and tear, such as cracks and openings on building wall indicate the wear and tear caused by stress over time, and when these defects occur in critical locations, such as load-bearing joints, they can lead to structural failure or collapse. Manually inspecting cracks can take time, and delays in identifying and repairing these cracks can have a significant impact on the structural integrity of infrastructure. To solve this issue we recommend implementing a crack detection method based on a convolutional neural network (CNN). The algorithm is composed of image processing image segmentation and CNN recognition. In the first part of the algorithm, cracks are easily recognized from the background image applying the Otsu’s thresholding technique and in the second step segmentation of the image is carried using k means clustering and the existence of cracks is recognized using CNN which is used to determine whether cracks are present or not. The outcomes have demonstrated how well the CNN model distinguished between wall cracks and non- cracks, and the accuracy results are graphically visualized.

## Dataset 
To train the model, a large dataset containing images of various concrete surfaces from Kaggle is used. The dataset contains two types of images. The positive class represents images of cracked wall surfaces, while the negative class represents images of cracked wall surfaces. Each class has a total of 20000 images.

[Dataset Download Link](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
## Results
<p align="center">
  <img alt="" src="https://github.com/SuhaanTonse/Structural_Damage_Detection/assets/83179192/7a6a5d7d-543f-4d75-828a-831d4f81ef83" width="45%">
  <img alt="" src="https://github.com/SuhaanTonse/Structural_Damage_Detection/assets/83179192/c3cd3769-ba6e-41fb-8f04-ba2d0d1daf76" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="https://github.com/SuhaanTonse/Structural_Damage_Detection/assets/83179192/643193f9-5118-43ac-8815-0ddafa3f2441" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp; 
  <img alt="" src="https://github.com/SuhaanTonse/Structural_Damage_Detection/assets/83179192/9503163d-eff4-47a5-92d5-023b1c215fa1" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp; 

</p>



