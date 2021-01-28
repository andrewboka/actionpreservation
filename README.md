# Measuring Action Preservation via Gaze Estimation

Measuring how well actions are preserved in identity-masked driver videos is an essential consideration when anonymizing driver data for further research. The primary goal is to de-identify the subjects in video, however maximizing the action information in the video defines how useful the data can be.

This repository contains an implementation of an action preservation measuring technique using a deep learning based gaze-estimator. 

## Outline

1. [Dependencies and Installation](#Dependencies-and-Installation)
2. [Directory Structure & Usage](#directory-structure--usage)
3. [Testing](#testing)
4. [List of Papers](#list-of-papers)


## Dependencies and Installation


We have tested this code in Google Colab with the following:
* Python 3.6
* PyTorch 1.7
* CUDA 10.1

If used on a cloud-based environment like Google Colab, zip the repository first before upload.
## Directory Structure & Usage

* `models`: Contains neural net gaze estimator and pre-trained weights from [Mobile FaceGaze](https://github.com/glefundes/mobile-face-gaze).
* `mtcnn`: Contains face-detection model from [Mobile FaceGaze](https://github.com/glefundes/mobile-face-gaze).

The repository also provides an example video and Jupyter notebook file of the gaze estimation based action preservation measure implementation. 

For use, run the Jupyter notebook cells in order. The notebook imports necessary packages and input videos. 
## Testing

The action preservation is estimated by extracting the predicted gaze-vectors from both the original un-filtered video and de-identified video and measuring the difference between the two vectors. Scoring closer to 0 implies effective action preservation since the gaze estimation is relatively unchanged. 

Currently the notebook computes this score for one frame of the video, but further code additions will be added to support computation over all frames in the video. Once the difference values are measured for every frame, an average can be computed for a more robust estimate of action preservation.

## List of Papers

The following lists titles of papers from this project.
1. Zhang, X., Sugano, Y., Fritz, M., Bulling, A.: [It's Written All Over Your
Face: Full-Face Appearance-Based Gaze Estimation](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/html/Bulling_Its_Written_All_CVPR_2017_paper.html). In: IEEE Conference
on Computer Vision and Pattern Recognition Workshops. pp. 51–60 (2017).
2.  Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao.
[Joint face detection and alignment using multitask cascaded
convolutional networks](https://arxiv.org/abs/1604.02878). SPL, 2016

