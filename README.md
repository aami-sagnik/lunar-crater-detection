# Lunar Crater Detection
This project aims to detect craters on the lunar surface using RetinaNet, Faster RCNN and convolutional block attention module (CBAM). We implemented both Faster RCNN and RetinaNet with CBAM on the FPN outputs in PyTorch and trained them on an annotated dataset that we curated from ISRO's OHRC dataset. The dataset is available [here](https://drive.google.com/file/d/1S_mkadfQkPtaK4uOn6jyPCxKxvtAl_W_/view?usp=sharing).

Here is a sample detection result using our RetinaNet + CBAM model:
![Sample lunar crater detection result](sample.png)

Here are the results we obtained:
![Lunar crater detection results](results.png)
