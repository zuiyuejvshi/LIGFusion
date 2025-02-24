
## Abstract

Multimodal image fusion plays a crucial role in complex scenarios such as nighttime surveillance and autonomous driving, with infrared-visible image fusion receiving significant attention. This paper proposes LIGFusion (Light-Integrated and Guided Fusion), a novel fusion guidance method designed to address the challenges of uneven illumination, limited global information extraction, and high-frequency information loss in existing multimodal image fusion techniques. LIGFusion introduces a light-discriminative perception module and a light-adaptive loss function, enabling dynamic adjustment of feature extraction and fusion strategies based on illumination conditions. To enhance global feature extraction, a Transformer-based module is incorporated to mitigate the limitations of CNNs. Additionally, Difference of Gaussian (DoG) feature extraction and fusion are employed to preserve high-frequency details during the forward propagation of the fusion network. Experimental results on the MSRS, RoadScene, and TNO datasets demonstrate that LIGFusion achieves superior performance in thermal target saliency, detail preservation, and noise suppression, outperforming mainstream methods such as TIMFusion and TUFusion across multiple evaluation metrics. Ablation studies validate the effectiveness and rationality of the proposed network modules. Furthermore, LIGFusion proves effective in object detection and semantic segmentation tasks, providing high-quality fused images to support applications in complex real-world scenarios.


### Training
1. Recommended Environment
 - [ ] einops == 0.4.1
 - [ ] h5py == 3.11.0
 - [ ] kornia == 0.2.0
 - [ ] numpy == 1.24.4 
 - [ ] opencv_python == 4.11.0.86 
 - [ ] scikit_image == 0.19.2 
 - [ ] timm == 0.6.12 
 - [ ] torch == 1.13.1 
 - [ ] torchvision == 0.14.1 
 - [ ] tqdm == 4.60.0

**2. Data Preparation**
Download the required weights and datasets for the code from [this link](https://drive.google.com/drive/folders/1wzKY8hzut4yNaKTZ-uYMng_eToGixhM1?usp=sharing).Create a folder named 'data' and a folder named 'dataset', and place the downloaded dataset into them

**3. Start Training**

The training stages I, II, and III in the paper correspond to the traincls in the code, respectively, train1，train2

Run 
```
python traincls.py
python train1.py
python train2.py

``` 
and the trained model is available in ``'./models/'``.

### Testing

The pre trained model and the required test set have been included in the link above.

Create a folder named 'test_img' and place the downloaded test set in it.
