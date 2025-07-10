# Image-Segmentation-for-Disaster-Resilience
FloodNet Track 1 Dataset 
Overview
The FloodNet 2021 Track 1 dataset is a high-resolution aerial imagery dataset designed for post-flood scene understanding, focusing on semi-supervised tasks such as image classification and semantic segmentation. Captured using DJI Mavic Pro quadcopters after Hurricane Harvey in 2017, the dataset provides valuable resources for disaster management, enabling rapid and accurate damage assessment. The dataset is approximately 12 GB in size and contains high-resolution unmanned aerial system (UAS) imagery with detailed semantic annotations for flood-related damages.
This dataset is particularly useful for researchers and practitioners in computer vision, search and rescue (SAR), and environmental studies, aiming to develop models for flood detection and segmentation. It is released under the Community Data License Agreement (permissive) and is publicly available for non-profit scientific research.
Dataset Details

Source: The data was collected in Ford Bend County, Texas, and other areas impacted by Hurricane Harvey (August 30 - September 04, 2017).
Total Images: 2,343 images, divided into three splits:
Training: 1,445 images (~60%)
Labeled: ~400 images (51 flooded, 347 non-flooded)
Unlabeled: ~1,050 images


Validation: 450 images (~20%)
Test: 448 images (~20%)


Image Resolution: Original images are 3000 × 4000 × 3 pixels (RGB). For compatibility with most computer vision models, images are often resized to 224 × 224 × 3 or 416 × 416 during preprocessing.
Annotations: Pixel-level semantic segmentation annotations are provided for a subset of images, covering 10 classes:
Background
Building Flooded
Building Non-Flooded
Road Flooded
Road Non-Flooded
Water
Tree
Vehicle
Pool
Grass


Tasks:
Semi-Supervised Classification: Classify images into "Flooded" or "Non-Flooded" classes. Only a small portion of training images (~25%) have labels, with the rest being unlabeled, posing a challenge for semi-supervised learning.
Semi-Supervised Semantic Segmentation: Segment images into the 10 classes listed above. Only a subset of training images includes corresponding masks, and some masks may be cropped relative to the original images.


Size: The dataset is approximately 12 GB when compressed, with images stored in high-resolution formats.

Dataset Structure
The dataset is organized into the following directories:

train/: Contains training images and annotations.
images/: 1,445 RGB images (labeled and unlabeled).
annotations/: Pixel-level segmentation masks for ~400 labeled images (PNG format).
labels/: Classification labels for ~400 images (indicating "Flooded" or "Non-Flooded").


valid/: Contains 450 validation images with corresponding annotations and labels.
test/: Contains 448 test images with annotations for evaluation.

Note: Some masks may be cropped relative to the original images, so preprocessing is required to align images and masks correctly.
Downloading the Dataset
The FloodNet Track 1 dataset can be downloaded from the following link:

Google Drive Link

Alternatively, you can use the following Python code to download the dataset programmatically via the Supervisely Developer Portal:
import dataset_tools as dtools
dtools.download(dataset='FloodNet 2021: Track 1', dst_dir='~/dataset-ninja/')

Important: Ensure you have sufficient storage (~12 GB) and a stable internet connection. If you encounter issues with incomplete downloads, try downloading individual directories as described in the dataset documentation.
Project Timeline and Tasks
The following timeline outlines a four-week project plan for utilizing the FloodNet Track 1 dataset to develop and evaluate semantic segmentation models for flood scene understanding.
Week 1: Data Collection and Exploratory Data Analysis

Tasks:
Collect satellite/disaster datasets, including FloodNet Track 1.
Preprocess images and annotations (resize images, align masks).
Perform exploratory data analysis (EDA):
Analyze image dimensions and label distribution.
Visualize sample images and annotations to verify data quality.




Deliverables:
Dataset downloaded and organized.
Preprocessed dataset ready for training.
EDA report with visualizations (e.g., histograms, sample images with masks).



Week 2: Initial Model Training

Tasks:
Train a basic Convolutional Neural Network (CNN) for classification and a U-Net model for semantic segmentation on preprocessed data.
Apply data augmentation techniques (e.g., rotation, flip, brightness adjustments) to improve model robustness.
Evaluate models using Intersection over Union (IoU) and Dice Coefficient metrics.
Visualize segmentation outputs with ground-truth masks and predictions.


Mid-Project Review (End of Week 2):
Basic U-Net model trained and evaluated.
Segmentation masks generated for validation set.
Evaluation metrics (IoU, Dice Coefficient) and visualizations completed.



Week 3: Model Fine-Tuning

Tasks:
Fine-tune U-Net with pretrained encoders (e.g., ResNet, VGG).
Address class imbalance using focal loss.
Perform hyperparameter tuning (e.g., learning rate, batch size).


Deliverables:
Improved U-Net model with pretrained encoders.
Optimized hyperparameters and reduced class imbalance effects.
Updated evaluation metrics and visualizations.



Week 4: Final Evaluation and Reporting

Tasks:
Finalize the best-performing model based on validation results.
Evaluate the model on the unseen test set.
Prepare a final report with side-by-side comparisons of input images, ground-truth masks, and model predictions, along with conclusions.


Final Project Review (End of Week 4):
Best segmentation model saved.
Test set results compiled (IoU, Dice Coefficient, etc.).
Final report and presentation prepared.



Usage Instructions

Prerequisites:

Python 3.7 or higher
Libraries: numpy, pandas, matplotlib, seaborn, torch, torchvision, tensorflow, or keras (depending on your framework)
GPU (recommended for training deep learning models)
Storage: At least 12 GB free space for the dataset and additional space for preprocessed data


Setup:

Download and extract the dataset to a local directory (e.g., ~/dataset-ninja/).
Verify the dataset structure and ensure all images and annotations are present.
Install required dependencies using pip:pip install numpy pandas matplotlib seaborn torch torchvision tensorflow keras




Preprocessing:

Resize images to a compatible resolution (e.g., 224 × 224 or 416 × 416) for model training.
Align segmentation masks with images, as some masks may be cropped.
Handle class imbalance in classification tasks using techniques like weighted sampling or oversampling the minority class (flooded images).
Apply data augmentation (e.g., rotation, flip, brightness adjustments) to improve model robustness.


Exploratory Data Analysis (EDA):

Analyze image dimensions and label distribution using Python libraries like Matplotlib and Seaborn.
Visualize sample images and annotations to verify data quality.
Check for missing or corrupted files, especially in the annotations folder.


Training Models:

Classification: Use convolutional neural networks (CNNs) like ResNet50 or Xception for classifying images as "Flooded" or "Non-Flooded." Reported performance includes:
Training Accuracy: 99.84% (Xception)
Test Accuracy: 93.69% (ResNet50)


Segmentation: Use architectures like U-Net or PSPNet for semantic segmentation. Reported baseline IoU: ~80.35% with PSPNet.
Implement semi-supervised learning techniques to leverage unlabeled data.
Use metrics like IoU, Dice Coefficient, Accuracy, F1 Score, Precision, Recall, and ROC-AUC for evaluation.


Visualization:

Visualize segmentation outputs with ground-truth masks and overlaid predictions using Matplotlib.
Ensure proper scaling of masks (e.g., normalize to [0, 255]) to avoid black or incorrect visualizations.


Example Code:

For EDA, model training, and visualization, refer to sample scripts available in the FloodNet GitHub repository or the Supervisely Developer Portal.



Challenges and Considerations

Class Imbalance: The labeled training set has a significant imbalance (51 flooded vs. 347 non-flooded images). Use weighted sampling or focal loss to address this.
Semi-Supervised Learning: Most training images are unlabeled, requiring techniques like pseudo-labeling or self-training.
Memory Constraints: High-resolution images (3000 × 4000) may require significant memory. Resize images or use batch sizes of 4–8 for training on standard GPUs.
Annotation Issues: Some masks are cropped or misaligned. Preprocess data to ensure alignment before training.
Kaggle Compatibility: If using Kaggle, verify dataset paths (e.g., /kaggle/input/floodnet/FloodNet/) and adjust code for memory and GPU constraints.

Citation
If you use the FloodNet dataset in your work, please cite the following paper:
@article{rahnemoonfar2020floodnet,
  title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin},
  journal={arXiv preprint arXiv:2012.02951},
  year={2020},
  doi={10.48550/arXiv.2012.02951}
}

References

Rahnemoonfar, M., et al. (2021). FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding. IEEE Access, 9, 89644–89654. doi:10.1109/ACCESS.2021.3090981
FloodNet GitHub Repository
Dataset Download Link

Contact
For issues or questions regarding the dataset, contact the Bina Lab at the University of Maryland, Baltimore County, or refer to the GitHub repository for support.
