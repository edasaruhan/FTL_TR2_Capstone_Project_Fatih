# Introduction and Literature Review

Abdominal pain constitutes a significant proportion of the reasons for admission to the emergency department. The abdominal cavity contains a diverse range of organ structures, implying a diverse range of disease causes. The first step to diagnosing the pain in the emergency department is radiological imaging. Radiological imaging tests are frequently used to confirm or exclude a suspected condition and to narrow the differential diagnosis list.

Parakh et al. [1] proposed a cascade CNN model, CNN-1 to determine the range of the urinary tract and CNN-2 to detect the stones. The study was conducted on 535 combined CT images, 100 for testing and 435 for training. It is worth mentioning that ImageNet and GrayNet(built-in datasets) have been used as pertained datasets. The proposed models produced a percentage of accuracy of 95% with GrayNet and 91% with ImageNet.

The study, proposed in [2] appendicitis detector. A 3D convolutional neural network (3D-CNN) model called AppendiXNet has been constructed to classify the CT images by developing AppendiXNet consists of 18 layers. Moreover, there were 646 CT images, of which 438 were taken to train the network, 156 for development and 102 for testing, in addition to 500,000 video clips, which are used as more pertinent data to enhance the system. The proposed system achieved an area under the curve (AUC), sensitivity, and accuracy of 0.81, 0.784, and 0.725, respectively.

In another study, Park et al. [3] employed a system for appendicitis identification. They used a 3D-CNN network with a kernel size of 3x3. A supervised localization approach has been used for network training. Cross-entropy has been used as a loss function. A dataset containing a total of 667 CT images (215 with acute appendicitis and 452 with a normal appendix) has been used for external and internal validation. Their model achieved an accuracy of 90% for both the internal and external validation.

In the method proposed at [4], the authors employed a cross-residual network (XResNet-50) for kidney stone detection. A different cross-sectional CT image dataset consists of a total of 1799 images, which have been used. Their approach reaches an accuracy of 96.82% for both large and small-sized kidney stones.

Moreover, authors in [5] proposed a system for ascites identification and quantification. They used 6337 CT as a training set and 1635 CT images as a testing set. The system employed a recurrent residual U-Net (R2U-Net), bidirectional U-Net, deep residual U-Net, and U-Net to develop a single deep learning model (DLM) to segment ascites areas. They attained segmentation accuracy with a mean intersection over union (IoU) value of 0.87, 0.80, 0.77, and 0.67 for deep residual U-Net, U-Net, bidirectional U-Net, and R2U-Net models, respectively. Consequently, the detection accuracy was 0.96, 0.90, 0.88, and 0.82 for U-Net, U-Net, bidirectional U-Net, and R2U-Net models, respectively. In terms of sensitivity and specificity, the U-Net model achieved the highest results of 0.96 and 0.96, respectively.

## Problem Statement

A diagnostic radiologist's essential role is to analyze and interpret medical images. [6], the large number of patients in the emergency department cause shortage of radiologist. In addition, using a computer-aided diagnosis system will minimize physician-induced errors.

Each year, kidney cancer affects over 430,000 people, leading to approximately 180,000 fatalities [7]. The incidence of kidney tumors surpasses this number, yet current radiographic techniques often cannot distinguish between malignant and benign tumors [8]. Furthermore, a significant proportion of presumed malignant tumors exhibit slow growth and indolent behavior, prompting the adoption of "active surveillance" as a preferred management approach for small renal masses [9].

The primary aim of this project is to alleviate the workload of healthcare professionals by implementing an accurate classification system for kidney cases across four distinct clinical conditions (Cyst, Normal, Stone, Tumor). By precisely categorizing cases into these clinical conditions, the project endeavors to streamline diagnosis and treatment processes, ultimately contributing to improved patient care and outcomes.

## Relevance to Sustainable Development Goals (SDGs)

The project aligns with key United Nations Sustainable Development Goals:

1. **Good Health and Well-being (SDG 3)**: Improving healthcare outcomes by developing an accurate kidney condition classification system for early intervention.
2. **Industry, Innovation, and Infrastructure (SDG 9)**: Fostering innovation in healthcare technology with the development of advanced deep learning models.

## Dataset

This project utilizes the publicly available CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone [10]. The dataset consists of 12,446 CT scan images categorized into four classes:
- Cyst (3,709 images)
- Normal (5,077 images)
- Stone (1,377 images)
- Tumor (2,283 images)

## Data Source and Collection

The CT images were collected from the Picture Archiving and Communication System (PACS) of several hospitals in Dhaka, Bangladesh. The dataset includes both contrast and non-contrast studies of the whole abdomen and urogram in both coronal and axial cuts. The researchers carefully selected DICOM studies containing confirmed diagnoses of kidney cysts, stones, normal findings, or tumors. They then created batches of DICOM images for each region of interest (ROI) associated with a specific radiological finding. Patient information and metadata were subsequently removed from the DICOM images before conversion to a lossless JPG format.

## Proposed Method

The data will be used will contain a contrast or non-contrast CT series of the kidney related to four different clinical conditions (Cyst, Normal, Stone, Tumor). First of all, the data will be preprocessed before the classification process. At this stage, CT images will be balanced and transformed according to the expected input size of the convolutional neural network (CCN). Accordingly, this reduces the memory used by CCN and processing time. Then, various data transformations such as vertical flip, rotation, zooming, and warping will be applied to the scaled CT images to ensure the diversity of the training set. Lastly, CT images will be normalized and fed to a faster RCNN based model. The architecture of the model will be modified to give high accuracy in classification. Also, our model will be pretrained on an alternative dataset or pre-trained model will be used to optimize the adjustment of its hyperparameters and weights. For model evaluation, confusion matrices, accuracy, precision, recall (sensitivity), specificity, F1-score, and Area Under Curve (AUC) will be computed.

## References

[1] A. L. H. L. J. H. E. B. H. S. D. V. &. D. S. Parakh, Urinary stone detection on CT images using deep convolutional neural networks: evaluation of model performance and generalization., Radiology: Artificial Intelligence, 1(4), e180066, 2019.
[2] P. P. A. I. J. C. C. B. M. M. D. .. &. P. B. N. Rajpurkar, AppendiXNet: deep learning for diagnosis of appendicitis from a small dataset of CT exams using video pretraining, Scientific reports, 10(1), 3958., 2020.
[3] J. J. K. K. A. N. Y. C. M. H. C. S. Y. &. R. J. Park, Convolutional-neural-network-based diagnosis of appendicitis via CT scans in patients with acute abdominal pain presenting in the emergency department, Scientific reports, 10(1), 9556., 2020.
[4] K. B. P. G. T. M. Y. O. K. M. &. A. U. R. Yildirim, Deep learning model for automated kidney stone detection using coronal., Computers in biology and medicine, 135, 104569.‏, 2021.
[5] H. H. J. K. K. W. C. H. K. Y. K. J. K. .. &. L. J. Ko, A deep residual u-net algorithm for automatic detection and quantification of ascites on abdominopelvic computed tomography images acquired in the emergency department: Model development and validation, Journal of Medical Internet Research, 24(1), e34415., 2022.
[6] M. H. J. W. H. X. &. K. P. Hesamian, Deep learning techniques for medical image segmentation: achievements and challenges, Journal of digital imaging, 32, 582-596., 2019.
[7] H. e. a. Sung, GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries, : CA: a cancer journal for clinicians 71.3 209-249., 2020.
[8] A. D. &. P. I. de Leon, Imaging and screening of kidney cancer, Radiologic Clinics, 55(6), 1235-1250, 2017.
[9] M. C. C. U. B. R. O. I. S. M. K. M. .. &. P. P. M. Mir, Role of active surveillance for localized small renal masses., European urology oncology, 1(3), 177-187., 2018.
[10] [CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone).

