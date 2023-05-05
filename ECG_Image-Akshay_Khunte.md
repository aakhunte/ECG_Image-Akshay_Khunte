# Automated multilabel diagnosis on electrocardiographic images and signals
#### Sangha et al. (2022)

Akshay Khunte

May 4, 2023

## Paper Walkthrough
### Motivation
This paper aimed to develop a deep learning algorithm that can automate the diagnosis of multiple cardiovascular conditions using electrocardiographic (ECG) images and signals. While automated diagnosis using signal data has been successful, the authors point out that ECG machines typically do not save or export signal data, instead printing them immediately after signal acquisition and subsequently deleting any data recorded on-device. This makes it challenging to apply signal-based models in real-world clinical settings, particularly in low-resource areas which are unable to purchase newer ECG devices which can directly integrate these models on-device. Additionally, existing models are trained and tested on data from a single source, limiting their generalizability to different institutions and health settings. 

The motivation for this paper is to address the gap in accessible AI-ECG models due to their dependency on signal data and to provide a tool that can improve clinical workflow, particularly for key cardiovascular conditions. The authors note that automated diagnosis based on ECG images can be particularly useful for paraclinical staff serving in remote settings and patients who lack access to experts for early diagnosis. 

To achieve this, the authors developed a multilabel prediction algorithm that can incorporate either ECG images or signals as inputs to predict the probability of various rhythm and conduction disorders. They used over 2 million ECGs from Brazil and independently validated the model using data from Germany. The image-based model was also tested on real-world printed ECGs, demonstrating its practical clinical utility. 

### Methodology
#### Data source and study population
The authors of the paper used 12-lead electrocardiograms (ECGs) collected by the Telehealth Network of Minas Gerais (TNMG) in Brazil between 2010 and 2017, as a part of the CODE study. These ECGs were collected from 811 municipalities in the state of Minas Gerais and were recorded as standard 12-lead recordings sampled at frequencies ranging from 300 to 600 Hz for 7–10 s. The dataset also included patient demographics and six clinical labels.

The authors used automated University of Glasgow statements and Minnesota codes to extract clinical labels from the ECGs. They also used a semi-supervised Lazy Associative Classifier trained on a dictionary created from text reports to extract text labels from expert reports written upon initial reading of the signals. The discrepancies between the extracted labels and automatic analysis were settled using manual review and cutoffs related to ST, SB, and 1dAVb.

To externally validate the model, the authors used a secondary annotation test dataset, which was validated by independent cardiologists based on American Heart Association criteria. A third ECG dataset from Germany was also used for external validation.

#### Data preprocessing
After ECGs containing significant missing signal were discarded, ECGs were processed by resampling all signals to 300 Hz, removing baseline wander, and capturing 5 seconds of signal across all 12 leads. ECGs were then plotted to make computer-generated images in four different formats, varying in the location and duration of different leads. ECGs were also rotated prior to training to further increase the model's robustness to orientation and data organization variations not seen during training. This ensured that the model would look for and thus use lead-specific information, instead of simply localizing its predictive cues to the same part of every given ECG.

Figure: ECG Preprocessing Strategy. Note that ECGs of four different formats were used to optimize generalizability of the model. Validation was conducted in multiple datasets.
![](https://github.com/aakhunte/ECG_Image-Akshay_Khunte/blob/main/images/ImagePreprocessingAndTrasnformation.png?raw=true)

#### Study Outcomes
Diagnostic labels for six different rhythm and conduction disorders (AF, RBBB, LBBB, 1dAVB, ST, and SB) were obtained from the ECG diagnostic statements. The model was developed to output predictions for these six labels, which are not mutually exclusive, and a seventh "hidden" label: gender, which is not typically discernable from ECGs by human readers.

#### Image Model Architecture
An EfficientNet-B3 Convolutional Neural Network model architecture pretrained with ImageNet weights was used for the model. This model, which consists of over 10 million trainable parameters, took a 300x300 pixel image as input, and the images were thus scaled accordingly. The model was trained with a minibatch size of 64 at a learning rate of 5x10^-3 for seven epochs.

#### Signal Model Architecture
An InceptNet-based Convolutional Neural Network model architecture was used to train the signal model, which served as a point of comparison for the image model. This model outputted predictions based on both the 12x1500 signal data and an 8x1 array of derived ECG elements, such as the size of various intervals.

#### Model Interpretability
One of the big advantages of an image-based model is the ability to interpret the model's predictions based on image-based activation mapping techniques, such as Gradient-weighted Class Activation Mapping (Grad-CAM). Grad-CAM allows the evaluation of whether model-assigned labels are based on clinical features or on other, non-phenotypic features of the ECG. The authors calculated the gradients of the final stack of filters in the network for each prediction class of interest and created filter importance weights by performing a global average pooling of the gradients in each filter. They then multiplied each filter in the final convolutional layer by its weight and combined them across filters to build a Grad-CAM heatmap and overlayed it on the original ECG images.

Two approaches were used to assess model interpretability: first, individual ECGs were examined, and the Grad-CAM heatmaps were overlayed for different formats to show that the heatmaps localized on different regions of the same ECG when inputted in different formats. Second, class activation maps for a condition of interest were averaged, and this average heatmap was overlayed over an ECG with that condition. LBBB and RBBB were used to illustrate interpretability as these labels have lead-specific information that is used to make the clinical diagnosis, whereas the other labels can be deduced from any of the ECG leads, thus not requiring any lead-specific learning.

### Results
#### Model Performance
The image model performed extremely well across diagnostic labels for both the held-out and internal validation test sets along with the external validation set. The study found that image and signal models performed comparably for clinical labels on both datasets, with high correlation between predictions across labels, indicating that the transformation to the image modality successfully retained all necessary diagnostic information. The image-based model outperformed the signal-based model for the higher-order label of gender, while both models had high discrimination across labels and in all three datasets. The label-level performance of both models was consistent, with the highest AUROC and AUPRC scores on the same clinical labels, LBBB and RBBB, and lowest scores on the same class, 1dAVb. Confusion matrices showed that predictions of LBBB, RBBB, and ST were the most accurate for both image and signal-based models among ECGs with only one clinical label. The researchers also found that the inclusion of peak morphology along with the signal data in the signal model did not boost performance significantly compared to a model trained on the signal alone.

Figure: Table containing results for image and signal-based model in held-out test set and the cardiologist-validated set. Similar performance was reported across the other external validation sets.
![](https://github.com/aakhunte/ECG_Image-Akshay_Khunte/blob/main/images/ResultsTable.png?raw=true)

#### Grad-CAM Analysis
The study used Grad-CAM to identify the regions of an ECG that were most important for the diagnosis of RBBB and LBBB, two rhythm disorders which require lead-specific information for diagnostic purposes. This is a critical distinction, as the model successfully localizing Grad-CAM weights to different regions of differently-arranged ECGs reflecting these conditions reflects its ability to successfully pick up on true ECG features across formats, instead of fixating on the same potentially variable regions of different ECGs. The other conduction and rhythm disorders evaluated in this study, along with gender, are not associated with such lead-specific cues and thus would not reflect the model predictions' interpretability as well.

The study found that the region of the ECG corresponding to the precordial leads was the most important for the prediction of RBBB across both the standard and alternate images, with the region corresponding to leads V4 and V5 especially important in the standard format, and V1, V2, and V3 in the alternate format. On the other hand, regions corresponding to lead V6 were most important for the prediction of LBBB across standard images, while regions corresponding to lead V4 and V5 were most important for LBBB predictions across alternate images. The study also found that the rhythm lead was important for the prediction of both LBBB and RBBB in the standard format. The Grad-CAMs for individual representative examples of model prediction of RBBB and LBBB on real-world images from the web-based dataset showed that the precordial leads were the most important for the prediction of the label, despite varying in the relative position of the leads and the difference in the number and type of the continuous rhythm strip at the bottom of the ECG image. Overall, the Grad-CAM analysis provided interpretability of the model's predictions and identified regions in the ECG that were important for the diagnosis of RBBB and LBBB, showing that the model successfully picks up on and identifies lead-specific cues independent of format, and thus should generalize well to unseen ECGs.

Figure: Grad-CAM Mapping for Predictive cues for RBBB/LBBB in Standard and Alternate format ECGs. As seen in the figure, the model focused on different regions of the differently-formatted ECGs even when evaluating for the same condition, and also varied across the two conditions when the format was held constant.
![](https://github.com/aakhunte/ECG_Image-Akshay_Khunte/blob/main/images/gradCamMapping.png?raw=true)

## Thoughts and Comments
### Most interesting aspects of the work
* The applicability of the model is very clear; it is accompanied by a website, and upon testing, it is very clear how quickly it is able to generate a diagnosis for an ECG and how versatile it is across labels.
* The image-based models are completely invariant to the layout of the ECG images, and even generalizes to formats unseen during training.
* Using Grad-CAM analysis enables interpretable recognition of how the model picks up on specific leads of interest and abnormalities in the ECG, an important feature for any clinical diagnostic tool which is not as feasible for signal-based models.
* The extensive external validation of the model, beyond the typical internal, held-out test set.
* The clear value proposition of the image-based model for rural and lower-resourced areas. Developing models for signal data is the more conventional approach and makes sense at face value, but the paper makes the inaccessibility of this approach much more obvious and allows for much quicker integration into clinical workflows, especially abroad.

### Limitations of the work as presented
* The reported performance on the held-out test and external validation sets were limited by the quality of labels, which likely varied given higher performance on certain conditions.
* The reasoning for why the model occasionally did not make the correct prediction or why a label was marked incorrectly could not be confirmed for all ECGs. This is problematic, as a model used for cardiovascular diagnosis would be relatively high-stakes if actually used in clinical setting, so some oversight is certainly necessary to avoid the life-or-death consequences of such mistakes.
* While the signal-based models have a higher frequency record of the electrocardiographic activity of the heart, it is unclear why the image-based models perform comparably. This suggests the signal model might require certain advancements, and that the performance gap between the signal and image models might be more reflective of the limitations of the signal model architecture used instead of the images being a better data modality for clinical interpretation.

### Potential future paths to follow to improve the model and build upon this research further.
* This model could be improved by using more data from a wider set of clinics for training, to reduce any overfitting to the Brazil dataset and improve generalizability.
* The model could also be trained directly on true ECG printout PDFs, instead of computer-generated images. These print-outs could have more artifacts and noise and may be more representative of the real-world application.
* More formats and image augmentation could be done during training, such as greater degrees of rotation, color augmentation, or masking, to improve real-world performance. This would be particularly useful for non-standardized input images, such as phone-captures photos of crumpled up or stained ECGs, which might otherwise yield incorrect predictions with the current approach.

### References (note: all figures are also from this paper)
* Sangha, Veer, Bobak J. Mortazavi, Adrian D. Haimovich, Antônio H. Ribeiro, Cynthia A. Brandt, Daniel L. Jacoby, 	Wade L. Schulz, Harlan M. Krumholz, Antonio Luiz P. Ribeiro, and Rohan Khera. 2022. “Automated 	Multilabel Diagnosis on Electrocardiographic Images and Signals.” Nature Communications 13 (1): 	1583.
