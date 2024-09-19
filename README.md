# **Diagnosis of Autoimmune Skin Disorders Using Deep Convolutional Neural Networks**

**Abstract** 


Objective : This study aimed to utilise a convolutional neural network (CNN) to distinguish between biopsy-confirmed clinical cases of three autoimmune skin disorders namely vitiligo, lichen planus and psoriasis. 

Materials and Methods : The data consisted of photographs from 1019 psoriasis cases, 916 lichen planus and 1153 vitiligo, all confirmed by histopathology. In addition to that, 778 photographs of normal skin were selected. Data augmentation was applied to the dataset to enhance the quantity and diversity of the photographs. 20% of the images from the dataset are used for validation, and the remaining 80% are used for training. The entire data directory was used for testing. The CNN model chosen was DenseNet-121. Performance was evaluated using metrics such as accuracy, positive predictive value, negative predictive value, sensitivity, specificity, and F1-score. Gradient-weighted class activation mapping was also employed to highlight key regions associated with clinical features critical for model discrimination.

Results : The CNN model DenseNet-121 was able to differentiate between vitiligo, lichen planus, psoriasis and normal skin using the photographs.

Conclusions : This study demonstrates that the CNN model DenseNet-121 achieved an accuracy of 98% and an average F1-score of 0.98.

**Introduction**

Human skin, the body's largest organ, plays a key role in regulating processes like inflammation, immune response, wound healing, and angiogenesis, all closely tied to the immune system's function. However, autoimmune skin disorders disrupt these interactions, creating complex challenges in dermatology.[1] Autoimmune skin disorders include conditions such as psoriasis, vitiligo, lichen planus, lupus erythematous, and scleroderma, which involve the immune system mistakenly attacking the body’s own skin cells, leading to inflammation, lesions, and other skin abnormalities. 

Psoriasis is a chronic autoimmune condition that affects the skin in both adults and children. Recent studies emphasise the involvement of T cells, especially Th17 and Th1, in triggering the inflammatory process that leads to cytokine production, contributing to the formation of skin lesions. Key cytokines in this process include interleukin-23 (IL-23), interleukin-17 (IL-17), and tumour necrosis factor-alpha (TNF-alpha). The development of monoclonal antibodies and small molecule inhibitors targeting these cytokines has transformed treatment approaches, providing patients with safer and more effective options.[1]

Based on current evidence from hospital-based studies, primarily in North India, the prevalence of psoriasis in adults ranges from 0.44% to 2.8%, with 
a significantly lower prevalence in children. The peak onset age in adults is typically in the third and fourth decades of life, with a slight predominance in males. Chronic plaque-type psoriasis is the most common form, making up over 90% of cases. Other notable forms include palmoplantar psoriasis, pustular psoriasis, and recalcitrant psoriasis. For epidemiological classification, psoriasis can be categorised into early and late onset types. [2]

Lichen planus (LP) is an inflammatory condition affecting the skin and mucous membranes, with an unknown cause. It presents as itchy, purple papules and plaques, often located on the wrists, lower back, and ankles. The lesions are typically overlaid by a network of white lines known as Wickham striae, which are most easily visible on the buccal mucosa, where erosions may also occur. [3]. Lichen planus is estimated to affect 1-2% of the global population, with a higher prevalence observed in adults over the age of 40. The cutaneous form of lichen planus exhibits a similar incidence between males and females. However, oral lichen planus is more frequently diagnosed in women. Approximately 50% of all lichen planus cases involve the oral mucosa, indicating a significant occurrence of oral lichen planus within the affected population.[4]    

Vitiligo is an acquired condition affecting the skin and mucous membranes, marked by distinct macules and patches caused by the selective destruction of melanocytes. The prevalence of the disease from 0.5% to 1%, with the highest rates reported among Indians from the Indian subcontinent, where it reaches approximately 8.8%, the highest in the world. Mexico and Japan also have a high prevalence of vitiligo. Studies in India, China, and Denmark have reported lower prevalence rates of 0.093%, 0.005%, and 0.38%, respectively. Family history significantly impacts the prevalence of vitiligo, with rates ranging from 7.7% to over 50% among individuals with affected relatives, and the average age of onset is also earlier in those with a positive family history.[5] Photographic examples of Lichen Planus, Psoriasis, Vitiligo and Normal Skin can be seen in the ‣Fig. 1.

![image](https://github.com/user-attachments/assets/43ab7053-dd81-41d4-846a-38531838db6c)


The field of Artificial Intelligence (AI) in human healthcare has positively affected various medical and dental fields playing a key role. This has been made possible through machine learning (ML) and deep learning (DL) which makes AI a promising solution for skin disorder diagnosis. Through deep neural networks such as DenseNet, DL is able to mimic human brain thus excellently learning from training data. It is one type of convolutional neural network (CNN) that utilises image classification principles looking at contrasting differences within radio-density gradients like lucency and obscurity by always learning from the input. Besides that DenseNet can also evaluate and differentiate objects according to shape, pattern, contour or colour thereby increasing precision in medical imaging diagnoses. 

**Resources and Methods**

Data Preparation 

Photographs of Vitiligo, Lichen Planus, Psoriasis, and Normal Skin were collected from various sources, including but not limited to publicly available datasets (such as HAM10000) and open-source images. Duplicate images were removed by computing MD5 hashes for each image, comparing them to identify duplicates, and retaining only unique images in the dataset. The photographs were manually cropped to eliminate irrelevant areas (such as medical instruments, hands, and teeth), allowing the CNN to focus solely on the relevant regions. 20% of the images from the dataset are used for validation, and the remaining 80% are used for training. The entire data directory was used for testing.    Data augmentation was performed to increase the number of images in each class by approximately 30%, using techniques such as rotation, vertical flip, and horizontal flip from TensorFlow's Keras library, utilising the ImageDataGenerator class. This class facilitated the generation of augmented images by applying these transformations on existing images iteratively, and PIL was used for handling image operations. ‣Fig. 2 shows a comparison between the number of images before and after augmentation. Data augmentation increased the number of images in each class, thereby reducing class imbalance and enhancing the diversity of the training data. The augmented dataset supports more robust and accurate training of machine learning models, particularly for tasks involving image classification. 

CNN model DenseNet - 121 for Diagnosis

Convolutional Neural Networks (CNNs) are a subset of deep neural networks frequently used for visual data analysis. While other neural network models like fully connected neural networks and recurrent neural networks perform well on different data types, CNNs are particularly effective for image classification tasks.    DenseNet121 was selected for diagnosing Lichen Planus, Psoriasis, Vitiligo and Normal Skin due to its efficient use of parameters and superior classification accuracy on the ImageNet task. DenseNet121, or Densely Connected Convolutional Network, stands out for its unique architecture where each layer is connected to every other layer in a feed-forward fashion. This dense connectivity pattern addresses the vanishing gradient problem often encountered in deep neural networks by allowing gradients to flow more easily through the network, thus enhancing feature reuse and reducing the number of parameters required compared to traditional architectures with the same depth.

The model's architecture comprises four dense blocks interspersed with transition layers that include batch normalisation, convolution, and pooling layers, facilitating efficient down-sampling and information flow. This design helps the model maintain high representational power without significantly increasing computational complexity. By leveraging pre-trained weights from ImageNet—a large visual database used to benchmark algorithms in image recognition—the model benefits from transfer learning, which allows it to utilise previously learned features, thereby accelerating the training process and improving performance on the target task.   To optimise the model, the Adam optimiser (Adaptive Moment Estimation) was used. Adam is a popular optimisation algorithm in deep learning that combines the advantages of two other extensions of stochastic gradient descent: AdaGrad, which works well with sparse gradients, and RMSProp, which works well in online settings. Adam maintains two moving averages of the gradient (mean and uncentered variance) and adjusts the learning rate for each parameter dynamically. This leads to faster convergence and more robust performance across a variety of datasets.

![image](https://github.com/user-attachments/assets/67b6e1e8-5deb-4357-ba75-a1bbfadfc67d)


The model was trained using a categorical cross-entropy loss function, which is well-suited for multi-class classification tasks where the goal is to minimise the difference between the predicted probability distribution and the actual distribution (one-hot encoded labels). A batch size of 32 was chosen, meaning that the model processes 32 samples at a time before updating the weights, striking a balance between memory efficiency and training stability. The training was conducted over 10 epochs, where each epoch represents a complete pass through the entire training dataset, allowing the model to iteratively learn and refine its weights based on the loss and accuracy observed in each cycle. This approach, combined with DenseNet121's architectural advantages, facilitates robust and accurate classification of skin conditions in the dataset. 

The performance of the CNN model in diagnosing various autoimmune skin disorders was evaluated using several metrics: accuracy, positive predictive value (PPV or precision), negative predictive value (NPV), sensitivity (recall), specificity, and F1-score, with the equations for each metric summarised in ‣Fig 3. In multi-class classification, True Positives (TP) refer to instances correctly identified as belonging to the specific target class, while False Negatives (FN) occur when instances belonging to the target class are misclassified into other classes. True Negatives (TN) are instances correctly identified as not belonging to the target class, encompassing all other classes, and False Positives (FP) are instances incorrectly predicted as the target class when they belong to different classes. These metrics are computed for each class individually in a one-vs-all approach, forming the basis of performance evaluation in multi-class models. Accuracy measures the overall correctness of the model’s predictions. PPV and NPV indicate the proportion of correctly identified images within each class predicted by the model. Sensitivity, or recall, calculates the percentage of actual positives correctly predicted, while specificity measures the percentage of actual negatives accurately identified. The F1-score, which is the harmonic mean of PPV and sensitivity, serves as an indicator of the model's overall effectiveness and reliability.

![image](https://github.com/user-attachments/assets/1a5a1cac-2dab-4672-88da-19f158d1ca6d)

**Results**

The classification report, as shown in ‣Table 1 reveals high performance across all classes, with precision values ranging from 0.97 to 0.99, indicating the model’s proficiency in minimising false positive rates. Specifically, Vitiligo and Normal Skin exhibit exceptional precision at 0.99 each, reflecting the model's ability to accurately identify instances of these conditions with minimal incorrect classifications. Recall values are also notably high, spanning from 0.97 for Psoriasis to a perfect score of 1.00 for Normal Skin, suggesting the model's strong capacity to correctly identify true positive instances, thereby reducing the occurrence of false negatives. The F1-scores, which represent the harmonic mean of precision and recall, are consistently high (0.97 to 0.99) across all classes, highlighting balanced performance in both sensitivity and precision.    

![image](https://github.com/user-attachments/assets/7552d63e-6ad8-496a-a24e-4c8d9e57e6d8)


The confusion matrix, shown in ‣Fig. 4, provides further granularity into the model’s classification behaviour. For Vitiligo, the model correctly classified 1471 instances while misclassifying 17 instances (8 as Normal Skin, 8 as Psoriasis, and 1 as Lichen Planus). Normal Skin exhibits perfect classification, with 1008 correct predictions and no misclassifications, aligning with the perfect recall reported. Psoriasis was accurately classified 1527 times, with 50 instances misclassified primarily as Lichen, reflecting a potential area for further refinement. Lichen was correctly identified in 1389 cases, with 28 instances incorrectly predicted as Psoriasis, which mirrors the classification challenges noted between these similar conditions. The misclassified images are shown in ‣Fig. 5.   

![image](https://github.com/user-attachments/assets/1e216dfc-977d-4e80-9d89-82510941e2c2)

The training process shows a steady improvement in accuracy and a consistent reduction in loss values across the epochs, starting from an initial accuracy of 66.96% and a loss of 0.9536 in the first epoch. By the tenth epoch, the model’s training accuracy reached 97.96%, with a training loss reduced to 0.0589. This steady decrease in loss and corresponding increase in accuracy suggest that the model is effectively optimising and converging as expected. The validation performance mirrors the trends observed during training, with the validation accuracy consistently remaining high throughout the epochs. It started at 91.99% in the first epoch and remained above 87%, peaking at 93.44% by the tenth epoch. The validation loss showed fluctuations, ranging between 0.1861 and 0.3657, reflecting slight variances possibly due to the validation set’s complexity or slight overfitting in some instances. However, the overall trend indicates strong generalisation ability, as the validation accuracy remains closely aligned with the training accuracy.

  
The model achieves an overall accuracy of 98%, underscoring its high level of performance in a multi-class classification setting. Both the macro and weighted averages for precision, recall, and F1-score are 0.98, indicating a uniformly high level of performance across all classes, irrespective of their individual prevalence. These results reflect the model’s ability to generalise well across diverse skin conditions, providing robust diagnostic capabilities.

![image](https://github.com/user-attachments/assets/1d6a4caf-d9a8-4e8a-aeba-e6c9ebbedaa9)


**Conclusion**

In summary, the model demonstrates exceptional performance in classifying Vitiligo, Normal Skin, Psoriasis, and Lichen Planus, with high accuracy and balanced performance across all metrics. The insights derived from the confusion matrix highlight areas for potential enhancement, particularly in distinguishing closely related conditions. With further refinement and validation, the model holds significant promise for deployment in clinical settings, contributing to improved diagnostic accuracy and patient outcomes in dermatology. Despite the excellent overall performance, the minor fluctuations in validation loss suggest potential areas for further enhancement, such as fine-tuning the learning rate, adjusting the batch size, or implementing advanced data augmentation techniques. Additionally, further testing on more diverse and extensive datasets could help confirm the model’s robustness across varied conditions.

**References**

[1] - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10650048/

[2] - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5134160/

[3] - https://www.ncbi.nlm.nih.gov/books/NBK526126/

[4] - https://www.nhsinform.scot/illnesses-and-conditions/skin-hair-and-nails/lichen-planus/#:~:text=Who's%20affected%20by%20lichen%20planus,planus%20(oral%20lichen%20planus).

[5] - https://www.ijced.org/html-article/16173




