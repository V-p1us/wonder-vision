## Task:
    - Each of the 20 camera scenes has a predefined region of interest (ROI) and a set of movements of interests (MOIs).
    - All vehicles appeared in the ROI are eligible to be counted.
    - If an eligible vehicle belongs to one of the MOI, it should be counted by the time it fully exits the ROI.  
    - Four-wheel vehicles and freight trucks are counted separately.
    - Two-wheelers (motorcycles, bicycles) are not counted.

## Training Rules
    - The validation sets are allowed to be used in training.
    - Data from previous edition(s) of the AI City Challenge are considered external data. These data and pre-trained models should NOT be utilized on submissions to the public leader-board. But it is allowed to re-train these models on the new training sets from the latest edition.
    - The use of any real external data is prohibited. There is NO restriction on synthetic data. Some pre-trained models trained on ImageNet/MSCOCO, such as classification models (pre-trained ResNet, DenseNet, etc.), detection models (pre-trained YOLO, Mask R-CNN, etc.), etc. that are not directly for the challenge tasks can be applied. Please confirm with us if you have any question about the data/models you are using. 

## Data Clarifications
    - Pickup trucks and vans should be counted as “cars” and the “truck” class mostly refers to freight trucks.
    - To be more specific, the following type of vehicles should be counted as “car”: sedan car, SUV, van, bus, small trucks such as pickup truck, UPS mail trucks, etc.
    - And the following type of vehicles should be counted as “truck”: medium trucks such as moving trucks, garbage trucks. large trucks such as tractor trailer, 18-wheeler, etc.

## Counting Clarification
    - The ROI is used in this track to remove the ambiguity that whether a certain vehicle should be counted or not especially near the start and end of a video segment. The rule is like this: any vehicle present in the ROI is eligible to be counted and a certain vehicle is counted at the moment of fully exiting the ROI. By following the rule, two people manually counting the same video should yield the same result. All ground truth files are manually created following this rule. During evaluation, a certain buffer will be applied to ensure that a couple frames of difference would not affect the final score. The detailed evaluation information will come up later.
    
    - A vehicle occluded during the entrance of the ROI but are visible in the ROI needs to be counted

    - A frame should be processed in 15s