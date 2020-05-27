[ICTNet](http://theictlab.org/lp/2019ICTNet/)

ICTNet: a novel network for semantic segmentation with the underlying architecture of a fully convolutional network, infused with feature re-calibrated Dense blocks at each layer.

The [source code](https://gitlab.com/Bodhiswatta/ictnet/tree/master/code) for ICTNet as described in the publication "On Building Classification from Remote Sensor Imagery Using Deep Neural Networks and the Relation Between Classification and Reconstruction Accuracy"


####IMPORTANT: To use this software, YOU MUST CITE the following in any resulting publication:

<pre>@inproceedings{chatterjee2019building,
  title={On Building Classification from Remote Sensor Imagery Using Deep Neural Networks and the Relation Between Classification and Reconstruction Accuracy Using Border Localization as Proxy},
  author={Chatterjee, Bodhiswatta and Poullis, Charalambos},
  booktitle={2019 16th Conference on Computer and Robot Vision (CRV)},
  pages={41--48},
  year={2019},
  organization={IEEE}
}</pre>


*  Software Dependencies
    1.  Python3
    2.  TensorFlow(GPU)
    3.  OpenCV 3.x
    4.  NumPy, GDAL, tqdm
*  Hardware Dependencies
    1.  GTX 1080 Ti or higher with 11GB or more frame buffer
    2.  System RAM 64GB (Recommended)

Dataset preparation
*  For training put images in folder data/train/images and ground truth in data/train/ground_truth under this project directory
*  For validation put images in folder data/validation/images and ground truth in data/validation/ground_truth under this project directory
*  For testing put images in folder data/test/images under this project directory


Train the model
*  Prepare training and validation data
*  Set is_training flag to True in code/config.py file
*  Run model using command [python main.py]
*  The numerical results for validation can be found in eval.txt
*  Validation results will be produced in data/validation/results/ with a new new folder created for every run of validation

Inference from the model
*  Prepare testing data
*  Set is_training flag to False in code/config.py file
*  Run model using command [python main.py]
*  Inference results will be produced in data/test/results/ with a new new folder created every time

Additional Configuration details (Coming soon)