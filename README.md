# PlanetDeepLearning
Wind throw detection using deep learning on Planetscope and high-resolution aerial images


The file "Tiling, Augmenting, Training and Predicting.ipynb" contains the code used during the thesis.
The file "trainvaltensorboard.py" contains additional code to create the TensorBoard data structure.

The file "CompletePredictions.tbx" contains the toolbox for ArcGIS Pro
The files "OrthoClassifier.py", "PlanetClassifier.py" and "TransferClassifierOrtho.py" are required by the toolbox.

The files ending with ".h5" are combiled and trained model files that can be used in the toolbox.
The files ending with ".hdf5" are weights for the models and are not required by the toolbox.
The file "VGG_model.zip" is a zipped version of the compiled VGG19 transfer learning model file.

The models/weights are named as follows:<br>
"Trained on satellite/ortho images" _ "Trained with satellite/ortho labels" _ "Prediction optimized for satellite/ortho labels" . "h5/hdf5"<br>
The VGG16 model is trained in ortho images, labels, and optimized for ortho label prediction.
