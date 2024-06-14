# Landmark-classification-and-tagging-for-social-media
This project focuses on using Convolutional Neural Networks (CNNs) to build a landmark classifier. It addresses the challenge of inferring location data for images lacking metadata, a common scenario in photo sharing services. By automatically detecting and classifying landmarks, this project streamlines the user experience. The process involves data preprocessing, training CNNs, and deploying the best model as an app.

The high level steps of the project include:

**Create a CNN to Classify Landmarks (from Scratch)** - Here, we will visualize the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks. We will also describe some of the decisions around data processing and how we chose we network architecture. We will then export we best network using Torch Script.

**Create a CNN to Classify Landmarks (using Transfer Learning) **- Next, we will investigate different pre-trained models and decide on one to use for this classification task. Along with training and testing this transfer-learned network, we will explain how we arrived at the pre-trained network we chose. We will also export we best transfer learning solution using Torch Script

**Deploy we algorithm in an app** - Finally, we will use we best model to create a simple app for others to be able to use we model to find the most likely landmarks depicted in an image. We will also test out the model and reflect on the strengths and weaknesses of we model.
