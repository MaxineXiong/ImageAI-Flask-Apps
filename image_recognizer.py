from imageai.Classification import ImageClassification
import os
import glob


class ImageRecognizer:
    # Dictionary mapping algorithms to their model file names
    MODELS = {'ResNet50': 'resnet50-19c8e357.pth', 'MobileNetV2': 'mobilenet_v2-b0353104.pth',
              'InceptionV3': 'inception_v3_google-1a9a5a14.pth', 'DenseNet121': 'densenet121-a639ec97.pth'}

    # execution_path: where the models are saved
    def __init__(self, execution_path, algorithm):
    """
    Initialize the ImageRecognizer object.

    Parameters:
    - execution_path: The path where the models are saved.
    - algorithm: The selected algorithm for image recognition.
    """
        self.prediction = ImageClassification()

         # Set the model type based on the selected algorithm
        if algorithm == 'ResNet50':
            self.prediction.setModelTypeAsResNet50()
        if algorithm == 'MobileNetV2':
            self.prediction.setModelTypeAsMobileNetV2()
        if algorithm == 'InceptionV3':
            self.prediction.setModelTypeAsInceptionV3()
        if algorithm == 'DenseNet121':
            self.prediction.setModelTypeAsDenseNet121()

        # Set the model path based on the execution path and selected algorithm
        self.prediction.setModelPath(os.path.join(execution_path, self.MODELS[algorithm]))

        # Load the model
        self.prediction.loadModel()


    def predict(self, images_path, image_extensions = ['jpg'], n=5):
    """
    Perform image prediction on the given images.

    Parameters:
    - images_path: The path to the folder containing the uploaded images.
    - image_extensions: A list of image file extensions to consider.
    - n: The number of predictions to return for each image input.

    Returns:
    - predictions_data: A list of dictionaries containing the predictions for each image.
    """
        predictions_data = []

        # Iterate over each image extension
        for image_extension in image_extensions:
            # Iterate over each image file in the folder with the given extension
            for image in glob.glob(os.path.join(images_path, "*." + image_extension)):
                image_name = os.path.basename(image)
                predictions_per_image = {'image': image_name, 'predictions': []}

                # Perform image classification and get the predictions and probabilities
                predictions, probabilities = self.prediction.classifyImage(image, result_count = n)

                # Get prediction data for the image
                for eachPrediction, eachProbability in zip(predictions, probabilities):
                    predictions_per_image['predictions'].append({'label': eachPrediction, 'probability': round(eachProbability, 2)})

                # Add the predictions for the image to the overall predictions_data list
                predictions_data.append(predictions_per_image)

        # Return the predictions for all uploaded images
        return predictions_data
