from imageai.Classification import ImageClassification
import os
import glob


class ImageRecognizer:
    MODELS = {'ResNet50': 'resnet50-19c8e357.pth', 'MobileNetV2': 'mobilenet_v2-b0353104.pth',
              'InceptionV3': 'inception_v3_google-1a9a5a14.pth', 'DenseNet121': 'densenet121-a639ec97.pth'}

    # execution_path: where the models are saved
    def __init__(self, execution_path, algorithm):
        self.prediction = ImageClassification()
        if algorithm == 'ResNet50':
            self.prediction.setModelTypeAsResNet50()
        if algorithm == 'MobileNetV2':
            self.prediction.setModelTypeAsMobileNetV2()
        if algorithm == 'InceptionV3':
            self.prediction.setModelTypeAsInceptionV3()
        if algorithm == 'DenseNet121':
            self.prediction.setModelTypeAsDenseNet121()
        self.prediction.setModelPath(os.path.join(execution_path, self.MODELS[algorithm]))
        self.prediction.loadModel()


    def predict(self, images_path, image_extensions = ['jpg'], n=5):
        predictions_data = []
        for image_extension in image_extensions:
            for image in glob.glob(os.path.join(images_path, "*." + image_extension)):
                image_name = os.path.basename(image)
                predictions_per_image = {'image': image_name, 'predictions': []}
                # result_count = n: return n predictions for each image input
                predictions, probabilities = self.prediction.classifyImage(image, result_count = n)
                for eachPrediction, eachProbability in zip(predictions, probabilities):
                    predictions_per_image['predictions'].append({'label': eachPrediction, 'probability': round(eachProbability, 2)})

                predictions_data.append(predictions_per_image)

        return predictions_data
