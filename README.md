# ImageAI Computer Vision Flask Apps

[![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&color=black&label)](https://github.com/MaxineXiong)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)](https://html.spec.whatwg.org/)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)](https://www.css3.com/)
[![ImageAI](https://img.shields.io/badge/ImageAI-5C3EE8?logo=OpenCV)](https://github.com/OlafenwaMoses/ImageAI)

<br/>

## Project Description

This project consists of two Flask applications that utilize **ImageAI's** [**image prediction algorithms**](https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Classification) and [**object detection models**](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md) to perform object recognition and analysis in images and videos. The first application, **Image Object Recognition Flask Application**, allows users to upload images and receive predictions about the objects present in them. The second application, **Video Object Detection Flask Application**, performs object detection in videos, generating frame-level data of detected objects and providing data analysis for the object detections.

<br/>

## Features

- **App 1 - Image Object Recognition Flask Application**:
    - Upload and predict objects in images.
    - Display predicted objects with confidence scores.
- **App 2 - Video Object Detection Flask Application**:
    - Upload and detect objects in videos.
    - Generate a raw dataset of detected objects at the frame level in CSV format.
    - Perform data analysis to provide insights on detected objects in the video, including:
        - *Average number of unique objects per frame*.
        - *Average number of unique objects per second*.
        - *Average number of unique objects per minute*.
        - *Average number of unique objects per hour.*
        - *Total number of unique objects in the entire video.*

<br/>

## Repository Structure

The repository has the following structure:

```
ImageAI-Flask-Apps/
├── ImageAI-web-app.py
├── image_recognizer.py
├── video_object_detector.py
├── templates/
│   ├── image-prediction.html
│   ├── index.html
│   └── video-object-detection.html
├── static/
│   ├── main.css
│   ├── images/
│   │   ├── demo-image-recognizer.gif
│   │   ├── demo-video-object-detector.gif
│   │   └── web-icon.ico
├── models/
│   ├── image-prediction-models/
│   │   └── README.md
│   ├── video-object-detection-models/
│   │   └── README.md
├── files_for_testing/
├── requirements.txt
├── README.md
└── LICENSE
```

The description of each file and folder in the repository is as follows:

- **ImageAI-web-app.py**:  This is the core driver file for the Flask apps. It handles the routing and functionality for the **Image Object Recognition** and **Video Object Detection** apps.
- **image_recognizer.py**: This file contains the code responsible for image object recognition. It utilizes **ImageAI's** [**image prediction algorithms**](https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Classification) to predict objects in uploaded images.
- **video_object_detector.py**: This file handles video object detection. It uses **ImageAI's** [**object detection models**](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md) to detect objects in uploaded videos and generates frame-level data of the detected objects.
- **templates/**: This folder contains the HTML templates used by the Flask apps. It includes three files:
    - **index.html**: The main HTML template that serves as the home page for the Flask apps. It provides navigation links to the **Image Object Recognition** and **Video Object Detection** apps.
    - **image-prediction.html**: The HTML template for the **Image Object Recognition** app, which allows users to upload and predict objects in images.
    - **video-object-detection.html**: The HTML template for the **Video Object Detection** app, which enables users to upload and detect objects in videos.
- **static/**: This folder contains static files used by the Flask apps, such as CSS stylesheets and images. It includes the following:
    - **main.css**: The CSS styling file that defines the appearance and layout of the HTML templates.
    - **images/**: A subfolder that stores various images used in the project, including demo GIFs (**demo-image-recognizer.gif** and **demo-video-object-detector.gif**) and a favicon (**web-icon.ico**).
- **models/**: This folder is meant to store ImageAI's models for image classification and object detection. However, it currently contains two empty subfolders:
    - **image-prediction-models/**: An empty subfolder where users should manually place **ImageAI's** [**image prediction models**](https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Classification). Instructions for downloading and saving the models are provided in the **README.md** file inside this subfolder.
    - **video-object-detection-models/**: Another empty subfolder where users should manually add **ImageAI's** [**object detection models**](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md). Instructions for downloading and saving the models are provided in the **README.md** file inside this subfolder.
    
    **Note:** The **models/** folder is intentionally left empty in the GitHub repository due to the large size of the models. Users are required to manually download and place the models into the corresponding subfolders as described in the README files.
    
- **files_for_testing/**: This folder contains a collection test images and videos that can be used for app testing and experimentation.
- **requirements.txt**: This file lists the necessary dependencies and packages required to run the Flask apps. It provides a convenient way to install all the dependencies.
- **README.md**: The README file for the project. It provides an overview of the project, installation instructions, usage guidelines, and other relevant information.
- **LICENSE**: The license file for the project, which specifies the terms and conditions under which the code and resources are shared.

<br/>

## Usage

To use the **ImageAI Computer Vision Flask Apps**, please follow these steps:

1. Install the required dependencies by running the following command to install all the necessary packages and dependencies required to run the Flask apps:
    
    ```
    pip install -r requirements.txt
    ```
    
    This command will ensure that all the required libraries and dependencies are installed in your Python environment.
    
2. Download the image classification and object detection models:
    - Inside the **models/image-prediction-models/** subfolder, follow the instructions provided in the **README.md** file to manually download and save the **ImageAI's** [**image prediction models**](https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Classification) in the current directory.
    - Similarly, inside the **models/video-object-detection-models/** subfolder, follow the instructions in the **README.md** file to manually download and save **ImageAI's** [**object detection models**](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md) in the current directory.
3. Once the dependencies are installed and the models are added, open the command prompt or terminal and navigate to the project directory.
4. Run the following command to start the Flask apps:
    
    ```
    python ImageAI-web-app.py
    ```
    
5. After running the above command, access the Flask apps by visiting **`http://127.0.0.1:5000/`** in your web browser. This will take you to the home page of the Flask apps.
6. From the home page, you can navigate to the **Image Object Recognition Flask Application** or the **Video Object Detection Flask Application** to perform the respective tasks.
    - In the **Image Object Recognition Flask Application**, you can upload images and predict objects present in them. The predicted objects will be displayed along with their confidence scores.
                
        <p align='center'>
          <img src="./static/images/demo-image-recognizer.gif">
          <br><i>This GIF demonstrates the process of uploading an image, predicting objects in the image, and displaying the predicted objects with confidence scores.</i>
        </p>
        
    - In the **Video Object Detection Flask Application**, you can upload videos and detect objects in them. The app will generate raw frame-level object data in CSV format that you can download. Additionally, it performs data analysis to provide insights on the detected objects in the video, such as the *average number of unique objects per frame*, *per second*, *per minute, per hour*, and *the total number of unique objects in the entire video*.
        
        <p align='center'>
          <img src="./static/images/demo-video-object-detector.gif">
          <br><i>This GIF showcases the steps of uploading a video, detecting objects in the video, generating and downloading frame-level object data in CSV format, and performing data analysis on the detected objects to provide insights.</i>
        </p>
       
**Note**: The **files_for_testing** folder contains test images and videos that you can use to explore and test the functionalities of the Flask apps. Feel free to experiment with these test files to familiarize yourself with the Flask apps. Alternatively, you can also use your own images or videos when working with the apps in a production environment.

Make sure to follow these instructions in the given sequence to ensure the proper setup and usage of the ImageAI Flask Apps.

<br/>

## Contribution

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

<br/>

## License

This project is licensed under the **[MIT License](https://choosealicense.com/licenses/mit/)**.

<br/>

## Acknowledgement

I would like to thank the [**creator of ImageAI**](https://github.com/OlafenwaMoses) for providing the image prediction and object detection algorithms used in this project. His contributions have been instrumental in the development of these Flask applications. To delve deeper into the computer vision algorithms and models developed by [**ImageAI**](https://github.com/OlafenwaMoses/ImageAI), I encourage you to visit the [**ImageAI GitHub Page**](https://github.com/OlafenwaMoses/ImageAI) and explore its [**official documentation**](https://imageai.readthedocs.io/en/latest/).
