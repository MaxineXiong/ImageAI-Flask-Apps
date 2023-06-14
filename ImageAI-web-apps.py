from flask import Flask, render_template, request
import os
from image_recognizer import ImageRecognizer
from video_object_detector import VideoObjectDetector
from moviepy.editor import VideoFileClip


# Creating a Flask application instance
app = Flask(__name__)


def intialize_files_folder():
"""
Function to initialize the files folder for storing uploaded files and output files.
It checks if the folder exists, and if not, creates it.
It also deletes any existing files inside the folder.
Returns the path to the folder.
"""
    # Create the './static/files/' folder if it doesn't exist
    if not os.path.exists('./static/files/'):
        os.mkdir('./static/files')

    # Path to the `files` folder
    path = os.path.join(os.path.join(os.getcwd(), 'static'), 'files')
    # Get the list of the names of the files inside the folder
    files = os.listdir(path)

    # Delete any existing files inside the folder
    if len(files) > 0:
        for file in files:
            os.remove(os.path.join(path, file))

    # Return the path to the `files` folder
    return path


# Home page
@app.route('/')
def home():
"""
Route for the home page.
Renders the index.html template.
"""
    return render_template('index.html')


# Image Prediction
@app.route('/image-prediction.html', methods = ['GET', 'POST'])
def predict_images():
"""
Route for image prediction.
Accepts both GET and POST requests.
If a POST request is received with image files, it saves the uploaded images to the `files` folder.
It then initializes the ImageRecognizer with the selected algorithm and predicts the objects on images.
Finally, it renders the image-prediction.html template with the predictions and selected algorithm.
"""
    if request.method == 'POST':
        # If a POST request is received with image files
        if request.files['images']:
            # Initialize the `files` folder for images
            images_path = intialize_files_folder()

            # Get the list of uploaded images
            uploaded_images = request.files.getlist('images')
            for uploaded_image in uploaded_images:
                # Save each image to the `files` folder
                uploaded_image.save(os.path.join(images_path, uploaded_image.filename))

            # Get the selected algorithm
            selected_algo = request.form['algorithm']

            # Path to the image prediction models
            execution_path = os.path.join(os.path.join(os.getcwd(), 'models'), 'image-prediction-models')
            # Create an instance of ImageRecognizer
            recognizer = ImageRecognizer(execution_path, selected_algo)

            # Predict the objects on images
            predictions = recognizer.predict(images_path = images_path,
                                             image_extensions = ['jpg', 'jpeg', 'png'],
                                             n = 5)

            # Renders the image-prediction.html template with the predictions and selected algorithm
            return render_template('image-prediction.html',
                                    predictions = predictions,
                                    algo = selected_algo)

    # Render the image-prediction.html template if the request method is GET
    return render_template('image-prediction.html')


# Video Object Detection
@app.route('/video-object-detection.html', methods = ['GET', 'POST'])
def detect_objects_in_video():
"""
Route for video object detection.
Accepts both GET and POST requests.
If a POST request is received with a video file, it saves the video to the files folder.
It then initializes the VideoObjectDetector with the selected model and performs object detection on the video.
Finally, it renders the video-object-detection.html template with the video file name and GIF image name of the output video, and CSV file name of object detection data by frames.
"""
    if request.method == 'POST':
        if request.files['video']:
            # Initialize the `files` folder for videos
            videos_path = intialize_files_folder()

            # Get the uploaded video
            uploaded_video = request.files['video']
            # Get the filename of the uploaded video
            input_video_name = uploaded_video.filename
            # Save the video to the `files` folder
            uploaded_video.save(os.path.join(videos_path, input_video_name))

            # Get the selected model
            selected_model = request.form['model']
            # Path to the video object detection models
            execution_path = os.path.join(os.path.join(os.getcwd(), 'models'), 'video-object-detection-models')
            # Set the frames per second for video processing
            frames_per_second = 20

            # Create an instance of VideoObjectDetector
            object_detector = VideoObjectDetector(execution_path, selected_model, frames_per_second, videos_path, input_video_name)
            # Save the detected object data by frames as a CSV file
            csv_path = object_detector.save_csv()
            # Plot summary bar charts
            object_detector.plot_summaries()

            # Generate a GIF image of the output video
            output_vido_clip = VideoFileClip(os.path.join(videos_path, input_video_name.split('.')[0] + "_detected." + input_video_name.split('.')[-1]))
            output_vido_clip.write_gif(os.path.join(videos_path, input_video_name.split('.')[0] + "_detected.gif"))

            # Check if the summary bar chart 'Average Number of Unique Objects Per Second' exists and assign its path, otherwise assign None
            if os.path.exists(os.path.join(videos_path, 'summary_plot_second.png')):
                second_plot =  'summary_plot_second.png'
            else:
                second_plot = None

            # Check if the summary bar chart 'Average Number of Unique Objects Per Minute' exists and assign its path, otherwise assign None
            if os.path.exists(os.path.join(videos_path, 'summary_plot_minute.png')):
                minute_plot =  'summary_plot_minute.png'
            else:
                minute_plot = None

            # Check if the summary bar chart 'Average Number of Unique Objects Per Hour' exists and assign its path, otherwise assign None
            if os.path.exists(os.path.join(videos_path, 'summary_plot_hour.png')):
                hour_plot =  'summary_plot_hour.png'
            else:
                hour_plot = None

            # Renders the video-object-detection.html template with the video details and CSV file name
            return render_template('video-object-detection.html',
                                    video_name = input_video_name,
                                    output_video_name = input_video_name.split('.')[0] + "_detected." + input_video_name.split('.')[-1],
                                    output_video_gif = input_video_name.split('.')[0] + "_detected.gif",
                                    csv_file_name = os.path.basename(csv_path),
                                    second_plot = second_plot,
                                    minute_plot = minute_plot,
                                    hour_plot = hour_plot)

    # Renders the video-object-detection.html template if the request method is GET
    return render_template('video-object-detection.html')




# Run the Flask app in debug mode if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
