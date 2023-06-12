from flask import Flask, render_template, request
import os
from image_recognizer import ImageRecognizer
from video_object_detector import VideoObjectDetector
from moviepy.editor import VideoFileClip



app = Flask(__name__)


def intialize_files_folder():
    if not os.path.exists('./static/files/'):
        os.mkdir('./static/files')

    path = os.path.join(os.path.join(os.getcwd(), 'static'), 'files')
    files = os.listdir(path)

    if len(files) > 0:
        for file in files:
            os.remove(os.path.join(path, file))

    return path


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Image Prediction
@app.route('/image-prediction.html', methods = ['GET', 'POST'])
def predict_images():
    if request.method == 'POST':
        if request.files['images']:
            images_path = intialize_files_folder()

            uploaded_images = request.files.getlist('images')
            for uploaded_image in uploaded_images:
                uploaded_image.save(os.path.join(images_path, uploaded_image.filename))

            selected_algo = request.form['algorithm']

            execution_path = os.path.join(os.path.join(os.getcwd(), 'models'), 'image-prediction-models')
            recognizer = ImageRecognizer(execution_path, selected_algo)

            predictions = recognizer.predict(images_path = images_path,
                                             image_extensions = ['jpg', 'jpeg', 'png'],
                                             n = 5)

            return render_template('image-prediction.html',
                                    predictions = predictions,
                                    algo = selected_algo)

    return render_template('image-prediction.html')


# Video Object Detection
@app.route('/video-object-detection.html', methods = ['GET', 'POST'])
def detect_objects_in_video():
    if request.method == 'POST':
        if request.files['video']:
            videos_path = intialize_files_folder()

            uploaded_video = request.files['video']
            input_video_name = uploaded_video.filename
            uploaded_video.save(os.path.join(videos_path, input_video_name))

            selected_model = request.form['model']
            execution_path = os.path.join(os.path.join(os.getcwd(), 'models'), 'video-object-detection-models')
            frames_per_second = 20

            object_detector = VideoObjectDetector(execution_path, selected_model, frames_per_second, videos_path, input_video_name)
            # Save data as csv file
            csv_path = object_detector.save_csv()
            # Plot summary bar chart
            summary_plot_path = object_detector.plot_summary_graph()

            output_vido_clip = VideoFileClip(os.path.join(videos_path, input_video_name.split('.')[0] + "_detected." + input_video_name.split('.')[-1]))
            output_vido_clip.write_gif(os.path.join(videos_path, input_video_name.split('.')[0] + "_detected.gif"))

            return render_template('video-object-detection.html',
                                    video_name = input_video_name,
                                    output_video_name = input_video_name.split('.')[0] + "_detected." + input_video_name.split('.')[-1],
                                    output_video_gif = input_video_name.split('.')[0] + "_detected.gif",
                                    csv_file_name = os.path.basename(csv_path))

    return render_template('video-object-detection.html')





if __name__ == '__main__':
    app.run(debug=True)
