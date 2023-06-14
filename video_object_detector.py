from imageai.Detection import VideoObjectDetection
import os
import pandas as pd
from datetime import timedelta, datetime, date
import matplotlib.pyplot as plt
import numpy as np


class VideoObjectDetector:
    MODELS = {'RetinaNet': 'retinanet_resnet50_fpn_coco-eeacb38b.pth', 'YOLOv3': 'yolov3.pt', 'TinyYOLOv3': 'tiny-yolov3.pt'}

    def __init__(self, execution_path, model, frames_per_second, videos_path, input_video_name):
    """
    Initialize the VideoObjectDetector object.

    Parameters:
    - execution_path: The path where the models are saved.
    - model: The selected model for object detection.
    - frames_per_second: Number of frames per second in the video.
    - videos_path: The path to the folder containing the uploaded videos.
    - input_video_name: The name of the uploaded video file.
    """
        self.frames_per_second = frames_per_second
        self.videos_path = videos_path
        self.input_video_name = input_video_name

        detector = VideoObjectDetection()

        # Set the model type based on the selected model
        if model == 'RetinaNet':
            detector.setModelTypeAsRetinaNet()
        if model == 'YOLOv3':
            detector.setModelTypeAsYOLOv3()
        if model == 'TinyYOLOv3':
            detector.setModelTypeAsTinyYOLOv3()

        # Set the model path based on the execution path and selected model
        detector.setModelPath(os.path.join(execution_path, self.MODELS[model]))

        # Load the model
        detector.loadModel()

        # Perform object detection and call forFull() function
        detector.detectObjectsFromVideo(
            input_file_path = os.path.join(self.videos_path, self.input_video_name),
            output_file_path = os.path.join(self.videos_path, self.input_video_name.split('.')[0] + "_detected"),
            frames_per_second = self.frames_per_second,
            video_complete_function = self.forFull,
            minimum_percentage_probability = 30)


    def forFull(self, output_arrays, count_arrays, average_output_count):
    """
    Process the output arrays and create a DataFrame with the detected objects.

    Parameters:
    - output_arrays: List of arrays containing information about each detected object in each frame.
    - count_arrays: List of arrays containing the count of unique objects in each frame.
    - average_output_count: Average count of unique objects across all frames.
    """
        today = date.today()
        current_frame = datetime(today.year, today.month, today.day, 0, 0, 0)
        rows = []

        for i in range(len(output_arrays)):
            # Objects in each frame
            objects_per_frame = output_arrays[i]
            # Number of unique objects in each frame
            objects_count = count_arrays[i]

            # If any objects are detected in a frame
            if len(objects_per_frame) > 0:
                for j in range(len(objects_per_frame)):
                    # For each unique object in each frame
                    object = objects_per_frame[j]
                    # Construct a row and append it to rows list
                    rows.append([current_frame.strftime('%H:%M:%S.%f'), object['name'], object['percentage_probability']/100])
            # If no object is detected in a frame
            else:
                # Construct a row and append it to rows list
                rows.append([current_frame.strftime('%H:%M:%S.%f'), None, None])

            # Move on to the next frame
            current_frame = current_frame + timedelta(seconds = (1/self.frames_per_second))

        # Construct a dataframe with headers of 'frames', 'objects' and 'probability'
        self.df = pd.DataFrame(rows, columns = ['frames', 'objects', 'probability'])


    def save_csv(self):
    """
    Save the DataFrame as a CSV file.

    Returns:
    - csv_path: The path to the saved CSV file.
    """
        # Path to CSV data file
        csv_path = os.path.join(self.videos_path, 'objects_detected_' + self.input_video_name.split('.')[0] + '.csv')
        # Save the dataframe as CSV file
        self.df.to_csv(csv_path, index = False)

        # Return the CSV file path
        return csv_path


    def plot_summary_graph(self, by_interval):
    """
    Plot and format a summary graph of the detected objects by frame / second / minute / hour / full video

    Parameters:
    - by_interval: The time interval to group the data for plotting (frame, second, minute, hour, full).
    """
        # Calculate the number of unique objects in each frame/second/minute/hour
        if by_interval != 'full':
            df_count = self.df.groupby([by_interval.lower() + 's', 'objects']).count()
            df_count.reset_index(drop = False, inplace = True)
            df_count.rename(columns = {'probability': 'uniqueCounts'}, inplace = True)
            # Calculate the average number of unique objects per frame/second/minute/hour
            count_col = 'uniqueCountsPer' + by_interval.title()
            df_count = df_count[['objects', 'uniqueCounts']].groupby('objects').mean().rename(columns = {'uniqueCounts': count_col})
            chart_title = 'Average Number of Unique Objects Per ' + by_interval.title()
        # Calculate the total number of unique objects across all frames
        else:
            count_col = 'counts'
            df_count = self.df[['objects', 'probability']].groupby('objects').count().rename(columns = {'probability': count_col})
            chart_title = 'Total Number of Unique Objects In This Video'

        # Sort df_count in descending order by count_col
        df_count.sort_values(count_col, ascending = False, inplace = True)
        df_count.reset_index(drop = False, inplace =True)

        # Turn df_count into a bar chart
        fig, ax = plt.subplots()
        df_count.plot.bar(x = 'objects', y = count_col,
                          color = '#7289DA', rot = 0, ax = ax)

        # Hide x axis title
        ax.set(xlabel = None)
        # Reset the range of y axis
        ax.set_ylim(0, max(list(df_count[count_col])) * 1.3)
        # Hide y axis
        ax.axes.get_yaxis().set_visible(False)
        # Format x ticks
        plt.xticks(fontname = 'arial', color = 'white')
        # Hide legend box
        ax.legend().set_visible(False)
        # Set the colors of the border lines
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('#1e2b3a')
        ax.spines['right'].set_color('#1e2b3a')
        ax.spines['top'].set_color('#1e2b3a')
        # Add graph title
        plt.title(chart_title,
                  fontname = 'arial', fontsize = 14, color = 'white')
        # Add value labels to the bars
        self.add_value_labels(ax)

        # Save the bar graph as a PNG image with a transparent background
        fig.savefig(os.path.join(self.videos_path, 'summary_plot_{}.png'.format(by_interval.lower())), transparent = True)


    def plot_summary_by_second(self):
    """
    Plot a summary graph by second.
    """
        # Add a 'seconds' column to df
        self.df['seconds'] = self.df['frames'].str.split(':').str[-1].str.split('.').str[0]
        # Convert 'seconds' column to integer type
        self.df['seconds'] = self.df['seconds'].astype('int32')
        # Plot summary graphy by second if the length of video exceeds 1 second
        if max(list(self.df['seconds'])) > 0:
            self.plot_summary_graph('second')


    def plot_summary_by_minute(self):
    """
    Plot a summary graph by minute.
    """
        # Add a 'minutes' column to df
        self.df['minutes'] = self.df['frames'].str.split(':').str[1]
        # Convert 'minutes' column to integer type
        self.df['minutes'] = self.df['minutes'].astype('int32')
        # Plot summary graphy by minutes if the length of video exceeds 1 minute
        if max(list(self.df['minutes'])) > 0:
            self.plot_summary_graph('minute')


    def plot_summary_by_hour(self):
    """
    Plot a summary graph by hour.
    """
        # Add an 'hours' column to df
        self.df['hours'] = self.df['frames'].str.split(':').str[0]
        # Convert 'hours' column to integer type
        self.df['hours'] = self.df['hours'].astype('int32')
        # Plot summary graphy by hours if the length of video exceeds 1 hour
        if max(list(self.df['hours'])) > 0:
            self.plot_summary_graph('hour')


    def plot_summaries(self):
    """
    Plot summary graphs for different intervals.
    """
        self.plot_summary_graph('frame')
        self.plot_summary_by_second()
        self.plot_summary_by_minute()
        self.plot_summary_by_hour()
        self.plot_summary_graph('full')


    def add_value_labels(self, ax, spacing = 5):
    """Add labels to the end of each bar in a vertical bar chart.

    Arguments:
    - ax (matplotlib.axes.Axes): The matplotlib object containing the axes of the plot to annotate.
    - spacing (int): The distance between the labels and the bars.
    """

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va,                      # Vertically align label differently for positive and negative values.
                color='white', fontsize = 14, weight = 'bold')  # Format the annotation text fonts
