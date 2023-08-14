import os
import time
import cv2
import numpy as np
import pandas as pd
from createFeatures import extract_features
from colorMoments import calculate_color_moment
import tkinter as tk
from tkinter import filedialog, Scrollbar, messagebox
from PIL import Image, ImageTk
from similarity import euclidean, cosine

df = pd.DataFrame([])
query_image = None
query_image_file_path = None
times = 1
all_images_similarity = []


def calculate_using_color_moment(db_cm, q_cm, index):
    d_mom = calculate_color_moment(db_cm, q_cm)  # calculated Color Moment
    all_images_similarity.append([d_mom, index])
    return


def calculate_using_lbp_euclidean(db_image_lbp_fv, q_lbp_feature_vector, index):
    euclidean_distance = euclidean(q_lbp_feature_vector, db_image_lbp_fv)  # calculated LBPs Euclidean distance
    all_images_similarity.append([euclidean_distance, index])
    return


def calculate_using_lbp_cosine(db_image_lbp_fv, q_lbp_feature_vector, index):
    cosine_similarity = cosine(q_lbp_feature_vector, db_image_lbp_fv)  # calculated LBPs Euclidean distance
    all_images_similarity.append([cosine_similarity, index])
    return


def calculate_using_both(db_cm, q_cm, db_image_lbp_fv, q_lbp_feature_vector, index):
    d_mom = calculate_color_moment(db_cm, q_cm)  # calculated Color Moment

    euclidean_distance = euclidean(q_lbp_feature_vector, db_image_lbp_fv)  # calculated LBPs Euclidean distance

    overall_similarity = 0.8 * d_mom + 0.2 * euclidean_distance  # Combining the features of Color Moments and LBPs

    all_images_similarity.append([overall_similarity, index])
    return


def retrieve_similar_images():
    if query_image_file_path is None:
        return []

    # query image
    q_image_rgb = cv2.imread(query_image_file_path)

    # (feature vector for LBPs, feature vector for Colour Moments) of query image
    q_lbp_feature_vector, q_cm_feature_vector = extract_features(q_image_rgb)

    # both the feature vectors are concatenated to form a single feature vector for the query image
    q_feature_vector_combined = np.concatenate((q_lbp_feature_vector, q_cm_feature_vector), axis=0)

    st_time = time.time()

    print("Shape of df: ", df.shape)

    # Clears the previous query results (if any)
    all_images_similarity.clear()

    for i in range(len(df)):
        db_image_lbp_fv = df.iloc[i][1:257]
        db_image_cm_fv = df.iloc[i][257:]
        db_cm = db_image_cm_fv.values.reshape(3, 3)
        q_cm = np.array(q_cm_feature_vector).reshape(3, 3)

        # if we want to calculate the similarity using Colour Moments only
        # calculate_using_color_moment(db_cm, q_cm, df.iloc[i][0])

        # if we want to calculate the similarity using LBPs and Euclidean only
        # calculate_using_lbp_euclidean(db_image_lbp_fv, q_lbp_feature_vector, df.iloc[i][0])
        #
        # # if we want to calculate the similarity using LBPs and Euclidean only
        # calculate_using_lbp_cosine(db_image_lbp_fv, q_lbp_feature_vector, df.iloc[i][0])
        #
        # if we want to calculate the similarity using both LBP and Colour Moments
        calculate_using_both(db_cm, q_cm, db_image_lbp_fv, q_lbp_feature_vector, df.iloc[i][0])

        print("Done with : ", df.iloc[i][0], " with distance: ", all_images_similarity[i])

    # sorts in increasing order using the first column
    all_images_similarity.sort(key=lambda x: x[0])

    # sorts in decreasing order using the first column
    # all_images_similarity.sort(key=lambda x: x[0], reverse=True)

    output_feature_vectors = all_images_similarity[:10]

    end_time = time.time()

    print(output_feature_vectors)
    print("Time taken ", end_time - st_time)

    return output_feature_vectors


def update_similar_images():
    if query_image is None:
        messagebox.showinfo("Error", "Please select an query image before fetching")
        return

    # Retrieve similar images for query image
    # Returned value is of the form [[dist1, 50.0], [dist2, 405.0], ....]
    similar_images = retrieve_similar_images()

    query_canvas.delete('all')
    similar_canvas.delete('all')

    # Display query image in canvas
    query_canvas.create_image(0, 0, anchor='nw', image=query_image)
    # Display similar images in canvas
    image_size = (150, 150)
    spacing = 20

    similar_canvas.image_references = []  # Initialize list for image references

    for i, image in enumerate(similar_images):
        if i % 5 != 0:
            x += 150 + 20
            y = (i // 5) * (150 + spacing) + 20
        if i % 5 == 0:
            x = 200
            y = (i // 5) * (150 + spacing) + 20

        similar_image_name = str(image[1]).split('.')
        image = cv2.imread(os.path.join('images', f"{similar_image_name[0]}.jpg"))
        # Resize image for display
        image_resized = cv2.resize(image, image_size)
        # Convert image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        # Convert image to PhotoImage format
        image_tk = ImageTk.PhotoImage(image_pil)
        # Display image in canvas
        similar_canvas.create_image(x, y, anchor='nw', image=image_tk)

        similar_canvas.image_references.append(image_tk)


# Define function to update image in GUI window
def update_image(image_path):
    global query_image
    # Load image using OpenCV
    image = cv2.imread(image_path)
    # Resize image for display
    image_resized = cv2.resize(image, (150, 150))
    # Convert image to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    # Convert image to PhotoImage format
    query_image = ImageTk.PhotoImage(image_pil)
    # Display image in GUI
    query_canvas.create_image(0, 0, anchor='nw', image=query_image)


# Define function to handle button click event
def browse_image():
    # Open file dialog to select query image
    global query_image_file_path
    file_path = filedialog.askopenfilename()

    if file_path is None or len(file_path) == 0:
        return

    query_image_file_path = file_path
    update_image(file_path)


def load_database_features():
    s_time = time.time()
    global df
    df = pd.read_csv('database_features.csv')
    e_time = time.time()  # 10 minutes

    print("Reading CSV file completed in: ", e_time - s_time)


# Define function to create GUI window
def create_window():
    # Create window
    window = tk.Tk()

    # Set window title and size
    window.title('Content-Based Image Retrieval')
    print(window.winfo_screenwidth(), window.winfo_screenheight())
    window.geometry('1000x1000')

    # Create a canvas widget
    canvas = tk.Canvas(window, width=1000, height=1000, bg="white")

    # Create a scrollbar widget
    scrollbar = Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)

    # Set the scrollbar to control the y-axis of the canvas
    canvas.config(yscrollcommand=scrollbar.set)

    # Create a frame to contain the contents of the canvas
    frame = tk.Frame(canvas, bg="white", width=1000, height=1000)

    # canvas.create_window((0, 0), window=frame, anchor='center', tags='frame')
    # Add canvas to display images
    global query_canvas, query_canvas_images, similar_canvas, similar_canvas_images

    heading_label = tk.Label(frame, text="CONTENT BASED IMAGE RETRIEVAL", bg="white", font=('Arial', 20))
    heading_label.pack(side=tk.TOP, pady=10)

    # Add button to browse for query image
    button_browse = tk.Button(frame, text='Browse image', command=browse_image, padx=10, pady=5)
    button_browse.pack(side=tk.TOP, padx=10, pady=10)

    query_image_label = tk.Label(frame, text="Query Image", bg="white", font=('Arial', 14))
    query_image_label.pack(side=tk.TOP)

    query_canvas = tk.Canvas(frame, width=150, height=150, bg="white", highlightthickness=0)
    query_canvas.pack(side=tk.TOP, padx=50, pady=10)

    button_fetch = tk.Button(frame, text='Fetch results', command=update_similar_images, padx=10, pady=5)
    button_fetch.pack(side=tk.TOP, padx=10, pady=10)

    similar_images_label = tk.Label(frame, text="Retrieved Similar Images", bg="white", font=('Arial', 18))
    similar_images_label.pack(side=tk.TOP, pady=10)

    similar_canvas = tk.Canvas(frame, width=window.winfo_screenwidth() - 200, height=window.winfo_screenheight() / 1.5,
                               bg="white", highlightthickness=0)
    similar_canvas.pack(side=tk.LEFT, padx=50)

    # Place the frame inside the canvas
    canvas.create_window((0, 0), window=frame, anchor='nw')

    # Configure the canvas to resize along with the window
    frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

    # Pack the canvas and scrollbar widgets
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    global times
    if times == 1:
        load_database_features()
        times -= 1

    # Run main loop for GUI window
    window.mainloop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_window()

