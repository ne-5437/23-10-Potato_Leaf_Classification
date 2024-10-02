# Potato_Leaf_Disease_Classification
AI-powered potato leaf disease classification using TensorFlow (Keras API), seamlessly deployed with FastAPI and Uvicorn. Tested via Postman and paired with a sleek React/Node.js web interface for real-time results.

This project focuses on classifying potato leaf diseases using a deep learning model built with TensorFlow and Keras. The model distinguishes between healthy leaves, early blight, and late blight. The system was trained, deployed, and tested using various tools and technologies, including FastAPI, Uvicorn, and a React/Node.js web interface for real-time user interaction.

## 01. Data Collection

### Data Source
We used a publicly available dataset from Kaggle. This dataset was chosen due to its ease of access and pre-labeled nature, which allowed us to bypass the time-consuming task of manual data collection and labeling.

### Dataset Composition
The dataset contains over 3,000 images of potato leaves, categorized into three classes:
- **Healthy:** Leaves with no signs of disease.
- **Early Blight:** Leaves infected by early blight, caused by *Alternaria solani*.
- **Late Blight:** Leaves infected by late blight, caused by *Phytophthora infestans*.

### Dataset Structure
The dataset is organized into separate folders, one for each class:
- data/Healthy/`: Images of healthy leaves.
- data/Early_Blight/`: Images of leaves with early blight.
- data/Late_Blight/`: Images of leaves with late blight.

---

## 02. Data Modelling

### Platform
- **Development Platform:** Jupyter Notebook
- **Framework:** TensorFlow with Keras API
- **Language:** Python 3.x

### Libraries Used
- **TensorFlow:** For building and training the neural network.
- **Keras:** For managing the CNN architecture.
- **NumPy:** For data manipulation.
- **Matplotlib:** For visualizing training results.
- **OS and Shutil:** For managing dataset directories.

### Step 1: Dataset Preparation
- **Dataset Source:** The PlantVillage dataset containing potato leaf images in three categories (Early Blight, Late Blight, Healthy).
- **Total Images:** Approximately 3,000 images.
  - **Training Set:** 70% (~2,100 images)
  - **Validation Set:** 15% (~450 images)
  - **Test Set:** 15% (~450 images)

### Step 2: Dataset Loading and Preprocessing
- **Directory Structure:** The images are organized into subfolders based on disease categories.
- **Image Loading:** Using image_dataset_from_directory()`, images were loaded, resized to 256x256 pixels, batched, and labeled using categorical encoding.
- **Data Augmentation:** Random horizontal flips and rotations were applied to augment the dataset.
- **Rescaling:** Images were rescaled from pixel values of [0, 255] to [0, 1]` for faster convergence.

### Step 3: Model Architecture
- We built a **Convolutional Neural Network (CNN)** consisting of:
  - Convolutional layers for spatial feature extraction.
  - MaxPooling layers for downsampling.
  - Flattening layer to convert feature maps into a 1D vector.
  - Dense layers with ReLU activation.
  - Output layer with softmax activation for multi-class classification.

### Step 4: Model Compilation
- **Optimizer:** Adam optimizer with a learning rate of 0.001`.
- **Loss Function:** Categorical crossentropy for multi-class classification.
- **Metrics:** Accuracy was used to measure model performance.

### Step 5: Training the Model
- **Epochs:** 15
- **Batch Size:** 32
- **Validation Split:** 15% of the data was used for validation.
- The model was trained using the fit()` method.

### Step 6: Evaluation
- The model was evaluated on the test set, achieving a **test accuracy of 98.9%**, demonstrating strong generalization on unseen data.

---

## 03. Deploying the Model

### Step 1: Install Required Python Dependencies
- **Dependencies:**
  - FastAPI`: Modern web framework for building APIs.
  - Uvicorn`: ASGI server to run FastAPI applications.
  - Python-Multipart`: For handling file uploads.
  - Pillow`: Python Imaging Library for image processing.
  - TensorFlow-Serving-API`: For serving TensorFlow models via RESTful APIs.
  - Matplotlib` & `NumPy`: For data visualization and numerical operations.

### Step 2: Import FastAPI
Create a FastAPI application and import necessary libraries.

### Step 3: Set Up the API Endpoint
- We defined a `/predict` endpoint that accepts image uploads via a POST request. The endpoint processes the image and returns the predicted class (Healthy, Early Blight, Late Blight).

### Step 4: Start the FastAPI Application
- We used **Uvicorn** to run the FastAPI application.

### Step 5: Testing the API
- Initial testing was done using PyCharm, followed by Postman to ensure proper API functionality and documentation. The POST request URL was set to `http://localhost:8000/predict`.

---

## 04. Hosting a Webpage

### Step 1: Set Up React Application
We used **Node.js** and **npm** to set up a React application for the front-end.

### Step 2: Create Components
- **App.js:** The main functional component rendering the image upload interface.
- **Home.js:** Handles image upload functionality and returns an HTML fragment to `App.js`.

### Step 3: Implement Drag and Drop Feature
- Integrated **Material-UI** to add drag-and-drop functionality for image uploads.

### Step 4: Handle File Selection
- The **useState** hook was used to manage the selected image file before uploading.

### Step 5: Send File to Backend using Axios
- Used **Axios** to send a POST request to the FastAPI backend, with the image file encapsulated in `FormData`.

### Step 6: Integrate with Postman for Testing
- Postman was used to verify that the React frontend communicates with the FastAPI backend correctly by sending requests to the API.

### Step 7: CORS Policy Handling
- To resolve **CORS** issues between the frontend (port 3000) and the backend (port 8000), we added middleware to the FastAPI application:



