

# Bone Fracture Detection

## Overview

The Bone Fracture Detection project aims to develop a machine learning model that can accurately identify fractures in bone X-ray images. This tool is designed to assist radiologists and healthcare professionals in diagnosing fractures more efficiently and accurately through an interactive web application built with Streamlit.



## Features

- **Automated Detection**: Uses Convolutional Neural Networks (CNN) to automatically detect bone fractures in X-ray images.
- **User-Friendly Interface**: Streamlit interface allows users to easily upload images and view results.
- **High Accuracy**: Leveraging advanced machine learning techniques for precise fracture detection.
- **Image Processing**: Incorporates Digital Image Processing (DIP) techniques to enhance image quality before analysis.
- **Visualization**: Displays highlighted fractured areas on the X-ray images along with confidence levels of predictions.

## Technologies Used

- **Python**: The primary programming language for development.
- **Google Streamlit**: For building the web application interface.
- **PyTorch**: To implement the Convolutional Neural Network (CNN) model.
- **scikit-learn**: For additional machine learning utilities and model evaluation.
- **Digital Image Processing (DIP)**: Techniques for preprocessing and enhancing X-ray images.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bone-fracture-detection.git
   cd bone-fracture-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py
```

Once the application is running, you can access it in your web browser at `http://localhost:8501`.

### How to Use the Application

1. **Upload X-ray Image**: Use the file uploader to select and upload an X-ray image.
2. **Check for Fractures**: Click the "Check" button to analyze the uploaded image.
3. **View Results**: The application will display the predicted fractured areas, if any, along with the confidence level of the prediction.

## Dataset

The model is trained on the [Bone Fracture Dataset](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data), which includes a variety of X-ray images with labeled fractures. Ensure that you have the necessary permissions to use this dataset.

## Training

To train the model, run the following command:

```bash
python train.py
```

This will start the training process using the specified parameters in `config.py`.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


