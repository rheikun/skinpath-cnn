# SkinPath-CNN

SkinPath-CNN is an AI-based application that uses Convolutional Neural Network (CNN) models to detect skin types from images uploaded by users. The application provides recommendations for skincare products based on the analysis results.

## Key Features

- **Skin Type Detection**: 
  Upload a photo or take a picture directly to analyze your skin type. The application will identify whether your skin is dry, oily, or normal.

- **Skincare Product Recommendations**: 
  Once the skin type is detected, the application provides suggestions for suitable products and information about ingredients to avoid or use, based on the skincare goals selected by the user.

- **Ingredient Explanations**: 
  For each ingredient mentioned in the recommendations, the application provides detailed explanations regarding its benefits and potential side effects.

## Dataset Source

This application uses a dataset containing images and information about various skin types, allowing for accurate and relevant analysis.

## Disclaimer

The skin type analysis results are for reference only. Other variables, such as skin sensitivity, cannot be measured by this application. We recommend consulting a healthcare professional or dermatologist before making skincare decisions.

## Interface Preview

If you'd like to see how the application works, you can check the link in [url.txt](url.txt) and access it from your browser.
## Setup Environment - Anaconda
```
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```
mkdir project_cnn
cd project_cnn
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run streamlit app
```
streamlit run app.py
```
