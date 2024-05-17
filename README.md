# Copper Modelling Industrial

## Introduction

This repository contains the Copper Modelling Industrial project, which includes data preprocessing, model training, and a Streamlit application for predicting the status and selling price of copper products. The project uses a classifier model to predict the status and a regressor model to predict the selling price.

## Features

- **Status Prediction**: Predicts the status of copper products using a classifier model.
- **Selling Price Prediction**: Predicts the selling price of copper products using a regressor model.
- **File Upload**: Supports CSV file uploads for batch predictions.
- **Visualization**: Offers various plot types to visualize predictions.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/yourusername/copper-modelling-industrial.git
cd copper-modelling-industrial
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

1. **Start the Streamlit application:**

    ```bash
    streamlit run Home.py
    ```

2. **Open your browser and navigate to:**

    ```
    http://localhost:8501
    ```

## Dependencies

The project relies on the following Python packages:

- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter
- nbconvert

These can be installed using the `requirements.txt` file provided in the repository:

```bash
pip install -r requirements.txt
```

## Executing the Jupyter Notebook

If the model files are not present, the Streamlit app will attempt to generate them by executing the Jupyter Notebook:

```bash
jupyter nbconvert --to notebook --execute Copper_modelling.ipynb
```

## Example Input Parameters

For manual input, you will need to provide the following parameters:

- Customer ID
- Country
- Status
- Item Type
- Application
- Width
- Product Ref
- Quantity Tons
- Thickness
- Selling Price
- Item Date Day
- Item Date Month
- Item Date Year
- Delivery Date Day
- Delivery Date Month
- Delivery Date Year

## Models

- `status_pred_model.pkl`: Classifier model for predicting the status.
- `selling_price_prediction.pkl`: Regressor model for predicting the selling price.

## Data

The input data should be in CSV format with the following columns:

- `customer`
- `country`
- `status`
- `item type`
- `application`
- `width`
- `product_ref`
- `quantity_tons`
- `thickness`
- `selling_price`
- `item_date_day`
- `item_date_month`
- `item_date_year`
- `delivery_date_day`
- `delivery_date_month`
- `delivery_date_year`

