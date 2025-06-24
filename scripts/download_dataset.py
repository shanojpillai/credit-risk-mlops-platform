#!/usr/bin/env python3
"""
Download Credit Card Default Dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
"""

import sys
import pandas as pd
import requests
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url: str, destination: str) -> bool:
    """Download file from URL to destination"""
    try:
        logger.info(f"Downloading from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded to {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False

def download_credit_card_dataset():
    """Download the credit card default dataset"""
    
    # Create data directories (following README structure)
    raw_data_dir = Path("mlflow/artifacts/datasets")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset information
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    dataset_file = raw_data_dir / "default_credit_card_clients.xls"
    
    logger.info("Starting Credit Card Default Dataset download...")
    
    # Download the dataset
    if download_file(dataset_url, str(dataset_file)):
        logger.info("Dataset downloaded successfully!")
        
        # Convert XLS to CSV for easier processing
        try:
            logger.info("Converting XLS to CSV...")
            df = pd.read_excel(dataset_file, header=1)  # Skip first row which contains metadata
            
            # Save as CSV
            csv_file = raw_data_dir / "default_credit_card_clients.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Converted to CSV: {csv_file}")
            
            # Display basic info about the dataset
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Target variable distribution:")
            if 'default payment next month' in df.columns:
                logger.info(df['default payment next month'].value_counts())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert XLS to CSV: {e}")
            return False
    else:
        logger.error("Failed to download dataset")
        return False

def create_dataset_info():
    """Create dataset information file"""
    
    dataset_info = """
# Credit Card Default Dataset

## Source
- **Dataset**: Default of Credit Card Clients Dataset
- **Original Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

## Description
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

## Features
- **ID**: ID of each client
- **LIMIT_BAL**: Amount of given credit in NT dollars
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: Education level (1=graduate school, 2=university, 3=high school, 4=others)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0 to PAY_6**: Repayment status in September 2005 to April 2005
- **BILL_AMT1 to BILL_AMT6**: Amount of bill statement (NT dollar)
- **PAY_AMT1 to PAY_AMT6**: Amount of previous payment (NT dollar)
- **default payment next month**: Default payment (1=yes, 0=no) - TARGET VARIABLE

## Dataset Statistics
- **Total Records**: 30,000
- **Features**: 23 + 1 target variable
- **Target Distribution**: Approximately 22% default rate
- **Missing Values**: None

## Use Case
This dataset is perfect for:
- Credit risk modeling
- Default prediction
- Customer segmentation
- Feature engineering experiments
- MLflow experiment tracking
"""
    
    info_file = Path("mlflow/artifacts/datasets/dataset_info.md")
    with open(info_file, 'w') as f:
        f.write(dataset_info)
    
    logger.info(f"Dataset information saved to {info_file}")

def main():
    """Main function to download and setup the dataset"""
    
    logger.info("=== Credit Card Default Dataset Setup ===")
    
    # Download the dataset
    if download_credit_card_dataset():
        # Create dataset information
        create_dataset_info()
        
        logger.info("✅ Dataset setup completed successfully!")
        logger.info("📁 Files created:")
        logger.info("   - mlflow/artifacts/datasets/default_credit_card_clients.xls (original)")
        logger.info("   - mlflow/artifacts/datasets/default_credit_card_clients.csv (processed)")
        logger.info("   - mlflow/artifacts/datasets/dataset_info.md (documentation)")
        
        return True
    else:
        logger.error("❌ Dataset setup failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
