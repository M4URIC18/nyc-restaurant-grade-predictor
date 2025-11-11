# 🍽 NYC Restaurant Grade Predictor

A Streamlit web application that predicts restaurant inspection grades (A, B, C, P, Z) using a trained RandomForestClassifier machine learning model.

## Features

- 🍽️ **Grade Prediction**: Predict restaurant inspection grades based on various features
- 📊 **Visualizations**: Interactive charts showing prediction probabilities and feature importances
- 🎨 **Color-coded Grades**: Visual grade display with colored boxes (Green for A, Yellow for B, Orange for C, Red for Z, Gray for P)
- 📈 **Feature Importance**: Horizontal bar chart showing which features most influence predictions using `model.named_steps['randomforestclassifier'].feature_importances_`
- 🗺️ **Interactive Google Maps**: Click on map locations or search addresses to automatically predict restaurant grades

## Requirements

- Python 3.8 or higher
- Trained model file (`baseline_model.pkl` or `model.pkl`)
- Google Maps API key (optional, for map features)

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your model file**:
   - Ensure your trained model file (`baseline_model.pkl` or `model.pkl`) is in the project root directory or in an `app/` folder
   - The app will automatically search for the model in common locations

## Running the Application

Run the app from the project root directory:

```bash
streamlit run app/main.py
```

The application will open in your default web browser at `http://localhost:8501`

## Google Maps API Setup (Optional)

To enable interactive map features:

1. **Get a Google Maps API Key**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the **Geocoding API** and **Maps JavaScript API**
   - Create credentials (API Key)
   - Restrict the API key for security (recommended)

2. **Set the API Key**:
   - **Option A**: Set as environment variable (recommended)
     ```bash
     export GOOGLE_MAPS_API_KEY="your-api-key-here"
     ```
   - **Option B**: Enter it in the app's sidebar (less secure, but convenient for testing)

## Usage

1. **Fill in the input parameters** in the sidebar:
   - **Borough** (dropdown): Select from Manhattan, Brooklyn, Queens, Bronx, Staten Island
   - **Inspection Type** (dropdown): Select the type of inspection
   - **Cuisine Description** (text input): Enter the type of cuisine (e.g., American, Italian, Chinese)
   - **NFH Index Score** (number): Neighborhood Food Hazard Index Score (0-100)
   - **Median Income** (number): Median income of the area
   - **Poverty Rate** (number): Poverty rate percentage (0-100)

2. **Click "Predict Grade"** to get the prediction

3. **View the results**:
   - Predicted grade with color-coded display (Green for A, Yellow for B, Orange for C, Red for Z, Gray for P)
   - Probability distribution chart
   - Feature importance chart (horizontal bar chart) showing top features that influence the prediction

### Using the Interactive Map (Optional)

If you have a Google Maps API key:

1. **Search for a restaurant location**:
   - Enter the restaurant address in the sidebar search box
   - Click "Search on Map" to locate it on the map
   - The borough will be automatically filled from the address

2. **Click on the map**:
   - Click anywhere on the map to select a restaurant location
   - The address will be automatically retrieved
   - The borough will be auto-filled if detected

3. **Make a prediction**:
   - Fill in the remaining input parameters
   - Click "Predict Grade"
   - The map marker will be updated with the predicted grade
   - Click the marker to see the address and predicted grade in a popup

## Model Input Features

The model expects the following features:
- `borough`: NYC borough (dropdown selection)
- `cuisine_description`: Type of cuisine (text input)
- `inspection_type`: Type of inspection (dropdown selection)
- `nfh_indexscore`: Neighborhood Food Hazard Index Score (0-100)
- `median_income`: Median income of the area
- `poverty_rate`: Poverty rate percentage (0-100)

## Grade Meanings

- **A** (Green): Excellent (Score: 0-13)
- **B** (Yellow): Good (Score: 14-27)
- **C** (Orange): Acceptable (Score: 28+)
- **P** (Gray): Pending (Issued when restaurant is re-inspected)
- **Z** (Red): Grade Pending (Issued when restaurant requires re-inspection)

## Project Structure

```
Project Folder/
├── app/
│   ├── main.py            # Streamlit application
│   └── utils.py           # Utility functions
├── baseline_model.pkl     # Trained model file (can be in app/ or root)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Troubleshooting

### Model not found
- Ensure `baseline_model.pkl` or `model.pkl` exists in the project root or `app/` folder
- Check the file name matches exactly (case-sensitive)

### Prediction errors
- Ensure all required fields are filled
- Check that numerical inputs are within valid ranges
- Verify the model file is not corrupted

## Dependencies

- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- plotly >= 5.14.0
- scikit-learn >= 1.3.0
- folium >= 0.14.0
- streamlit-folium >= 0.15.0
- requests >= 2.31.0

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues or questions, please check the troubleshooting section above or review the code comments for additional guidance.

