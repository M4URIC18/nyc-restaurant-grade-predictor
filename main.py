"""
Streamlit web app for Restaurant Inspection Grade Prediction.

This app uses a trained RandomForestClassifier to predict restaurant
inspection grades (A, B, C, P, Z) based on various features.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
import os
from streamlit_folium import st_folium
from utils import (
    load_model,
    prepare_input_data,
    predict_grade,
    get_feature_importances,
    validate_inputs,
    get_feature_names,
    geocode_address
)

# Page configuration
st.set_page_config(
    page_title="Restaurant Inspection Grade Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .grade-box-A {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background-color: #28a745;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .grade-box-B {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background-color: #ffc107;
        color: #000;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .grade-box-C {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background-color: #ff9800;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .grade-box-P {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background-color: #dc3545;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .grade-box-Z {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background-color: #dc3545;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model():
    """
    Load and cache the model to avoid reloading on every interaction.
    """
    try:
        model = load_model()
        return model, None
    except Exception as e:
        return None, str(e)


def get_borough_options():
    """Get list of NYC boroughs."""
    return [
        "Manhattan",
        "Brooklyn",
        "Queens",
        "Bronx",
        "Staten Island"
    ]


def get_cuisine_options():
    """Get list of common cuisine types."""
    return [
        "American",
        "Chinese",
        "Italian",
        "Mexican",
        "Japanese",
        "Thai",
        "Indian",
        "French",
        "Mediterranean",
        "Latin (Cuban, Dominican, Puerto Rican, South & Central American)",
        "Caribbean",
        "Asian",
        "Seafood",
        "Pizza",
        "Bakery",
        "Café/Coffee/Tea",
        "Fast Food",
        "Steakhouse",
        "Vegetarian",
        "Other"
    ]


def get_inspection_type_options():
    """Get list of inspection types."""
    return [
        "Initial Inspection",
        "Re-inspection",
        "Pre-permit (Operational) / Initial Inspection",
        "Cycle Inspection / Initial Inspection",
        "Cycle Inspection / Re-inspection",
        "Pre-permit (Operational) / Re-inspection",
        "Administrative Miscellaneous / Re-inspection",
        "Compliance Inspection",
        "Smoking Violation"
    ]


def display_grade_with_color(grade):
    """
    Display the predicted grade with colored box.
    
    Args:
        grade: Predicted grade (A, B, C, P, or Z)
    """
    grade_class_map = {
        'A': 'grade-box-A',
        'B': 'grade-box-B',
        'C': 'grade-box-C',
        'P': 'grade-box-P',
        'Z': 'grade-box-Z'
    }
    
    box_class = grade_class_map.get(grade, 'grade-box-A')
    
    st.markdown(
        f'<div class="{box_class}">{grade}</div>',
        unsafe_allow_html=True
    )


def create_map_with_marker(lat: float, lon: float, grade: str, address: str, api_key: str = None):
    """
    Create a Folium map with a marker showing the restaurant location and predicted grade.
    
    Args:
        lat: Latitude
        lon: Longitude
        grade: Predicted grade
        address: Restaurant address
        api_key: Google Maps API key (for optional Google Maps tiles)
        
    Returns:
        Folium map object
    """
    # Grade colors for markers
    grade_colors = {
        'A': 'green',
        'B': 'yellow',
        'C': 'orange',
        'P': 'red',
        'Z': 'red'
    }
    
    color = grade_colors.get(grade, 'blue')
    
    # Create map centered on the location
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add marker with popup
    popup_text = f"""
    <div style="text-align: center;">
        <h3 style="margin: 5px 0;">Restaurant Location</h3>
        <p style="margin: 5px 0;"><strong>Address:</strong><br>{address}</p>
        <p style="margin: 5px 0;"><strong>Predicted Grade:</strong></p>
        <h2 style="margin: 5px 0; color: {color};">{grade}</h2>
    </div>
    """
    
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=f"Predicted Grade: {grade}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)
    
    return m


def main():
    """Main application function."""
    
    # Title
    st.markdown('<h1 class="main-header">🍽️ Restaurant Inspection Grade Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, error = load_cached_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Please ensure the model file ('baseline_model.pkl' or 'model.pkl') exists in the current directory or 'app' folder.")
        return
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📋 Input Parameters")
        st.markdown("---")
        
        # Categorical inputs
        borough = st.selectbox(
            "Borough *",
            options=get_borough_options(),
            help="Select the borough where the restaurant is located"
        )
        
        cuisine_description = st.selectbox(
            "Cuisine Type *",
            options=get_cuisine_options(),
            help="Select the type of cuisine served"
        )
        
        inspection_type = st.selectbox(
            "Inspection Type *",
            options=get_inspection_type_options(),
            help="Select the type of inspection"
        )
        
        st.markdown("---")
        
        # Numerical inputs
        st.subheader("Numerical Features")
        
        nfh_indexscore = st.number_input(
            "NFH Index Score *",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1,
            help="Neighborhood Food Hazard Index Score"
        )
        
        median_income = st.number_input(
            "Median Income ($) *",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.0f",
            help="Median income of the area"
        )
        
        poverty_rate = st.number_input(
            "Poverty Rate (%) *",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=0.1,
            help="Poverty rate percentage"
        )
        
        st.markdown("---")
        
        # Google Maps section
        st.subheader("🗺️ Location (Optional)")
        restaurant_address = st.text_input(
            "Restaurant Address",
            placeholder="e.g., 123 Main St, New York, NY 10001",
            help="Enter the restaurant address to display on map"
        )
        
        # Google Maps API key input
        google_maps_api_key = st.text_input(
            "Google Maps API Key (Optional)",
            type="password",
            help="Enter your Google Maps API key to enable geocoding and maps. You can also set GOOGLE_MAPS_API_KEY environment variable."
        )
        
        # Check if API key is set via environment variable
        if not google_maps_api_key:
            google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
        
        st.markdown("---")
        st.markdown("*Required fields")
        
        # Predict button
        predict_button = st.button("🔮 Predict Grade", type="primary")
    
    # Main content area
    if predict_button:
        # Validate inputs
        is_valid, error_msg = validate_inputs(
            borough, cuisine_description, inspection_type,
            nfh_indexscore, median_income, poverty_rate
        )
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        # Prepare input data
        try:
            input_data = prepare_input_data(
                borough, cuisine_description, inspection_type,
                nfh_indexscore, median_income, poverty_rate
            )
            
            # Make prediction
            with st.spinner("🔮 Predicting grade..."):
                prediction, probabilities = predict_grade(model, input_data)
            
            # Display prediction
            st.markdown("---")
            st.markdown("## 📊 Prediction Result")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Predicted Grade")
                display_grade_with_color(prediction)
            
            with col2:
                st.markdown("### Prediction Confidence")
                
                if probabilities:
                    # Create bar chart for probabilities
                    prob_df = pd.DataFrame({
                        'Grade': list(probabilities.keys()),
                        'Probability': list(probabilities.values())
                    })
                    
                    # Sort by probability
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Create color mapping
                    grade_colors_map = {
                        'A': '#28a745',
                        'B': '#ffc107',
                        'C': '#ff9800',
                        'P': '#dc3545',
                        'Z': '#6c757d'
                    }
                    colors = [grade_colors_map.get(grade, '#000000') for grade in prob_df['Grade']]
                    
                    fig = px.bar(
                        prob_df,
                        x='Grade',
                        y='Probability',
                        labels={'Probability': 'Probability (%)'},
                        color='Grade',
                        color_discrete_map=grade_colors_map,
                        text='Probability'
                    )
                    fig.update_traces(
                        texttemplate='%{text:.1%}',
                        textposition='outside',
                        hovertemplate='Grade: %{x}<br>Probability: %{y:.1%}<extra></extra>'
                    )
                    fig.update_layout(
                        yaxis=dict(tickformat='.0%'),
                        showlegend=False,
                        height=400,
                        title="Probability Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display probabilities as text
                    st.markdown("**Detailed Probabilities:**")
                    for grade, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- **Grade {grade}**: {prob:.2%}")
            
            # Feature importances
            st.markdown("---")
            st.markdown("## 🔍 Feature Importance")
            st.markdown("Understanding which features most influence the prediction:")
            
            try:
                importances = get_feature_importances(model)
                
                if importances:
                    # Create feature importance chart
                    importance_df = pd.DataFrame({
                        'Feature': list(importances.keys()),
                        'Importance': list(importances.values())
                    })
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        labels={'Importance': 'Importance Score'},
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        title="Feature Importance Scores",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display top features
                    st.markdown("**Top Contributing Features:**")
                    for i, (feature, importance) in enumerate(list(importances.items())[:3], 1):
                        st.write(f"{i}. **{feature.replace('_', ' ').title()}**: {importance:.4f}")
                else:
                    st.info("Feature importances are not available for this model.")
            except Exception as e:
                st.warning(f"Could not retrieve feature importances: {str(e)}")
            
            # Google Maps display
            if restaurant_address and google_maps_api_key:
                st.markdown("---")
                st.markdown("## 🗺️ Restaurant Location")
                
                with st.spinner("📍 Geocoding address and loading map..."):
                    lat, lon, formatted_address = geocode_address(restaurant_address, google_maps_api_key)
                    
                    if lat and lon:
                        # Create and display map
                        map_obj = create_map_with_marker(
                            lat, lon, prediction, 
                            formatted_address or restaurant_address,
                            google_maps_api_key
                        )
                        
                        # Display the map
                        st_folium(map_obj, width=700, height=500)
                        
                        st.success(f"✅ Location found: {formatted_address or restaurant_address}")
                    else:
                        st.warning("⚠️ Could not geocode the address. Please check the address and API key.")
            elif restaurant_address and not google_maps_api_key:
                st.info("💡 To view the restaurant location on a map, please provide a Google Maps API key.")
            
            # Input summary
            st.markdown("---")
            with st.expander("📝 View Input Summary"):
                input_summary = {
                    'Borough': borough,
                    'Cuisine Type': cuisine_description,
                    'Inspection Type': inspection_type,
                    'NFH Index Score': f"{nfh_indexscore:.1f}",
                    'Median Income': f"${median_income:,.0f}",
                    'Poverty Rate': f"{poverty_rate:.1f}%"
                }
                summary_df = pd.DataFrame(list(input_summary.items()), columns=['Feature', 'Value'])
                st.table(summary_df)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("💡 Please check your inputs and try again.")
    
    else:
        # Display instructions when no prediction has been made
        st.info("""
        👆 **Instructions:**
        1. Fill in all the input parameters in the sidebar
        2. Click the **"Predict Grade"** button to get the predicted inspection grade
        3. View the prediction result, confidence scores, and feature importances
        
        **Grade Meanings:**
        - **A**: Excellent (Score: 0-13) - Green
        - **B**: Good (Score: 14-27) - Yellow
        - **C**: Acceptable (Score: 28+) - Orange
        - **P**: Pending (Issued when restaurant is re-inspected) - Red
        - **Z**: Grade Pending (Issued when restaurant requires re-inspection) - Red
        
        **Optional Features:**
        - Enter a restaurant address and Google Maps API key to view the location on an interactive map
        - The map will display a marker with the predicted grade
        """)


if __name__ == "__main__":
    main()

