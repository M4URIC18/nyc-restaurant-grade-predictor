"""
🍽 NYC Restaurant Grade Predictor

Streamlit web app for predicting restaurant inspection grades (A, B, C, P, Z)
using a trained RandomForestClassifier model.

Run this app with: streamlit run app/main.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os
from utils import (
    load_model,
    prepare_input_data,
    predict_grade,
    get_feature_importances,
    validate_inputs,
    geocode_address,
    reverse_geocode,
    normalize_borough_name
)

# Page configuration
st.set_page_config(
    page_title="NYC Restaurant Grade Predictor",
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
    .dashboard-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    .grade-card-A {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 16px rgba(40, 167, 69, 0.3);
        border: 3px solid #1e7e34;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .grade-card-A::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    }
    .grade-card-B {
        background: linear-gradient(135deg, #ffc107 0%, #ffeb3b 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 16px rgba(255, 193, 7, 0.3);
        border: 3px solid #f57c00;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .grade-card-B::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
    }
    .grade-card-C {
        background: linear-gradient(135deg, #ff9800 0%, #ff6f00 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 16px rgba(255, 152, 0, 0.3);
        border: 3px solid #e65100;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .grade-card-C::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    }
    .grade-card-P {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 16px rgba(108, 117, 125, 0.3);
        border: 3px solid #343a40;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .grade-card-P::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    }
    .grade-card-Z {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 16px rgba(220, 53, 69, 0.3);
        border: 3px solid #bd2130;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .grade-card-Z::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    }
    .grade-display {
        font-size: 5rem;
        font-weight: 900;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .grade-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.95);
        margin-top: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    .grade-label-B {
        color: #000;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    }
    .prediction-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1f77b4 0%, #155a8a 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(31, 119, 180, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 1.5rem 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #e9ecef;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        border-color: #1f77b4;
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


def display_grade_card(grade):
    """
    Display the predicted grade as a dashboard card with emoji and color coding.
    Color coding: green for A, yellow for B, orange for C, red for Z, gray for P.
    
    Args:
        grade: Predicted grade (A, B, C, P, or Z)
    """
    grade_config = {
        'A': {
            'class': 'grade-card-A',
            'emoji': '✅',
            'label': 'Excellent',
            'label_class': ''
        },
        'B': {
            'class': 'grade-card-B',
            'emoji': '👍',
            'label': 'Good',
            'label_class': 'grade-label-B'
        },
        'C': {
            'class': 'grade-card-C',
            'emoji': '⚠️',
            'label': 'Acceptable',
            'label_class': ''
        },
        'P': {
            'class': 'grade-card-P',
            'emoji': '⏳',
            'label': 'Pending',
            'label_class': ''
        },
        'Z': {
            'class': 'grade-card-Z',
            'emoji': '❌',
            'label': 'Grade Pending',
            'label_class': ''
        }
    }
    
    config = grade_config.get(grade, grade_config['A'])
    label_class = f" {config['label_class']}" if config['label_class'] else ""
    
    card_html = f"""
    <div class="{config['class']}">
        <div class="grade-display">{config['emoji']} {grade}</div>
        <div class="grade-label{label_class}">{config['label']}</div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Title
    st.markdown('<h1 class="main-header">🍽 NYC Restaurant Grade Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, error = load_cached_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Please ensure the model file ('baseline_model.pkl' or 'model.pkl') exists in the current directory, parent directory, or 'app' folder.")
        return
    
    # Initialize session state for map interactions
    if 'map_clicked_lat' not in st.session_state:
        st.session_state.map_clicked_lat = None
    if 'map_clicked_lng' not in st.session_state:
        st.session_state.map_clicked_lng = None
    if 'selected_address' not in st.session_state:
        st.session_state.selected_address = None
    if 'selected_borough' not in st.session_state:
        st.session_state.selected_borough = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'predicted_grade' not in st.session_state:
        st.session_state.predicted_grade = None
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🗺️ Google Maps Integration")
        st.markdown("---")
        
        # Google Maps API key input
        google_maps_api_key = st.text_input(
            "Google Maps API Key",
            type="password",
            help="Enter your Google Maps API key to enable map features. You can also set GOOGLE_MAPS_API_KEY environment variable.",
            value=os.getenv('GOOGLE_MAPS_API_KEY', '')
        )
        
        if not google_maps_api_key:
            st.warning("⚠️ Google Maps API key required for map features")
        
        st.markdown("---")
        
        # Address search
        if google_maps_api_key:
            st.subheader("📍 Search Restaurant Location")
            restaurant_address = st.text_input(
                "Restaurant Address",
                placeholder="e.g., 123 Main St, New York, NY 10001",
                help="Enter restaurant address and click 'Search on Map'",
                key="address_input"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                search_button = st.button("🔍 Search on Map", use_container_width=True)
            with col2:
                clear_button = st.button("🗑️ Clear", use_container_width=True)
            
            if clear_button:
                st.session_state.map_clicked_lat = None
                st.session_state.map_clicked_lng = None
                st.session_state.selected_address = None
                st.session_state.selected_borough = None
                st.session_state.prediction_made = False
                st.rerun()
            
            if search_button and restaurant_address:
                with st.spinner("📍 Searching location..."):
                    lat, lng, formatted_address, address_components = geocode_address(
                        restaurant_address, google_maps_api_key
                    )
                    if lat and lng:
                        st.session_state.map_clicked_lat = lat
                        st.session_state.map_clicked_lng = lng
                        st.session_state.selected_address = formatted_address
                        # Try to extract borough
                        if address_components and 'borough' in address_components:
                            borough_from_geo = normalize_borough_name(address_components['borough'])
                            if borough_from_geo in get_borough_options():
                                st.session_state.selected_borough = borough_from_geo
                        st.success(f"✅ Location found: {formatted_address}")
                        st.rerun()
                    else:
                        st.error("❌ Could not find location. Please check the address.")
        
        st.markdown("---")
        st.header("📋 Input Parameters")
        st.markdown("---")
        
        # Borough dropdown (auto-filled from map if available)
        borough_options = get_borough_options()
        default_borough_index = 0
        if st.session_state.selected_borough and st.session_state.selected_borough in borough_options:
            default_borough_index = borough_options.index(st.session_state.selected_borough)
        
        borough = st.selectbox(
            "Borough *",
            options=borough_options,
            index=default_borough_index,
            help="Select the borough where the restaurant is located (auto-filled from map if available)"
        )
        
        # Inspection type dropdown
        inspection_type = st.selectbox(
            "Inspection Type *",
            options=get_inspection_type_options(),
            help="Select the type of inspection"
        )
        
        # Cuisine description text input
        cuisine_description = st.text_input(
            "Cuisine Description *",
            placeholder="e.g., American, Italian, Chinese",
            help="Enter the type of cuisine served at the restaurant"
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
        st.markdown("*Required fields")
        
        # Predict button
        predict_button = st.button("🔮 Predict Grade", type="primary")
    
    # Helper function to create and display map
    def create_map_with_marker():
        """Create and display the interactive map with markers."""
        if not google_maps_api_key:
            return None
        
        # Create map
        if st.session_state.map_clicked_lat and st.session_state.map_clicked_lng:
            # Center map on selected location
            map_center_lat = st.session_state.map_clicked_lat
            map_center_lng = st.session_state.map_clicked_lng
            zoom_level = 15
        else:
            # Default to NYC center
            map_center_lat = 40.7128
            map_center_lng = -74.0060
            zoom_level = 11
        
        # Create Folium map
        m = folium.Map(
            location=[map_center_lat, map_center_lng],
            zoom_start=zoom_level,
            tiles='OpenStreetMap'
        )
        
        # If location is selected, add marker
        if st.session_state.map_clicked_lat and st.session_state.map_clicked_lng:
            # Determine marker color based on predicted grade
            grade_colors = {
                'A': 'green',
                'B': 'yellow',
                'C': 'orange',
                'P': 'gray',
                'Z': 'red'
            }
            
            marker_color = 'blue'
            if st.session_state.prediction_made and st.session_state.predicted_grade:
                marker_color = grade_colors.get(st.session_state.predicted_grade, 'blue')
            
            # Create popup content
            popup_content = f"""
            <div style="text-align: center; min-width: 200px;">
                <h4 style="margin: 5px 0;">🍽️ Restaurant Location</h4>
                <p style="margin: 5px 0; font-size: 12px;"><strong>Address:</strong><br>{st.session_state.selected_address or 'Selected Location'}</p>
            """
            
            if st.session_state.prediction_made and st.session_state.predicted_grade:
                popup_content += f"""
                <hr style="margin: 10px 0;">
                <p style="margin: 5px 0;"><strong>Predicted Grade:</strong></p>
                <h2 style="margin: 5px 0; color: {marker_color};">{st.session_state.predicted_grade}</h2>
                """
            
            popup_content += "</div>"
            
            # Create custom icon based on grade
            icon_type = 'info-sign'
            if st.session_state.prediction_made and st.session_state.predicted_grade:
                # Use different icons for different grades
                icon_map = {
                    'A': 'ok-sign',
                    'B': 'info-sign',
                    'C': 'warning-sign',
                    'P': 'question-sign',
                    'Z': 'remove-sign'
                }
                icon_type = icon_map.get(st.session_state.predicted_grade, 'info-sign')
            
            folium.Marker(
                [st.session_state.map_clicked_lat, st.session_state.map_clicked_lng],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Restaurant Location{' - Grade: ' + st.session_state.predicted_grade if st.session_state.prediction_made else ''}",
                icon=folium.Icon(color=marker_color, icon=icon_type)
            ).add_to(m)
        
        return m
    
    # Main content area - Map section (if API key is available) - Show before prediction
    if google_maps_api_key:
        st.markdown("---")
        st.markdown("## 🗺️ Interactive Map")
        st.markdown("**Click on the map to select a restaurant location, or search for an address above.**")
        
        # Create and display map
        m = create_map_with_marker()
        if m:
            # Display map and get click events
            map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])
            
            # Handle map clicks
            if map_data["last_clicked"] is not None:
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lng = map_data["last_clicked"]["lng"]
                
                # Only update if it's a new click (not the same as current)
                if (st.session_state.map_clicked_lat != clicked_lat or 
                    st.session_state.map_clicked_lng != clicked_lng):
                    with st.spinner("📍 Getting location details..."):
                        formatted_address, address_components = reverse_geocode(
                            clicked_lat, clicked_lng, google_maps_api_key
                        )
                        
                        if formatted_address:
                            st.session_state.map_clicked_lat = clicked_lat
                            st.session_state.map_clicked_lng = clicked_lng
                            st.session_state.selected_address = formatted_address
                            
                            # Try to extract borough
                            if address_components and 'borough' in address_components:
                                borough_from_geo = normalize_borough_name(address_components['borough'])
                                if borough_from_geo in get_borough_options():
                                    st.session_state.selected_borough = borough_from_geo
                            
                            st.session_state.prediction_made = False
                            st.success(f"✅ Location selected: {formatted_address}")
                            st.rerun()
        
        st.markdown("---")
    
    # Main content area - Prediction section
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
            
            # Store prediction in session state for map display
            st.session_state.prediction_made = True
            st.session_state.predicted_grade = prediction
            
            # Display prediction in dashboard card style
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Prediction Result")
            
            # Show map update message if location is selected
            if google_maps_api_key and st.session_state.map_clicked_lat and st.session_state.map_clicked_lng:
                st.info("📍 **Map Updated:** Scroll up to see the restaurant marker on the map with the predicted grade!")
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🎯 Predicted Grade")
                display_grade_card(prediction)
                
                # Grade explanation in info card style
                grade_meanings = {
                    'A': 'Excellent (Score: 0-13)',
                    'B': 'Good (Score: 14-27)',
                    'C': 'Acceptable (Score: 28+)',
                    'P': 'Pending (Re-inspection required)',
                    'Z': 'Grade Pending (Re-inspection required)'
                }
                st.markdown(f"""
                <div class="info-card">
                    <strong>ℹ️ Grade {prediction}:</strong><br>
                    {grade_meanings.get(prediction, 'Unknown')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📈 Prediction Confidence")
                
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
                        'A': '#28a745',  # Green
                        'B': '#ffc107',  # Yellow
                        'C': '#ff9800',  # Orange
                        'P': '#6c757d',  # Gray
                        'Z': '#dc3545'   # Red
                    }
                    
                    fig = px.bar(
                        prob_df,
                        x='Grade',
                        y='Probability',
                        labels={'Probability': 'Probability'},
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
                else:
                    st.info("ℹ️ Probability scores are not available for this model.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature importances in dashboard card
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("## 🔍 Feature Importance Analysis")
            st.markdown("**Discover which features most influence the prediction:**")
            st.markdown("---")
            
            try:
                importances = get_feature_importances(model)
                
                if importances:
                    # Create DataFrame with feature importances
                    importance_df = pd.DataFrame({
                        'Feature': list(importances.keys()),
                        'Importance': list(importances.values())
                    })
                    
                    # Get top 5 features
                    top_5_df = importance_df.head(5).copy()
                    
                    # Calculate percentage based on ALL features (not just top 5)
                    total_importance_all = importance_df['Importance'].sum()
                    top_5_df['Percentage'] = (top_5_df['Importance'] / total_importance_all * 100).round(2)
                    
                    # Clean up feature names for display
                    top_5_df['Feature_Clean'] = top_5_df['Feature'].apply(
                        lambda x: str(x).replace('_', ' ').title()
                    )
                    
                    # Create color gradient based on importance
                    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
                    top_5_df['Color'] = colors[:len(top_5_df)]
                    
                    # Create horizontal bar chart for top 5 features
                    fig = px.bar(
                        top_5_df,
                        x='Importance',
                        y='Feature_Clean',
                        orientation='h',
                        labels={
                            'Importance': 'Importance Score',
                            'Feature_Clean': 'Feature'
                        },
                        color='Feature_Clean',
                        color_discrete_sequence=colors[:len(top_5_df)],
                        text='Importance'
                    )
                    
                    # Update traces for better styling
                    fig.update_traces(
                        texttemplate='%{x:.4f}',
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>' +
                                    'Importance Score: %{x:.4f}<br>' +
                                    'Percentage: %{customdata:.2f}%<br>' +
                                    '<extra></extra>',
                        customdata=top_5_df['Percentage'],
                        marker=dict(
                            line=dict(width=2, color='white'),
                            opacity=0.9
                        ),
                        textfont=dict(size=11, color='#2c3e50')
                    )
                    
                    # Update layout for better appearance
                    fig.update_layout(
                        yaxis={
                            'categoryorder': 'total ascending',
                            'title': '',
                            'tickfont': {'size': 12, 'color': '#2c3e50'}
                        },
                        xaxis={
                            'title': 'Importance Score',
                            'titlefont': {'size': 14, 'color': '#2c3e50'},
                            'tickfont': {'size': 11}
                        },
                        height=350,
                        title={
                            'text': '📊 Top 5 Most Important Features',
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial'}
                        },
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=60, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display summary statistics
                    top_5_total_pct = top_5_df['Percentage'].sum()
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-top: 1rem; margin-bottom: 1.5rem; border-left: 4px solid #1f77b4;">
                        <strong>💡 Insight:</strong> These top 5 features account for 
                        <strong style="color: #1f77b4; font-size: 1.1em;">{top_5_total_pct:.1f}%</strong> of the total feature importance 
                        across all <strong>{len(importance_df)}</strong> features.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Optional: Display detailed breakdown in metric cards
                    st.markdown("#### 🏆 Detailed Feature Breakdown")
                    # Create columns for metric cards
                    feature_list = list(importances.items())[:6]
                    if len(feature_list) <= 3:
                        cols = st.columns(len(feature_list))
                    else:
                        cols = st.columns(3)
                    
                    for i, (feature, importance) in enumerate(feature_list):
                        # Clean up feature names
                        clean_feature = str(feature).replace('_', ' ').title()
                        col_idx = i % 3 if len(feature_list) > 3 else i
                        with cols[col_idx]:
                            # Use emoji for ranking
                            emoji_map = {1: '🥇', 2: '🥈', 3: '🥉'}
                            rank_emoji = emoji_map.get(i+1, f'#{i+1}')
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{rank_emoji}</div>
                                <div style="font-size: 0.9rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">{clean_feature}</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #1f77b4;">{importance:.4f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Feature importances are not available for this model.")
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve feature importances: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Input summary in dashboard card
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("## 📝 Input Summary")
            with st.expander("🔽 Expand to view all input parameters"):
                input_summary = {
                    'Borough': borough,
                    'Cuisine Description': cuisine_description,
                    'Inspection Type': inspection_type,
                    'NFH Index Score': f"{nfh_indexscore:.1f}",
                    'Median Income': f"${median_income:,.0f}",
                    'Poverty Rate': f"{poverty_rate:.1f}%"
                }
                if st.session_state.selected_address:
                    input_summary['Selected Address'] = st.session_state.selected_address
                summary_df = pd.DataFrame(list(input_summary.items()), columns=['Feature', 'Value'])
                st.table(summary_df)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show map update reminder if location was selected
            if google_maps_api_key and st.session_state.map_clicked_lat and st.session_state.map_clicked_lng:
                st.markdown("---")
                st.success("🗺️ **Map Updated!** The restaurant marker on the map above has been updated with the predicted grade. Scroll up to view it!")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("💡 Please check your inputs and try again.")
    
    else:
        # Display instructions when no prediction has been made
        instructions = """
        👆 **Instructions:**
        1. Fill in all the input parameters in the sidebar
        2. Click the **"Predict Grade"** button to get the predicted inspection grade
        3. View the prediction result, confidence scores, and feature importances
        
        **Grade Meanings:**
        - **A** (Green): Excellent (Score: 0-13)
        - **B** (Yellow): Good (Score: 14-27)
        - **C** (Orange): Acceptable (Score: 28+)
        - **P** (Gray): Pending (Issued when restaurant is re-inspected)
        - **Z** (Red): Grade Pending (Issued when restaurant requires re-inspection)
        """
        
        if google_maps_api_key:
            instructions += """
        
        **🗺️ Interactive Map Features:**
        - **Click on the map** to select a restaurant location
        - **Search for an address** in the sidebar to locate a restaurant
        - The **borough will be auto-filled** from the selected location
        - After prediction, the **map marker will show the predicted grade** with color coding
        - Click the marker to see the address and predicted grade in a popup
        """
        
        st.info(instructions)


if __name__ == "__main__":
    main()


"""
README - How to Run This App
=============================

This Streamlit app predicts NYC restaurant inspection grades using a trained
RandomForestClassifier model.

Prerequisites:
--------------
1. Python 3.8 or higher
2. Install required packages: pip install -r requirements.txt
3. Trained model file (baseline_model.pkl or model.pkl) in the project directory

Running the App:
----------------
To run this app, use the following command from the project root directory:

    streamlit run app/main.py

The app will open in your default web browser at http://localhost:8501

Usage:
------
1. Fill in the input parameters in the sidebar:
   - Select borough from dropdown
   - Select inspection type from dropdown
   - Enter cuisine description as text
   - Enter numerical values for NFH index score, median income, and poverty rate

2. Click "Predict Grade" to get the prediction

3. View the results:
   - Predicted grade with color-coded display
   - Probability distribution chart
   - Feature importance chart showing which features most influence the prediction

Model Input Features:
---------------------
- borough: NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
- cuisine_description: Type of cuisine (text input)
- inspection_type: Type of inspection
- nfh_indexscore: Neighborhood Food Hazard Index Score (0-100)
- median_income: Median income of the area
- poverty_rate: Poverty rate percentage (0-100)

File Structure:
---------------
Project Folder/
├── app/
│   ├── main.py          # Streamlit application
│   └── utils.py         # Utility functions
├── baseline_model.pkl   # Trained model file
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation

Notes:
------
- The model file can be located in the app/ directory, parent directory, or current directory
- The app handles both Pipeline models and standalone RandomForestClassifier models
- Feature importances are extracted from model.named_steps['randomforestclassifier'] if the model is a Pipeline
- All inputs are validated before making predictions
- The app handles missing or invalid inputs gracefully with error messages
"""

