# This Flask application provides a unified weather endpoint by simulating 
# the processing and provision of NASA-derived weather data, tailored for 
# the complex React WeatherDashboard.

import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import random

# --- 1. CONFIGURATION ---

app = Flask(__name__)
# CRITICAL: Enable CORS for cross-origin access
CORS(app) 

# Replace this with your actual NASA API Key when going live with a real endpoint
NASA_API_KEY = "DEMO_KEY" 

# --- 2. DATA MOCK (Simulating NASA API Processing) ---

def fetch_nasa_data_and_process(lat, lon, date_str):
    """
    Simulates fetching and processing weather data based on location and date.
    In a real app, this is where you would call an external NASA API 
    (e.g., satellite imagery processing, planetary data) using the lat/lon/date.
    
    The output structure is precisely designed to feed the React component's 
    internal calculation logic.
    """
    try:
        query_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except:
        # Fallback to today's date if date format is bad
        query_date = datetime.now().date()
        date_str = query_date.isoformat()

    # --- Simulated Daily Raw Metrics (Max/Min Temp, Rain, Wind) ---
    # These values will vary slightly based on the latitude to simulate a real service
    
    # Simple temperature simulation: higher latitude -> generally colder.
    base_temp_max = 30 - abs(float(lat) / 90) * 20 + random.uniform(-5, 5)
    base_temp_min = 15 - abs(float(lat) / 90) * 10 + random.uniform(-3, 3)
    
    # Simple precipitation simulation: 
    base_rain = 5.0 + (random.randint(0, 10) / 10) * abs(float(lon) / 180) * 5
    
    # Simple wind speed simulation:
    base_wind = 20.0 + random.uniform(0, 15)
    
    daily_data = {
        "max_temp": round(base_temp_max, 1),
        "min_temp": round(base_temp_min, 1),
        "rain": round(base_rain, 1),
        "wind": round(base_wind, 1)
    }
    
    # --- Simulated 7-Day Forecast ---
    forecast = []
    for i in range(7):
        ds = (query_date + timedelta(days=i)).isoformat()
        
        # Apply simple daily variation to the base metrics
        maxT = base_temp_max + random.uniform(-3, 3)
        minT = base_temp_min + random.uniform(-2, 2)
        rainP = max(0, base_rain + random.uniform(-3, 3))
        windS = max(5, base_wind + random.uniform(-5, 5))
        
        # Condition logic to match the frontend's simple text expectations
        condition = 'Clear'
        if maxT > 30:
            condition = 'Hot'
        elif minT < 5:
            condition = 'Cold'
        elif rainP > 5:
            condition = 'Rainy'

        forecast.append({
            "date": ds,
            "maxTemp": round(maxT, 1),
            "minTemp": round(minT, 1),
            "precipitation": round(rainP, 1),
            "windSpeed": round(windS, 1),
            "condition": condition
        })
        
    return {
        "city": f"Weather at Lat {float(lat):.2f}, Lon {float(lon):.2f}",
        "daily_data": daily_data,
        "forecast": forecast
    }

# --- 3. FLASK ENDPOINT (The 'R' in CRUD) ---

@app.route('/api/weather', methods=['GET'])
def get_nasa_weather():
    """
    Endpoint for the React frontend to fetch processed weather data.
    Requires 'lat', 'lon', and 'date' query parameters.
    """
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    date_str = request.args.get('date')
    
    if not lat or not lon or not date_str:
        return jsonify({
            "error": "Missing parameters.",
            "details": "Requires lat, lon, and date query parameters."
        }), 400
    
    try:
        # Fetch and process the data based on parameters
        processed_data = fetch_nasa_data_and_process(lat, lon, date_str)
        
        # Return the structured data as JSON
        return jsonify(processed_data)

    except Exception as e:
        print(f"An unexpected error occurred in backend processing: {e}")
        return jsonify({
            "error": "Internal server error during data processing.",
            "details": str(e)
        }), 500

# --- 4. RUNNER ---

if __name__ == '__main__':
    # Run with: python nasa_weather_app.py
    print("Starting Flask server on http://127.0.0.1:5000...")
    app.run(debug=True, port=5000)
