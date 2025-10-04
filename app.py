import os
import json
from datetime import datetime, timedelta
import pathlib

# Import Flask and CORS, and the function to send files
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import Prophet dependencies
import pandas as pd
from prophet import Prophet
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
# Get API key from environment variables (CRITICAL for production)
NASA_API_KEY = os.environ.get('NASA_API_KEY', 'YOUR_NASA_API_KEY_HERE')
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# --- NEW: Geocoding API Configuration ---
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search" # Base URL

# --- FLASK APP INITIALIZATION FOR STATIC FILES ---
# Find the absolute path to the 'dist' folder for deployment reliability
# This assumes the 'dist' folder is a sibling of 'app.py' inside the 'backend' folder.
base_dir = pathlib.Path(__file__).parent.resolve()
dist_path = base_dir / 'dist'

app = Flask(
    __name__,
    static_folder=str(dist_path / 'assets'),  # For serving compiled JS/CSS/Images
    template_folder=str(dist_path)             # For serving index.html
)
# Enable CORS for the API endpoints only (Frontend and Backend are now on the same host)
CORS(app)

# --- SERVING REACT FRONTEND (STATIC ROUTES) ---

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """
    Catch-all route to serve the index.html file for any non-API path.
    This allows React Router to handle client-side routing.
    """
    if path != "" and (dist_path / path).exists():
        # If the request is for a specific file in dist/ (e.g., /favicon.ico), serve it.
        # Note: We configure 'static_folder' above for assets/
        return send_from_directory(dist_path, path)
    
    # Otherwise, serve the main entry point (index.html)
    return send_from_directory(dist_path, 'index.html')


# --- Helper Functions ---

def geocode_location(search_term):
    """
    Helper function to convert a location name (search_term) into latitude and longitude 
    using the Nominatim API.
    
    NOTE: This function is currently UNUSED by the primary /api/weather endpoint 
    as it expects lat/lon directly. This is for demonstration/future expansion.
    """
    params = {
        'q': search_term,
        'format': 'json',
        'limit': 1
    }
    
    try:
        response = requests.get(NOMINATIM_API_URL, params=params, headers={'User-Agent': 'WeatherApp/1.0'}, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            print(f"Geocoding failed for: {search_term}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Geocoding request failed: {e}")
        return None, None


def fetch_historical_data(lat, lon, start_date, end_date):
    """
    Fetches historical daily maximum temperature (T2M_MAX) data from the NASA POWER API.
    
    The data required for Prophet must be a Pandas DataFrame with columns 'ds' (datetime) 
    and 'y' (the value to predict).
    """
    print(f"Fetching NASA POWER data for {lat}, {lon} from {start_date} to {end_date}...")
    
    # Define the parameters for the API call
    params = {
        'request': 'execute',
        'parameters': 'T2M_MAX',  # Daily Maximum Air Temperature at 2 Meters
        'start_date': start_date,
        'end_date': end_date,
        'latitude': lat,
        'longitude': lon,
        'community': 'RE',        # Renewable Energy community data
        'timeframe': 'daily',
        'format': 'json',
        'user': NASA_API_KEY      # The API key acts as the 'user' parameter
    }

    # Set up session with retry logic for robust API calls
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"HEAD", "GET", "OPTIONS"}
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        # Use a custom header to comply with Nominatim's requirements (though we are calling NASA here)
        response = session.get(NASA_POWER_API_URL, params=params, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        # --- Process NASA JSON Response ---
        # Data is stored in 'properties' -> 'parameter' -> 'T2M_MAX'
        if not data.get('properties'):
            raise ValueError("API response is missing 'properties' key. Check parameters.")
        
        raw_data = data['properties']['parameter']['T2M_MAX']
        
        # Convert the dictionary of 'YYYYMMDD': value into a list of (date, value) tuples
        df_list = []
        for date_str, temp in raw_data.items():
            if temp is not None and temp != -999.0: # Filter out null/missing values
                df_list.append({
                    'ds': datetime.strptime(date_str, '%Y%m%d'),
                    'y': float(temp)
                })

        if not df_list:
             # If no real data is found, we should still return an empty DF or raise an error
             print("WARNING: No valid historical data returned from NASA. Prophet will fail.")
             return pd.DataFrame({'ds': [], 'y': []})

        df = pd.DataFrame(df_list)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        # In a real scenario, you might want to return mock data here as a fallback
        return pd.DataFrame({'ds': [], 'y': []})
    except Exception as e:
        print(f"Data processing failed: {e}")
        return pd.DataFrame({'ds': [], 'y': []})


def run_prophet_forecast(df_history, forecast_days=7):
    """
    Trains the Prophet model and generates a 7-day forecast.
    """
    if df_history.empty:
        # Fallback for when historical data fetching failed
        print("Prophet cannot run: Historical data is empty.")
        # Return the mock data that the React frontend is currently designed to use as a fallback
        # This prevents the forecast from crashing if the NASA API fails.
        mock_forecast = [
            {'day': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'), 
             'maxTemp': 22 + (i % 3), 'minTemp': 10 + (i % 2), 
             'condition': 'Fallback Data', 'windSpeed': 5, 'humidity': 65}
            for i in range(forecast_days)
        ]
        return mock_forecast

    # 1. Initialize the Prophet model
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    m.fit(df_history)
    
    # 2. Create future DataFrame and Predict
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    
    # 3. Extract and format the 7 forecasted days
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
    
    formatted_forecast = []
    for index, row in forecast_df.iterrows():
        # Temperature conversion: Prophet predicts Max Temp (yhat) in Celsius
        max_temp = int(row['yhat'])
        min_temp = max(0, max_temp - 10) # Simple approximation of Min Temp
        
        # Simple condition logic
        condition = 'Sunny'
        if max_temp < 15:
            condition = 'Partly Cloudy'
        elif max_temp < 10:
            condition = 'Rain'
            
        formatted_forecast.append({
            'day': row['ds'].strftime('%Y-%m-%d'),
            'maxTemp': max_temp,
            'minTemp': min_temp,
            # Placeholder for other weather variables (NASA API doesn't give them all easily)
            'condition': condition,
            'windSpeed': 5 + index % 10,
            'humidity': 60 + index * 2
        })
        
    return formatted_forecast


# --- API ENDPOINT (No change needed here) ---

@app.route('/api/weather', methods=['GET'])
def get_weather_forecast():
    """
    API Endpoint to provide current weather and 7-day Prophet forecast.
    """
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            return jsonify({"error": "Latitude (lat) and Longitude (lon) are required."}), 400

        # --- 1. Define Historical Period ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * 365) # Use 3 years of historical data
        
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')

        # --- 2. Fetch Historical Data ---
        historical_df = fetch_historical_data(lat, lon, start_date_str, end_date_str)

        # --- 3. Generate Forecast using Prophet ---
        seven_day_forecast = run_prophet_forecast(historical_df)

        # --- 4. Prepare Current Weather Data ---
        # Use the first day's forecast as the current prediction
        if seven_day_forecast:
            first_day = seven_day_forecast[0]
            current_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'temp': first_day['maxTemp'] - 3, 
                'maxTemp': first_day['maxTemp'],
                'minTemp': first_day['minTemp'],
                'sunrise': '06:30',
                'sunset': '18:00',
                'condition': first_day['condition'],
                'humidity': first_day['humidity'],
                'windSpeed': first_day['windSpeed']
            }
        else:
             # Use a basic mock if the whole process failed
             current_data = {
                'date': datetime.now().strftime('%Y-%m-%d'), 'temp': 20, 'maxTemp': 25, 
                'minTemp': 10, 'sunrise': '06:00', 'sunset': '18:30', 
                'condition': 'No Data', 'humidity': 50, 'windSpeed': 5
            }


        # --- 5. Return JSON Response ---
        return jsonify({
            "current": current_data,
            "forecast": seven_day_forecast
        })

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "Internal server error during forecasting.", "details": str(e)}), 500

if __name__ == '__main__':
    # When running locally, Flask uses the default host and port
    # Note: To test the static file serving locally, run: 
    # python backend/app.py
    app.run(debug=True)
