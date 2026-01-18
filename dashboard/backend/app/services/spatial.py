import os
import json
import pandas as pd
import h3
from app.core.config import settings

class SpatialService:
    def __init__(self):
        self.h3_data = None
        self.last_query_result = None
        self.available_layers = ["NEIGHBORHOODS", "WALKABILITY"]
        
    def load_data(self):
        """Loads H3 data into memory for RAG context."""
        if os.path.exists(settings.H3_DATA_PATH):
            print(f"Loading H3 Data from: {settings.H3_DATA_PATH}")
            self.h3_data = pd.read_parquet(settings.H3_DATA_PATH)
            if 'h3_index' in self.h3_data.columns:
                self.h3_data.set_index('h3_index', inplace=True)
            print(f"H3 Data loaded: {len(self.h3_data)} locations.")
        else:
            print(f"Warning: H3 Data file not found at {settings.H3_DATA_PATH}")

    def get_location_context(self, lat: float, lon: float) -> str:
        """Retrieves context description for a given lat/lon using H3 index."""
        if self.h3_data is None or lat is None or lon is None:
            return ""
        
        try:
            h_idx = h3.latlng_to_cell(lat, lon, 9)
            if h_idx in self.h3_data.index:
                row = self.h3_data.loc[h_idx]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row.get('text_description', "")
            else:
                return ""
        except Exception as e:
            print(f"Error looking up location context: {e}")
            return ""

    def get_neighborhoods_geojson(self):
        """Reads and returns the Neighborhood GeoJSON data."""
        if not os.path.exists(settings.NEIGHBORHOODS_PATH):
             raise FileNotFoundError("Neighborhood data not found")
        
        with open(settings.NEIGHBORHOODS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_h3_records(self):
        """Reads H3 parquet and returns records for API."""
        if not os.path.exists(settings.H3_DATA_PATH):
             raise FileNotFoundError("H3 data not found")
        
        df = pd.read_parquet(settings.H3_DATA_PATH)
        return df.to_dict(orient='records')

    def set_query_result(self, geojson_data):
        self.last_query_result = geojson_data
        
    def get_query_result(self):
        return self.last_query_result

spatial_service = SpatialService()
