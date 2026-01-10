
import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import pyogrio
from shapely.geometry import Point

# --- Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREETS_PATH = os.path.join(BASE_DIR, "..", "data", "streets.geojson")
GPKG_PATH = os.path.join(BASE_DIR, "..", "data", "sam3_results_50_overalping.gpkg")

st.set_page_config(layout="wide", page_title="Advanced Street Analytics", page_icon="🏙️")

@st.cache_data
def get_data_bounds():
    try:
        layers = pyogrio.list_layers(GPKG_PATH)
        target = layers[0][0]
        for l in layers:
            if "road" in l[0].lower():
                target = l[0]
                break
        info = pyogrio.read_info(GPKG_PATH, layer=target)
        return info['total_bounds'] 
    except:
        return None

def sanitize_gdf(gdf):
    if gdf is None or gdf.empty:
        return None
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
    gdf = gdf[~gdf.geometry.type.isin(['GeometryCollection'])]
    if gdf.empty:
        return None
    return gdf

@st.cache_data
def load_streets(filter_bounds=None):
    if not os.path.exists(STREETS_PATH):
        return None, None
    try:
        if filter_bounds:
            minx, miny, maxx, maxy = filter_bounds
            pad = 0.005 
            bbox = (minx-pad, miny-pad, maxx+pad, maxy+pad)
            streets = gpd.read_file(STREETS_PATH, bbox=bbox, engine="pyogrio")
        else:
            streets = gpd.read_file(STREETS_PATH, engine="pyogrio")
        
        if streets.empty:
            return None, None

        if 'id' in streets.columns:
            streets['id'] = streets['id'].astype(str)
        else:
            streets['id'] = streets.index.astype(str)
        
        # Ensure ID stripped
        streets['id'] = streets['id'].str.strip()
        
        streets = sanitize_gdf(streets)
        
        if streets is not None:
             if streets.crs and streets.crs.to_string() != "EPSG:4326":
                 streets_wgs84 = streets.to_crs("EPSG:4326")
             else:
                 streets_wgs84 = streets
                 if not streets.crs:
                     streets_wgs84 = streets_wgs84.set_crs("EPSG:4326")
             
             streets_proj = streets_wgs84.to_crs("EPSG:32618")
             return streets_wgs84, streets_proj
             
        return None, None
    except Exception as e:
        st.error(f"Error loading streets: {e}")
        return None, None

def load_segments_in_bbox(bbox, layer_names):
    segments = {}
    for layer in layer_names:
        try:
            pad = 0.0005 
            padded_bbox = (bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad)
            gdf = pyogrio.read_dataframe(GPKG_PATH, layer=layer, bbox=padded_bbox)
            if not gdf.empty:
                segments[layer] = gdf
        except:
            pass
    return segments

def get_layer_names():
    try:
        available = pyogrio.list_layers(GPKG_PATH)
        names = [l[0] for l in available if "raw_streets" not in l[0]]
        return names
    except:
        return ["road"]

def calculate_linear_coverage(street_geom, sidewalk_gdf):
    try:
        if sidewalk_gdf is None or sidewalk_gdf.empty:
            return 0.0
        sidewalk_inf = sidewalk_gdf.buffer(8) 
        covered_line = street_geom.intersection(sidewalk_inf.unary_union)
        if covered_line.is_empty: return 0.0
        return covered_line.length / street_geom.length
    except:
        return 0.0

def generate_reasoning(metrics):
    reasons = []
    consist = metrics.get('sidewalk_consistency', 0)
    if consist > 0.8:
        reasons.append("✅ **Excellent Continuity**: Sidewalks accompany >80% of the street.")
    elif consist > 0.5:
        reasons.append("⚠️ **Fragmented Sidewalks**: Only 50-80% of the street has sidewalks.")
    else:
        reasons.append("❌ **Disconnected**: Less than 50% of the street has pedestrian paths.")
    return reasons

# --- Main App ---
def main():
    st.title("🏙️ Intelligent Street Walkability Analytics")
    
    data_bounds = get_data_bounds()
    if not data_bounds: st.stop()
    
    with st.spinner("Loading Network..."):
        streets_wgs84, streets_proj = load_streets(filter_bounds=data_bounds)
        
    if streets_wgs84 is None: st.stop()
    
    valid_ids = [str(x) for x in streets_wgs84['id'].tolist()]
    
    if 'selected_street_id' not in st.session_state:
        st.session_state.selected_street_id = valid_ids[0]
        
    current_sel = str(st.session_state.selected_street_id)
    if current_sel not in valid_ids:
        st.session_state.selected_street_id = valid_ids[0]
        current_sel = valid_ids[0]

    st.markdown(f"### 📍 Inspecting Street ID: `{current_sel}`")

    col_sidebar, col_main = st.columns([1, 3])
    
    with col_sidebar:
        st.header("Controls")
        buffer_val = st.slider("Buffer (m)", 5, 100, 30)
        st.info("👆 Click any street on the map to inspect.")
        # Removed Dropdown list as specificed by user request

    with col_main:
        if 'map_center' not in st.session_state:
             minx, miny, maxx, maxy = data_bounds
             cx, cy = (minx+maxx)/2, (miny+maxy)/2
             st.session_state.map_center = [cy, cx]
        
        m = folium.Map(location=st.session_state.map_center, zoom_start=16, tiles="CartoDB positron")
        
        folium.Rectangle(
            bounds=[[data_bounds[1], data_bounds[0]], [data_bounds[3], data_bounds[2]]],
            color="green", fill=False, dash_array="5,5", weight=2
        ).add_to(m)
        
        def style_fn(feature):
            fid = str(feature['properties']['id']).strip()
            if fid == current_sel:
                return {'color': '#FF0000', 'weight': 6, 'opacity': 1}
            return {'color': '#3388ff', 'weight': 3, 'opacity': 0.6}
            
        def highlight_fn(feature):
            return {'color': '#00FF00', 'weight': 5, 'opacity': 0.9}

        folium.GeoJson(
            streets_wgs84,
            name="Streets",
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.GeoJsonTooltip(fields=['id']),
            zoom_on_click=False 
        ).add_to(m)
        
        out = st_folium(m, height=450, width="100%", returned_objects=["last_object_clicked"])
        
        # --- Click Logic ---
        if out and out.get("last_object_clicked"):
            clicked = out["last_object_clicked"]
            
            # 1. Properties Click (Ideal)
            if clicked and 'properties' in clicked:
                cid = str(clicked['properties'].get('id', '')).strip()
                if cid and cid in valid_ids and cid != current_sel:
                    st.session_state.selected_street_id = cid
                    st.rerun()
            
            # 2. Spatial Fallback (Lat/Lng)
            elif clicked and 'lat' in clicked and 'lng' in clicked:
                lat, lng = clicked['lat'], clicked['lng']
                
                # Create Point
                click_point = Point(lng, lat)
                click_gdf = gpd.GeoSeries([click_point], crs="EPSG:4326")
                click_proj = click_gdf.to_crs("EPSG:32618").iloc[0]
                
                # Check distances to all streets (in simple loop or vector)
                # Filter to streets roughly near by (bounding box) to speed up?
                # For small dataset, full calc is fine.
                
                # Calculate distance to all geometry
                # We need streets_proj indexed by ID
                distances = streets_proj.geometry.distance(click_proj)
                min_dist = distances.min()
                nearest_idx = distances.idxmin()
                
                if min_dist < 40: # 40 meter tolerance (generous click)
                    found_id = str(streets_proj.loc[nearest_idx, 'id']).strip()
                    if found_id != current_sel:
                        st.session_state.selected_street_id = found_id
                        st.rerun()

    # --- Analytics ---
    st.divider()
    sel_row = streets_proj[streets_proj['id'] == current_sel]
    
    if sel_row.empty:
        st.error(f"Error: Geometry for ID {current_sel} not found.")
        st.stop()
        
    geom_proj = sel_row.iloc[0].geometry
    if not geom_proj.is_valid: geom_proj = geom_proj.buffer(0)
    
    buffer_proj = geom_proj.buffer(buffer_val)
    buffer_wgs84 = gpd.GeoSeries([buffer_proj], crs=streets_proj.crs).to_crs("EPSG:4326")
    
    with st.spinner("Processing Data..."):
        layer_list = get_layer_names()
        segments_map = load_segments_in_bbox(buffer_wgs84.total_bounds, layer_list)
        
    analytics = {}
    intersected_vis = []
    sidewalk_union = None
    
    for name, gdf in segments_map.items():
        try:
            if gdf.crs != streets_proj.crs:
                 gdf_proj = gdf.to_crs(streets_proj.crs)
            else:
                 gdf_proj = gdf
            
            gdf_proj = sanitize_gdf(gdf_proj)
            if gdf_proj is None:
                analytics[name] = {"Count": 0, "Area": 0}
                continue
                
            if "sidewalk" in name.lower():
                sidewalk_union = pd.concat([sidewalk_union, gdf_proj]) if sidewalk_union is not None else gdf_proj

            matches = gdf_proj[gdf_proj.intersects(buffer_proj)]
            if not matches.empty:
                area = matches.intersection(buffer_proj).area.sum()
                analytics[name] = {"Count": len(matches), "Area": area}
                
                vis = matches.to_crs("EPSG:4326")
                vis = sanitize_gdf(vis)
                if vis is not None:
                     vis = vis[~vis.geometry.type.isin(['GeometryCollection'])]
                     if not vis.empty:
                         intersected_vis.append((name, vis))
            else:
                analytics[name] = {"Count": 0, "Area": 0}
        except:
            analytics[name] = {"Count": 0, "Area": 0}

    def sum_key(k_list):
        a, c = 0, 0
        for name, d in analytics.items():
            for k in k_list:
                if k in name.lower():
                    a += d['Area']
                    c += d['Count']
                    break
        return a, c
        
    road_a, _ = sum_key(['road'])
    side_a, _ = sum_key(['sidewalk'])
    car_a, _ = sum_key(['car'])
    _, tree_c = sum_key(['tree'])
    
    metrics = {
        'walk_score': (side_a/road_a*10) if road_a > 0 else 0,
        'car_density': car_a/road_a if road_a > 0 else 0,
        'tree_count': tree_c,
        'sidewalk_consistency': calculate_linear_coverage(geom_proj, sidewalk_union)
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Consistency", f"{metrics['sidewalk_consistency']:.1%}")
        c2.metric("Car Density", f"{metrics['car_density']:.0%}")
        c3.metric("Trees", tree_c)
        st.progress(metrics['sidewalk_consistency'], text="Sidewalk Continuity")
        
        rows = []
        buf_area = buffer_proj.area
        for k, v in analytics.items():
            rows.append({
                "Layer": k,
                "Count": v['Count'],
                "Area": f"{v['Area']:.1f}",
                "% Buffer": f"{(v['Area']/buf_area):.1%}"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
    with col2:
        st.subheader("🔍 Inspection")
        bounds = buffer_wgs84.total_bounds
        cy, cx = (bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2
        m2 = folium.Map(location=[cy, cx], zoom_start=19, tiles="CartoDB positron")
        
        if not buffer_wgs84.empty:
            folium.GeoJson(buffer_wgs84, style_function=lambda x: {'color':'blue','fillOpacity':0.05,'dashArray':'5,5'}).add_to(m2)
            
        colors = ["#E74C3C", "#8E44AD", "#3498DB", "#1ABC9C", "#F1C40F", "#34495E"]
        for i, (name, vis) in enumerate(intersected_vis):
            col = colors[abs(hash(name)) % len(colors)]
            folium.GeoJson(vis, name=name, style_function=lambda x, col=col: {'color':col, 'weight':1, 'fillOpacity':0.6}, tooltip=name).add_to(m2)
            
        folium.LayerControl().add_to(m2)
        st_folium(m2, height=500, width="100%", key="detail")

if __name__ == "__main__":
    main()
