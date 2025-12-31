"""
Riyadh Urban Trends - Massive Data Collection Script
Collects monthly Sentinel-2 imagery (All Bands) from 2017 to Present.
Enforces strict pixel-to-pixel consistency for Deep Learning.
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import shapely.geometry as geom
from shapely.geometry import shape
from shapely.ops import unary_union
from pystac_client import Client
import planetary_computer as pc
from tqdm import tqdm
from datetime import datetime
import json
import traceback

# Fix PROJ path for Windows environments
if os.name == 'nt':
    # Try finding PROJ in commonly known locations/site-packages
    try:
        import rasterio
        os.environ['PROJ_LIB'] = os.path.join(os.path.dirname(rasterio.__file__), 'proj_data')
    except:
        pass

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
class Config:
    # ROI: 50km radius around Riyadh
    RIYADH_BBOX = geom.box(46.17, 24.26, 47.17, 25.16)
    
    # 2017 is roughly when S2 L2A starts becoming consistently available
    START_YEAR = 2017 
    END_YEAR = datetime.now().year
    
    # All Sentinel-2 Bands (Resampled to 10m where needed)
    BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    
    MAX_CLOUD = 10         # Relaxed slightly to find *some* data for every month
    MIN_COVERAGE = 98.0    # Strict geometric coverage
    
    # Output
    OUTPUT_DIR = "dataset_v2"
    RAW_DIR = os.path.join(OUTPUT_DIR, "monthly_tiffs")
    REPORT_FILE = os.path.join(OUTPUT_DIR, "collection_summary.txt")

    @staticmethod
    def setup():
        os.makedirs(Config.RAW_DIR, exist_ok=True)

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
class SummaryWriter:
    def __init__(self, path):
        self.path = path
        self.log = []
        
    def add(self, message):
        self.log.append(message)
        # Auto-save to avoid data loss on crash
        with open(self.path, "w") as f:
            f.write("\n".join(self.log))
            
    def print_and_add(self, message):
        print(message)
        self.add(message)

# ---------------------------------------------------------
# CONSISTENCY ENFORCER
# ---------------------------------------------------------
class ConsistencyEnforcer:
    """
    Ensures every single downloaded image matches the EXACT grid of the reference image.
    """
    def __init__(self):
        self.reference_profile = None
        self.reference_path = None
        
    def set_reference(self, profile, path):
        print(f"SETTING REFERENCE GRID from: {path}")
        self.reference_profile = profile
        self.reference_path = path
        
    def enforce(self, src_dataset, dest_path):
        """
        Reprojects/Resamples src_dataset to match self.reference_profile
        and saves to dest_path.
        """
        if self.reference_profile is None:
            # This IS the reference
            return True, "Is Reference"

        dst_profile = self.reference_profile.copy()
        dst_profile.update(driver='GTiff')
        
        with rasterio.open(dest_path, 'w', **dst_profile) as dst:
            for i in range(1, src_dataset.count + 1):
                reproject(
                    source=rasterio.band(src_dataset, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_dataset.transform,
                    src_crs=src_dataset.crs,
                    dst_transform=dst_profile['transform'],
                    dst_crs=dst_profile['crs'],
                    resampling=Resampling.nearest
                )
        return True, " aligned to Reference"

# ---------------------------------------------------------
# COLLECTOR
# ---------------------------------------------------------
class MonthlyCollector:
    def __init__(self, reporter):
        self.catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        self.reporter = reporter
        self.enforcer = ConsistencyEnforcer()

    def get_monthly_targets(self):
        targets = []
        for year in range(Config.START_YEAR, Config.END_YEAR + 1):
            for month in range(1, 13):
                # Don't predict future
                if year == datetime.now().year and month > datetime.now().month:
                    break
                targets.append((year, month))
        return targets

    def find_best_item(self, year, month):
        # Format date query for the whole month
        last_day = 31 if month in [1,3,5,7,8,10,12] else 30
        if month == 2: last_day = 28 # Simplified
        
        date_query = f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day}"
        
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=Config.RIYADH_BBOX,
            datetime=date_query,
            query={"eo:cloud_cover": {"lt": Config.MAX_CLOUD}}
        )
        items = list(search.items())
        
        if not items:
            return None, 0, []

        # Find best coverage
        best_group = None
        best_date = None
        max_score = -1 # Score = Coverage - (Cloud/100)
        
        # Group by date first
        by_date = {}
        for item in items:
            d = item.datetime.strftime("%Y-%m-%d")
            by_date.setdefault(d, []).append(item)
            
        for d, day_items in by_date.items():
            # Geometric Coverage
            polys = [shape(i.geometry) for i in day_items]
            unified = unary_union(polys)
            intersection = unified.intersection(Config.RIYADH_BBOX)
            coverage_pct = (intersection.area / Config.RIYADH_BBOX.area) * 100
            
            # Avg Cloud cover
            avg_cloud = np.mean([i.properties.get("eo:cloud_cover", 100) for i in day_items])
            
            # We prioritize Coverage, then Low Cloud
            if coverage_pct < Config.MIN_COVERAGE:
                continue
                
            score = coverage_pct - avg_cloud
            
            if score > max_score:
                max_score = score
                best_group = day_items
                best_date = d
                
        return best_date, max_score, best_group

    def process_month(self, year, month):
        month_str = f"{year}-{month:02d}"
        out_dir = os.path.join(Config.RAW_DIR, month_str)
        
        # 1. Search
        best_date, score, items = self.find_best_item(year, month)
        if not best_date:
            return "MISSING (No suitable data)"
            
        # 2. Setup Output
        os.makedirs(out_dir, exist_ok=True)
        self.reporter.add(f"  > Date Found: {best_date} (Score: {score:.1f})")
        
        # 3. Process All Bands
        # We process B04 first to establish/check reference grid
        ordered_bands = ["B04"] + [b for b in Config.BANDS if b != "B04"]
        
        for band in ordered_bands:
            out_path = os.path.join(out_dir, f"{band}.tif")
            
            if os.path.exists(out_path):
                # If existing, we assume it's good, IF we have a reference.
                # If we don't have a reference yet, we must read it to set it.
                if self.enforcer.reference_profile is None and band == "B04":
                    with rasterio.open(out_path) as src:
                        self.enforcer.set_reference(src.profile, out_path)
                continue

            sources = []
            try:
                # 3a. Download URLs
                for item in items:
                    href = pc.sign(item).assets[band].href
                    sources.append(rasterio.open(href))
                
                # 3b. Determine Mosaic Bounds (UTM)
                if self.enforcer.reference_profile and band != "B04":
                    # Reuse reference bounds/crs logic implicitly via reproject later?
                    # No, for mosaicing we need common ground.
                    # We merge normally, THEN align to reference.
                    dst_crs = self.enforcer.reference_profile['crs']
                    # This transform_bounds can be approximate, merge handles it
                else:
                    # First time setup (or independent merge)
                    dst_crs = sources[0].crs
                    
                ref_src = sources[0]
                dst_bounds = transform_bounds("EPSG:4326", dst_crs, *Config.RIYADH_BBOX.bounds)
                
                # 3c. Mosaic
                mosaic, out_trans = merge(sources, bounds=dst_bounds)
                
                # 3d. Update Meta
                out_meta = sources[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "crs": dst_crs,
                    "count": 1
                })
                
                # 3e. CONSISTENCY CHECK & SAVE
                # We save to a temporary memory file or disk, then enforce consistency
                temp_path = out_path + ".temp.tif"
                with rasterio.open(temp_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                    
                # Enforce
                with rasterio.open(temp_path) as src:
                    # If this is the very first image (B04 of first month), it becomes Reference.
                    if self.enforcer.reference_profile is None and band == "B04":
                        self.enforcer.set_reference(src.profile, out_path)
                        # Just rename temp to final
                        src.close() # Close to allow move
                        if os.path.exists(out_path): os.remove(out_path)
                        os.rename(temp_path, out_path)
                    else:
                        # Reproject temp to final using reference
                        self.enforcer.enforce(src, out_path)
                        # Clean up temp
                        src.close()
                        os.remove(temp_path)
                        
            except Exception as e:
                self.reporter.add(f"    ! Error on {band}: {e}")
                traceback.print_exc()
            finally:
                for s in sources: s.close()
                
        return f"SUCCESS ({best_date})"

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    Config.setup()
    reporter = SummaryWriter(Config.REPORT_FILE)
    collector = MonthlyCollector(reporter)
    
    reporter.print_and_add("=== RIYADH SATELLITE DATA COLLECTION ===")
    reporter.print_and_add(f"Time Range: {Config.START_YEAR} to {Config.END_YEAR}")
    reporter.print_and_add(f"Coverage Target: {Config.MIN_COVERAGE}%")
    reporter.print_and_add("========================================")
    
    targets = collector.get_monthly_targets()
    
    # Progress Bar
    pbar = tqdm(targets, desc="Collecting Months", unit="mo")
    
    success_count = 0
    
    for year, month in pbar:
        month_label = f"{year}-{month:02d}"
        pbar.set_description(f"Processing {month_label}")
        
        status = collector.process_month(year, month)
        
        if "SUCCESS" in status:
            success_count += 1
            
        reporter.add(f"{month_label}: {status}")
        
    reporter.print_and_add("========================================")
    reporter.print_and_add(f"Collection Complete.")
    reporter.print_and_add(f"Total Months Gathered: {success_count} / {len(targets)}")
    reporter.print_and_add(f"Data saved to: {Config.RAW_DIR}")

if __name__ == "__main__":
    main()
