import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import cartopy.io.shapereader as shpreader
import ssl
from fiona import Path

def makeNYGrid(km = 3):

    """
    This function returns a grid of evenly spaced squares across the entire coordinate space
    of New York state. 

    Inputs
    km: desired grid sizing in kilometers. Default is 3

    Outputs
    grid_gdf: GeoDataframe of geometries
    centers_gdf: Geodataframe of grid centers
    coords_list: List of centers as lon, lat tuples
    """

    url = "data/ny_cartography_data/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
    gdf = gpd.read_file(url)

    # Filter for New York
    ny_state = gdf[gdf['name'] == 'New York']

    ny_geometry = ny_state.geometry

    # Define NY bounding box min lon, min lat, max lon, max lat
    min_lon, min_lat, max_lon, max_lat = ny_geometry.total_bounds

    # Define step sizes for X km grid
    lat_step = 0.009*km  # ~X km in latitude
    lon_step = 0.011*km  # ~X km in longitude at NY's latitude

    # Generate grid squares
    grid_polygons = []
    grid_centers = []

    lat_values = np.arange(min_lat, max_lat, lat_step)
    lon_values = np.arange(min_lon, max_lon, lon_step)

    ny_geometry = ny_state.geometry.iloc[0]

    for lat in lat_values:
        for lon in lon_values:
            # Define a square polygon (1x1 km)
            square = Polygon([
                (lon, lat),
                (lon + lon_step, lat),
                (lon + lon_step, lat + lat_step),
                (lon, lat + lat_step),
                (lon, lat)  # Close the polygon
            ])
            
            #Save portion of square that is inside NY State
            square = square.intersection(ny_geometry)
            #Record center of grid section
            center = square.centroid
            
            #Save the square if it is non empty
            if square:
                grid_polygons.append(square)
                grid_centers.append(center)
                
    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs="EPSG:4326")
    centers_gdf = gpd.GeoDataFrame(geometry=grid_centers, crs="EPSG:4326")
    coords_list = [(row.x, row.y) for row in centers_gdf['geometry'].values]

    return grid_gdf, centers_gdf, coords_list


