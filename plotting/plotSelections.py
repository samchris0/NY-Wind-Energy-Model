def plot_samples(coords_dict, k, dist, show_plot=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import geopandas as gpd
    from cartopy import crs as ccrs
    from cartopy.feature import ShapelyFeature
    from shapely.geometry import box

    # Load NY state geometry
    url = "data/ny_cartography_data/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
    gdf = gpd.read_file(url)
    ny_state = gdf[gdf['name'] == 'New York']
    ny_geometry = ny_state.geometry.iloc[0]

    # Set map extent
    map_extent = [-79.9, -71.75, 40.45, 45.15]

    # Load wind turbine dataset
    df = pd.read_csv("data/uswtdb_V8_0_20250225.csv")
    df_ny = df[df["t_state"] == "NY"]
    ny_turbine_coords = list(zip(df_ny["ylat"], df_ny["xlong"]))

    # Function to draw a NY map and plot turbine points and highlight set
    def draw_ny_map(ax, highlight_coords, title=""):
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.spines['geo'].set_visible(False)

        ny_feature = ShapelyFeature([ny_geometry], ccrs.PlateCarree(), facecolor='lightgray', linewidth=0.3)
        ax.add_feature(ny_feature)

        for lat, lon in ny_turbine_coords:
            ax.plot(lon, lat, marker='o', color='grey', markersize=3, alpha=0.5, transform=ccrs.PlateCarree())

        for lat, lon in highlight_coords:
            ax.plot(lon, lat, marker='o', color='blue', markersize=6, transform=ccrs.PlateCarree())

        ax.set_title(title, fontsize=20)

    # Create 2x2 grid of subplots with PlateCarree projection
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    keys = list(coords_dict.keys())

    for i in range(4):
        coords = coords_dict[keys[i]]
        draw_ny_map(axes[i], coords, title=keys[i])

    plt.tight_layout()
    fig.subplots_adjust(hspace=-0.35, wspace=0.02, top=1.1)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Wind Turbines',
               markerfacecolor='grey', markersize=8, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='Selected Sensors',
               markerfacecolor='blue', markersize=8),
        Line2D([0], [0], linestyle='None', label=f'k = {k}'),
        Line2D([0], [0], linestyle='None', label=f'radius = {dist} km')
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=4,
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    if show_plot:
        plt.show()
 
    else:
        plt.close(fig)
        return fig
