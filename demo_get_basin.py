#! /usr/bin/env python -tt
# -*- coding: utf-8; mode: python -*-
r"""

demo_get_basin
~~~~~~~~~~~~~~
"""
# Standard Imports
import os
import pickle
import glob

# Third-Party Imports
import numpy as np
import numpy.typing as npt
import numpy.ma as ma
import pandas
import geopandas
import cartopy.crs as ccrs
# import cartopy
# import pyproj
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cartopy.feature as cf
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon

# STARE Imports
import pystare
import starepandas

# Local Imports

##
# List of Public objects from this module.
__all__ = ['demo_get_basin']

##
# Markup Language Specification (see NumpyDoc Python Style Guide https://numpydoc.readthedocs.io/en/latest/format.html)
__docformat__ = "Numpydoc"
# ------------------------------------------------------------------------------

# Define Global Constants and State Variables
# -------------------------------------------

##
# Select base dataset to work with
SRC_DSET = ["SNODAS", "IMERG", "MODIS"][0]

##
# Select target area/basin to work with
BASIN = ["SIERRA NEVADA", "FEATHER RIVER", "TUOLUMNE RIVER"][0]

##
# Select STARE encoding level
"""
Q-00 ~Length Scale: 10240 km      Q-14  ~Length Scale:        0.625 km
Q-01 ~Length Scale:  5120 km      Q-15  ~Length Scale:       0.3125 km
Q-02 ~Length Scale:  2560 km      Q-16  ~Length Scale:      0.15625 km
Q-03 ~Length Scale:  1280 km      Q-17  ~Length Scale:     0.078125 km
Q-04 ~Length Scale:   640 km      Q-18  ~Length Scale:    0.0390625 km
Q-05 ~Length Scale:   320 km      Q-19  ~Length Scale:    0.0195312 km
Q-06 ~Length Scale:   160 km      Q-20  ~Length Scale:   0.00976562 km
Q-07 ~Length Scale:    80 km      Q-21  ~Length Scale:   0.00488281 km
Q-08 ~Length Scale:    40 km      Q-22  ~Length Scale:   0.00244141 km
Q-09 ~Length Scale:    20 km      Q-23  ~Length Scale:    0.0012207 km
Q-10 ~Length Scale:    10 km      Q-24  ~Length Scale:  0.000610352 km
Q-11 ~Length Scale:     5 km      Q-25  ~Length Scale:  0.000305176 km
Q-12 ~Length Scale:   2.5 km      Q-26  ~Length Scale:  0.000152588 km
Q-13 ~Length Scale:  1.25 km      Q-27  ~Length Scale:  7.62939E-05 km

"""
#       0   1   2   3   4   5   6
QLEV = [10, 12, 14, 15, 16, 17, 18][6]

#             0   1   2   3   4   5   6
QLEV_BASIN = [10, 12, 14, 15, 16, 17, 18][6]

#             0   1   2   3   4   5   6
QLEV_DGRID = [10, 12, 14, 15, 16, 17, 18][2]

##
# Parallel computing settings.
N_PARTS = 1
N_CORES = 1

def LON_TO_180(x): return ((x + 180.0) % 360.0) - 180.0
def LON_TO_360(x): return (x + 360.0) % 360.0
PLOT_DPI = 600
LON_0_GLOBAL = 0
GLOBE = ccrs.Globe(datum='WGS84', ellipse='WGS84')
GEOD_CRS = ccrs.Geodetic(globe=GLOBE)
FLAT_CRS = ccrs.PlateCarree()
PC_CRS = ccrs.PlateCarree(central_longitude=LON_0_GLOBAL)

##
# Set data source base path
if SRC_DSET == "SNODAS":
    SRC_DIR = "/Volumes/saved/data/SNODAS/"
    SRC_SDIR = ("ssmv11034tS__T0001/", "ssmv11038wS__A0024/", "ssmv11044bS__T0024/", "ssmv11036tS__T0001/")[0]
elif SRC_DSET == "IMERG":
    SRC_DIR = "/Volumes/saved/data/GOES/IMERG/"
    SRC_SDIR = ''
elif SRC_DSET == "MODIS":
    SRC_DIR = "/Volumes/saved/data/MODIS/"
    SRC_SDIR = ''
ALT_DIR = "/Volumes/saved/hidden/STAREpull/"
COM_DIR = "/Volumes/saved/common/"

###############################################################################
# PUBLIC ij2grid()
# ----------------
def ij2grid(j: int, i: int, im: int, jm: int) -> int:
    """Returns 1d grid index of an jm,im 'row-major' array.
        i the 'x' index between 0 , im-1 where im is index max
        j the 'y' index between 0 , jm-1 where jm is index max
    """
    k = None
    if i < im and j < jm:
        k = j * im + i
    if k is None:
        raise ValueError(f"Error in ij2grid(): j = {j:d}, i = {i:d} im = {im:d} jm = {jm:d}")
    return int(k)

###############################################################################
# PUBLIC plot_conusw()
# --------------------
def plot_conusw(pname: str, snodas_lats2d: npt.ArrayLike, snodas_lons2d: npt.ArrayLike, snodas_npts: int, snodas_nlats: int, snodas_nlons: int, snodas_w_masked_gids: list[int]) -> None:
    """Plot CONUS West SNODAS grids.

    Parameters
    ----------
    pname : str
        Plot name/path
    snodas_lats2d : npt.ArrayLike
        Datagrid latitudes
    snodas_lons2d : npt.ArrayLike
        Datagrid longitudes
    snodas_npts : int
        Number of SNODAS grids
    snodas_nlats : int
        Number of SNODAS grid latitudes
    snodas_nlons : int
        Number of SNODAS grid longitudes
    snodas_w_masked_gids : list[int]
        List of SNODAS grid indices
    """
    opts = {'projection': PC_CRS}
    fig, geo_axes = plt.subplots(figsize=(16, 9), dpi=PLOT_DPI, subplot_kw=opts)
    min_lat, max_lat = np.min(snodas_lats2d), np.max(snodas_lats2d)
    min_lon, max_lon = LON_TO_360(np.min(snodas_lons2d)), LON_TO_360(np.max(snodas_lons2d))
    map_extent = (LON_TO_180(min_lon), LON_TO_180(max_lon), min_lat, max_lat)
    geo_axes.set_extent(map_extent, crs=FLAT_CRS)
    extent = (min_lon, max_lon, min_lat, max_lat)
    # Add coastlines
    geo_axes.add_feature(cf.COASTLINE)
    geo_axes.add_feature(cf.BORDERS)
    tmp = np.zeros((snodas_npts,))
    tmp[snodas_w_masked_gids] = 0.3
    tmp = ma.masked_equal(tmp, 0)
    tmp = np.reshape(tmp, (snodas_nlats, snodas_nlons))
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    plt.pcolormesh(snodas_lons2d, snodas_lats2d, tmp, transform=PC_CRS, cmap='jet', norm=norm, alpha=1)
    fig.savefig(pname, dpi=PLOT_DPI, facecolor='w', edgecolor='w',
                orientation='landscape', bbox_inches='tight', pad_inches=0.02)
    plt.clf()
    plt.close('all')
    return

###############################################################################
# PUBLIC plot_conus_regions()
# ---------------------------
def plot_conus_regions(pname: str, snodas_lats2d: npt.ArrayLike, snodas_lons2d: npt.ArrayLike, snodas_npts: int, snodas_nlats: int, snodas_nlons: int) -> None:
    """Plot CONUS example regions.

    Parameters
    ----------
    pname : str
        Plot name/path
    snodas_lats2d : npt.ArrayLike
        Datagrid latitudes
    snodas_lons2d : npt.ArrayLike
        Datagrid longitudes
    snodas_npts : int
        Number of SNODAS grids
    snodas_nlats : int
        Number of SNODAS grid latitudes
    snodas_nlons : int
        Number of SNODAS grid longitudes
    """
    opts = {'projection': PC_CRS}
    fig, geo_axes = plt.subplots(figsize=(16, 9), dpi=PLOT_DPI, subplot_kw=opts)
    min_lat, max_lat = np.min(snodas_lats2d), np.max(snodas_lats2d)
    min_lon, max_lon = LON_TO_360(np.min(snodas_lons2d)), LON_TO_360(np.max(snodas_lons2d))
    map_extent = (LON_TO_180(min_lon), LON_TO_180(max_lon), min_lat, max_lat)
    geo_axes.set_extent(map_extent, crs=FLAT_CRS)
    # Add coastlines
    geo_axes.add_feature(cf.COASTLINE)
    geo_axes.add_feature(cf.BORDERS)
    plt.plot([-110, -110], [min_lat, max_lat], color='blue', linewidth=1, marker='o', markersize=3)
    poly = Polygon([(LON_TO_180(min_lon), min_lat), (-110, min_lat), (-110, max_lat), (LON_TO_180(min_lon), max_lat)], facecolor='blue', edgecolor='k', linewidth=1.5)
    plt.gca().add_patch(poly)
    fig.savefig(pname, dpi=PLOT_DPI, facecolor='w', edgecolor='w',
                orientation='landscape', bbox_inches='tight', pad_inches=0.02)
    plt.clf()
    plt.close('all')
    return

###############################################################################
# PUBLIC plot_grids()
# --------------------
def plot_grids(pname: str, snodas_lats2d: npt.ArrayLike, snodas_lons2d: npt.ArrayLike, snodas_npts: int, snodas_nlats: int, snodas_nlons: int, the_grids: list[int], basin_roi: starepandas.staredataframe.STAREDataFrame, dgrid_roi: starepandas.staredataframe.STAREDataFrame, add_trixels: bool, flip_trixels: bool, trixel_double: bool, zoomin: bool) -> None:
    """Plot basin grids and trixels

    Parameters
    ----------
    pname : str
        Plot name/path
    snodas_lats2d : npt.ArrayLike
        Datagrid latitudes
    snodas_lons2d : npt.ArrayLike
        Datagrid longitudes
    snodas_npts : int
        Number of SNODAS grids
    snodas_nlats : int
        Number of SNODAS grid latitudes
    snodas_nlons : int
        Number of SNODAS grid longitudes
    the_grids : list[int]
        List of SNODAS grid basin indices to plot
    basin_roi : starepandas.staredataframe.STAREDataFrame
        Basin STARE DF with trixels
    dgrid_roi : starepandas.staredataframe.STAREDataFrame
        Datagrid STARE DF with trixels
    add_trixels : bool
        Flag to add basin trixels
    flip_trixels : bool
        Flag to do a reverse and show datagrid trixels and basin polygon (use_trixels must be True)
    trixel_double: bool
        Flag to show basin and datagrid trixels rather than datagrids
    zoomin : bool
        Flag to zoom in on basin
    """
    opts = {'projection': PC_CRS}
    fig, geo_axes = plt.subplots(figsize=(16, 9), dpi=PLOT_DPI, subplot_kw=opts)
    min_lat, max_lat = np.min(snodas_lats2d), np.max(snodas_lats2d)
    min_lon, max_lon = LON_TO_360(np.min(snodas_lons2d)), LON_TO_360(np.max(snodas_lons2d))
    map_extent = (LON_TO_180(min_lon), LON_TO_180(max_lon), min_lat, max_lat)
    use_dpi = PLOT_DPI

    if zoomin:
        use_dpi = PLOT_DPI * 2
        if pname.find("sierra") != -1:
            map_extent = (LON_TO_180(-124.5), LON_TO_180(-117.5), 35, 42.2)
        elif pname.find("feather") != -1:
            map_extent = (LON_TO_180(-122.1), LON_TO_180(-119.6), 39.3, 40.6)
        elif pname.find("tuolumne") != -1:
            map_extent = (LON_TO_180(-120.68), LON_TO_180(-119.1), 37.5, 38.3)

    geo_axes.set_extent(map_extent, crs=FLAT_CRS)
    extent = (min_lon, max_lon, min_lat, max_lat)
    # Add coastlines
    geo_axes.add_feature(cf.COASTLINE)
    geo_axes.add_feature(cf.BORDERS)
    tmp = np.zeros((snodas_npts,))
    tmp[the_grids] = 0.3
    tmp = ma.masked_equal(tmp, 0)
    tmp = np.reshape(tmp, (snodas_nlats, snodas_nlons))
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    if flip_trixels:
            ##
            # Plot the basin datagrids
            plt.pcolormesh(snodas_lons2d, snodas_lats2d, tmp, transform=PC_CRS, cmap='jet', norm=norm, zorder=1)
            #
            # Plot the trixels
            dgrid_roi.plot(ax=geo_axes, trixels=True, boundary=True, figsize=(16, 9), aspect=None, zorder=2,
                           linewidth=0.5, color='r', alpha=0.7, transform=GEOD_CRS)
            ##
            # Plot the basin polygon
            if pname.find("sierra") != -1:
                ##
                # Plot Sierra Nevada Conservancy Area
                basin_dat_proj = basin_roi.to_crs(PC_CRS)
                basin_dat_proj.plot(ax=geo_axes, kind='geo', facecolor='none', edgecolor='k', linewidth=0.2, zorder=3)
            elif pname.find("feather") != -1:
                ##
                # Plot the Feather River Watershed Improvement Program (WIP) Boundary
                basin_dat_proj = basin_roi.to_crs(PC_CRS)
                # basin_dat_proj.plot(ax=geo_axes, kind='geo', facecolor='none', edgecolor='k', linewidth=0.2, zorder=4)
                basin_dat_proj.plot(ax=geo_axes, kind='geo', facecolor='k', edgecolor='k', linewidth=0.2, alpha=0.25, zorder=3)
            elif pname.find("tuolumne") != -1:
                ##
                # Plot the Tuolumne River Watershed Improvement Program (WIP) Boundary
                basin_dat_proj = basin_roi.to_crs(PC_CRS)
                basin_dat_proj.plot(ax=geo_axes, kind='geo', facecolor='none', edgecolor='k', linewidth=0.2, zorder=3)
    elif trixel_double:
            # Plot the trixels
            dgrid_roi.plot(ax=geo_axes, trixels=True, boundary=True, figsize=(16, 9), aspect=None, zorder=2,
                           linewidth=0.5, color='b', alpha=0.7, transform=GEOD_CRS)
            basin_roi.plot(ax=geo_axes, trixels=True, boundary=True, figsize=(16, 9), aspect=None, zorder=3,
                           linewidth=0.25, color='r', alpha=1, transform=GEOD_CRS)
    else:
        if add_trixels:
            # Plot the trixels
            basin_roi.plot(ax=geo_axes, trixels=True, boundary=True, figsize=(16, 9), aspect=None, zorder=3,
                           linewidth=0.5, color='r', alpha=0.7, transform=GEOD_CRS)
            plt.pcolormesh(snodas_lons2d, snodas_lats2d, tmp, transform=PC_CRS, cmap='jet', norm=norm, zorder=2)
        else:
            plt.pcolormesh(snodas_lons2d, snodas_lats2d, tmp, transform=PC_CRS, cmap='jet', norm=norm, zorder=2)

    fig.savefig(pname, dpi=use_dpi, facecolor='w', edgecolor='w',
                orientation='landscape', bbox_inches='tight', pad_inches=0.02)
    plt.clf()
    plt.close('all')
    return

###############################################################################
# PUBLIC demo_get_basin()
# -------------------------
def demo_get_basin(mixed_qlev: bool, shrink_conusw: bool, make_basins: bool, make_dgrid: bool, make_grids:bool, make_plot: bool, use_trixels: bool, trixel_flip: bool, double_trixel: bool, verbose: bool) -> None:
    """Demo to get basin grids and trixels.

    Parameters
    ----------
    mixed_qlev : bool
        Flag to use Use QLEV_BASIN and QLEV_DGRID rather than unified QLEV encodings
    shrink_conusw: bool
        Flag to use a narrow version of CONUS_W for faster graphics.
    make_basins : bool
        Flag to make basins STARE dataframes and pickle (read polygon and make sids), rather than read from a pickle file.
    make_dgrid : bool
        Flag to make datagrid STARE dataframes and pickle, rather than read from a pickle file.
    make_grids : bool
        Flag to using basin and datagrid STARE dataframes to find and save intersecting basin datagrids, rather than read from a pickle file.
    make_plot : bool
        Flag to plot things.
    use_trixels : bool
        Flag to use basin trixels in some plots.
    trixel_flip: bool
        Flag to do a reverse and show datagrid trixels and basin polygon (use_trixels must be True)
    double_trixel: bool
        Flag to show basin and datagrid trixels rather than datagrids
    verbose : bool
        Flag to print verbose output.
    """
    if verbose:
        print(f"\nUsing QLEV {QLEV}")

    if mixed_qlev:
        sierra_nevada_pkl_file = f"{ALT_DIR}sierra_nevada_{QLEV_BASIN:02d}.pkl"
        feather_river_pkl_file = f"{ALT_DIR}feather_river_{QLEV_BASIN:02d}.pkl"
        tuolumne_river_pkl_file = f"{ALT_DIR}tuolumne_river_{QLEV_BASIN:02d}.pkl"
    else:
        sierra_nevada_pkl_file = f"{ALT_DIR}sierra_nevada_{QLEV:02d}.pkl"
        feather_river_pkl_file = f"{ALT_DIR}feather_river_{QLEV:02d}.pkl"
        tuolumne_river_pkl_file = f"{ALT_DIR}tuolumne_river_{QLEV:02d}.pkl"

    if make_basins:
        use_qlev = QLEV_BASIN if mixed_qlev else QLEV
        ##
        # Get Sierra Nevada Basin
        sierra_nevada_file = f"{ALT_DIR}Sierra_Nevada_Conservancy_Boundary.geojson"
        sierra_nevada_gdf = geopandas.read_file(sierra_nevada_file)
        sierra_nevada_gdf = sierra_nevada_gdf.set_crs(epsg=4326)
        sierra_nevada_gdf_proj = sierra_nevada_gdf.to_crs(GEOD_CRS)
        # #
        # Get SIDs and cover geopandas.GeoDataFrame
        sierra_nevada_sids = starepandas.sids_from_gdf(sierra_nevada_gdf_proj, level=use_qlev)
        sierra_nevada_roi = starepandas.STAREDataFrame(sierra_nevada_gdf_proj, sids=sierra_nevada_sids)
        ##
        # Add Trixels
        _trixels = sierra_nevada_roi.make_trixels(sid_column='sids', wrap_lon=False, n_partitions=N_PARTS, num_workers=N_CORES)
        sierra_nevada_roi.set_trixels(_trixels, inplace=True)
        if verbose:
            print(f"\nSaved {sierra_nevada_pkl_file}")
        with open(sierra_nevada_pkl_file, 'wb') as f:
            pickle.dump(sierra_nevada_roi, f)

        ##
        # Get Feather River Basin
        feather_river_file = f"{ALT_DIR}WIP_Assessment_Areas.geojson"
        feather_gdf = geopandas.read_file(feather_river_file)
        feather_wip_gdf = feather_gdf[feather_gdf.WIP_AA=="Feather"]
        ##
        # Get SIDs and cover geopandas.GeoDataFrame
        feather_wip_gdf.reset_index(inplace=True, drop=True)
        feather_wip_gdf_proj = feather_wip_gdf.to_crs(GEOD_CRS)
        feather_sids = starepandas.sids_from_gdf(feather_wip_gdf_proj, level=use_qlev) # , force_ccw=True
        feather_river_roi = starepandas.STAREDataFrame(feather_wip_gdf_proj, sids=feather_sids)
        ##
        # Add Trixels
        _trixels = feather_river_roi.make_trixels(sid_column='sids', wrap_lon=False, n_partitions=N_PARTS, num_workers=N_CORES)
        feather_river_roi.set_trixels(_trixels, inplace=True)
        if verbose:
            print(f"Saved {feather_river_pkl_file}")
        with open(feather_river_pkl_file, 'wb') as f:
            pickle.dump(feather_river_roi, f)

        ##
        # Get Tuolumne River Basin
        tuolumne_river_file = f"{ALT_DIR}WIP_Assessment_Areas.geojson"
        tuolumne_gdf = geopandas.read_file(tuolumne_river_file)
        tuolumne_wip_gdf = tuolumne_gdf[tuolumne_gdf.WIP_AA=="Tuolumne"]
        ##
        # Get SIDs and cover geopandas.GeoDataFrame
        tuolumne_wip_gdf.reset_index(inplace=True, drop=True)
        tuolumne_wip_gdf_proj = tuolumne_wip_gdf.to_crs(GEOD_CRS)
        tuolumne_sids = starepandas.sids_from_gdf(tuolumne_wip_gdf_proj, level=use_qlev) # , force_ccw=True
        tuolumne_river_roi = starepandas.STAREDataFrame(tuolumne_wip_gdf_proj, sids=tuolumne_sids)
        ##
        # Add Trixels
        _trixels = tuolumne_river_roi.make_trixels(sid_column='sids', wrap_lon=False, n_partitions=N_PARTS, num_workers=N_CORES)
        tuolumne_river_roi.set_trixels(_trixels, inplace=True)
        if verbose:
            print(f"Saved {tuolumne_river_pkl_file}")
        with open(tuolumne_river_pkl_file, 'wb') as f:
            pickle.dump(tuolumne_river_roi, f)
    else:
        if verbose:
            print(f"\nReading {sierra_nevada_pkl_file}")
        with open(sierra_nevada_pkl_file, 'rb') as f:
            sierra_nevada_roi = pickle.load(f)
        if verbose:
            print(f"Reading {feather_river_pkl_file}")
        with open(feather_river_pkl_file, 'rb') as f:
            feather_river_roi = pickle.load(f)
        if verbose:
            print(f"Reading {tuolumne_river_pkl_file}")
        with open(tuolumne_river_pkl_file, 'rb') as f:
            tuolumne_river_roi = pickle.load(f)

    ##
    # Read a src data file
    if SRC_DSET == "SNODAS":
        use_qlev = QLEV_DGRID if mixed_qlev else QLEV
        basin_file = f"{ALT_DIR}snodas_{use_qlev:02d}.pkl"
        if shrink_conusw:
            basin_file = basin_file.replace(".pkl", "_shrink.pkl")
        if mixed_qlev:
            grid_file = f"{ALT_DIR}snodas_basin_grids_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.pkl"
        else:
            grid_file = f"{ALT_DIR}snodas_basin_grids_{use_qlev:02d}.pkl"
        if make_dgrid:
            ##
            # Read the SNODAS datagrid lon/lats from src geotif
            ffiles = sorted(glob.glob(f'{SRC_DIR}{SRC_SDIR}*'))
            file_path = ffiles[0]
            if verbose:
                print(f"\nReading {file_path}")
            sdf_full = starepandas.read_geotiff(file_path, add_pts=True, add_latlon=True, add_coordinates=False, add_xy=False,
                                                add_sids=False, add_trixels=False, n_workers=N_CORES)
            sdf_full = sdf_full.rename(columns={"lat": "lons", "lon": "lats"})

            snodas_lats2d = sdf_full['lats'][:]
            snodas_lats2d = snodas_lats2d.to_numpy(dtype=np.float32)
            snodas_lats2d = snodas_lats2d.reshape((3351, 6935))
            snodas_lons2d = sdf_full['lons'][:]
            snodas_lons2d = snodas_lons2d.to_numpy(dtype=np.float32)
            snodas_lons2d = snodas_lons2d.reshape((3351, 6935))
            snodas_lats = snodas_lats2d[:,0]
            snodas_lons = snodas_lons2d[0,:]
            snodas_nlats, snodas_nlons = (len(snodas_lats), len(snodas_lons))
            snodas_npts = snodas_nlats * snodas_nlons
            if verbose:
                """
                SNODAS Domain:
                    snodas_lats          : (3351,)  +24.954 <-> +52.871
                    snodas_lons          : (6935,)  -124.73 <-> -66.946
                    snodas_npts          : 23239185
                """
                print("\nSNODAS Domain:")
                print(f"\tsnodas_lats          : ({snodas_nlats}) {snodas_lats.min():>+8.5g} <-> {snodas_lats.max():<+8.5g}")
                print(f"\tsnodas_lons          : ({snodas_nlons}) {snodas_lons.min():>+8.5g} <-> {snodas_lons.max():<+8.5g}")
                print(f"\tsnodas_npts          : {snodas_npts:<20d}")

            ##
            # Form SNODAS datagrid
            r"""
            Using an origin in the upper right corner:
                snodas_gids (5924568 of 23239185): [0, ... 23234017]
                    gidx          0 (+24.9537, -124.7296) Lower Left  Corner
                    gidx       6934 (+24.9537, -066.9463) Lower Right Corner
                    gidx   23232250 (+52.8704, -124.7296) Upper Left  Corner
                    gidx   23239184 (+52.8704, -066.9463) Upper Right Corner

            Using an origin in the upper left corner
                    gidx          0 (+52.8704, -124.7296) Upper Left  Corner
                    gidx       6934 (+52.8704, -066.9463) Upper Right Corner
                    gidx   23232250 (+24.9537, -124.7296) Lower Left  Corner
                    gidx   23239184 (+24.9537, -066.9463) Lower Right Corner
            """
            if verbose:
                print("\nSNODAS Data-grid")
            # snodas_gids = []
            snodas_gids = np.zeros((snodas_npts,), dtype=np.int32)
            k_idx = -1
            for j_idx, j_lat in enumerate(snodas_lats[::-1]):
                for i_idx, i_lon in enumerate(snodas_lons):
                    gidx = ij2grid(j_idx, i_idx, snodas_nlons, snodas_nlats)
                    # snodas_gids.append(gidx)
                    k_idx += 1
                    snodas_gids[k_idx] = gidx
                    if verbose:
                        if gidx in [0, snodas_nlons - 1, snodas_npts - snodas_nlons, snodas_npts - 1]:
                            print(f"\tgidx {gidx:10d} ({j_lat:+8.4f}, {i_lon:+8.4f})")
            if verbose:
                print(f"\n\tsnodas_gids ({len(snodas_gids)} of {snodas_npts}): [{snodas_gids[0]}, ... {snodas_gids[-1]}]")
            sdf_full['gids'] = snodas_gids

            """
            Here SNODAS_masked is roughly CONUS land + the Columbia River watershed in Canada - larger water bodies like the Great Lakes.
            The parameter 'snodas_npts' is the number of 1 km SNODAS grids.

            snodas_gids (23239185 of 23239185)         : [0, ... 23239184]
            snodas_external_npts                       : 11249583
            snodas_external_gids (11249583 of 23239185): [0, ... 23239184]
            conus_masked_npts                          : 11989602
            """
            # Area where sdf_full masked by -9999
            sdf_extra = sdf_full[sdf_full['band_1'] == -9999]
            snodas_external_npts = sdf_extra.shape[0]
            snodas_external_gids = sdf_extra['gids'].tolist()
            if verbose:
                print(f"\tsnodas_external_npts  : {snodas_external_npts:<20d}")
                print(f"\tsnodas_external_gids ({len(snodas_external_gids)} of {snodas_npts}): [{snodas_external_gids[0]}, ... {snodas_external_gids[-1]}]")

            # Area where sdf_full isn't masked by -9999
            sdf_masked = sdf_full[sdf_full['band_1'] >= 0]
            conus_masked_npts = sdf_masked.shape[0]
            if verbose:
                print(f"\tconus_masked_npts    : {conus_masked_npts:<20d}")

            ##
            # Trim to western CONUS
            """
            SNODAS_W is everything with west of longitude -110.

            conusw_npnts                                :  5924568
            conusw_external_npts                        :  2674288
            conusw_masked_npts                          :  3250280

            snodas_w_npts                               : 5924568
            snodas_w_npts                               : 25.49%
            snodas_w_gids (11249583 of 23239185)        : [0, ... 23239184]
            snodas_w_external_npts                      : 2674288
            snodas_w_external_gids (2674288 of 23239185): [0, ... 23234017]
            """
            if shrink_conusw:
                # Feather River
                # Latitude: 40d 21' 28.19" N
                # Longitude: -120d 27' 2.99" W
                sdfw = sdf_full[sdf_full['lons'] <= -115]
                sdfw = sdfw[sdfw['lats'] <= 45]
            else:
                sdfw = sdf_full[sdf_full['lons'] <= -110]
            sdfw.reset_index(drop=True, inplace = True)
            conusw_npnts = sdfw.shape[0]

            sdfw_extra = sdfw[sdfw['band_1'] == -9999]
            conusw_external_npts = sdfw_extra.shape[0]

            sdfw_masked = sdfw[sdfw['band_1'] >= 0]
            snodas_w_masked_npts = conusw_masked_npts = sdfw_masked.shape[0]
            snodas_w_masked_gids = sdfw_masked['gids'].tolist()

            snodas_w_sdf = starepandas.STAREDataFrame(sdfw, level=use_qlev)
            lons = snodas_w_sdf['lons'].values
            lats = snodas_w_sdf['lats'].values

            ##
            # Add SIDs
            sids = pystare.from_latlon(lats, lons, use_qlev)
            snodas_w_sdf.set_sids(sids, inplace=True)
            snodas_w_npts = sdfw.shape[0]
            snodas_w_gids = sdf_extra['gids'].tolist()
            ##
            # Add Trixels
            _trixels = snodas_w_sdf.make_trixels(sid_column='sids', wrap_lon=False, n_partitions=N_PARTS, num_workers=N_CORES)
            snodas_w_sdf.set_trixels(_trixels, inplace=True)
            if verbose:
                print(f"\n\tconusw_npnts         : {conusw_npnts:<20d}")
                print(f"\tsnodas_w_lats          : ({len(lats)}) {lats.min():>+8.5g} <-> {lats.max():<+8.5g}")
                print(f"\tsnodas_w_lons          : ({len(lons)}) {lons.min():>+8.5g} <-> {lons.max():<+8.5g}")
                print(f"\tconusw_external_npts   : {conusw_external_npts:<20d}")
                print(f"\tconusw_masked_npts     : {conusw_masked_npts:<20d}")
                print(f"\tsnodas_w_npts          : {snodas_w_npts:<20d}")
                print(f"\tsnodas_w_npts          : {100 * (snodas_w_npts/ snodas_npts):<7.2f}%")
                print(f"\tsnodas_w_gids ({len(snodas_w_gids)} of {snodas_npts}): [{snodas_w_gids[0]}, ... {snodas_w_gids[-1]}]")

            # Area where sdfw masked by -9999
            sdfw_extra = sdfw[sdfw['band_1'] == -9999]
            snodas_w_external_npts = sdfw_extra.shape[0]
            snodas_w_external_gids = sdfw_extra['gids'].tolist()
            if verbose:
                print(f"\tsnodas_w_external_npts : {snodas_w_external_npts:< 20d}")
                print(f"\tsnodas_w_external_gids ({len(snodas_w_external_gids)} of {snodas_npts}): [{snodas_w_external_gids[0]}, ... {snodas_w_external_gids[-1]}]")
            ##
            # Save to pickle
            odat = (sdf_full, snodas_lats2d, snodas_lons2d, snodas_lats, snodas_lons, snodas_nlats,
                    snodas_nlons, snodas_npts, sdf_extra, snodas_external_npts, snodas_external_gids,
                    sdf_masked, conus_masked_npts, sdfw, conusw_npnts, sdfw_extra, conusw_external_npts,
                    sdfw_masked, snodas_w_masked_npts, snodas_w_masked_gids, snodas_w_sdf, snodas_w_npts,
                    snodas_w_gids, snodas_w_external_npts, snodas_w_external_gids)
            if verbose:
                print(f"\nSaved {basin_file}")
            with open(basin_file, 'wb') as f:
                pickle.dump(odat, f)
        else:
            if verbose:
                print(f"\nReading {basin_file}")
            with open(basin_file, 'rb') as f:
                odat = pickle.load(f)

            (sdf_full, snodas_lats2d, snodas_lons2d, snodas_lats, snodas_lons, snodas_nlats,
             snodas_nlons, snodas_npts, sdf_extra, snodas_external_npts, snodas_external_gids,
             sdf_masked, conus_masked_npts, sdfw, conusw_npnts, sdfw_extra, conusw_external_npts,
             sdfw_masked, snodas_w_masked_npts, snodas_w_masked_gids, snodas_w_sdf, snodas_w_npts,
             snodas_w_gids, snodas_w_external_npts, snodas_w_external_gids) = odat
        ##
        # Free memory
        del odat

        if make_grids:
            ##
            # Intersect Sierra Nevada Conservancy with SNODAS_W
            """
            sierra_nevada_npts   :               166321 2.81%
            """
            sierra_nevada_sids = sierra_nevada_roi['sids'].values[0]
            sdf_intersect = snodas_w_sdf.stare_intersects(sierra_nevada_sids, n_partitions=N_PARTS, num_workers=N_CORES)
            sierra_nevada_gdf = snodas_w_sdf.loc[sdf_intersect.values]
            sierra_nevada_npts = int(sierra_nevada_gdf.shape[0])
            sierra_nevada_gids = sierra_nevada_gdf['gids'].tolist()
            if verbose:
                print(f"\n\tsierra_nevada_npts   : {sierra_nevada_npts:20d} {100 * (sierra_nevada_npts / conusw_npnts):.2f}% of conusw_npnts")

            ##
            # Intersect Feather River with SNODAS_W
            """
            feather_river_npts   :                14326 0.24%
            """
            feather_river_sids = feather_river_roi['sids'].values[0]
            sdf_intersect = snodas_w_sdf.stare_intersects(feather_river_sids, n_partitions=N_PARTS, num_workers=N_CORES)
            feather_river_gdf = snodas_w_sdf.loc[sdf_intersect.values]
            feather_river_npts = int(feather_river_gdf.shape[0])
            feather_river_gids = feather_river_gdf['gids'].tolist()
            if verbose:
                print(f"\tfeather_river_npts   : {feather_river_npts:20d} {100 * (feather_river_npts / conusw_npnts):.2f}% of conusw_npnts")

            ##
            # Intersect Tuolumne River with SNODAS_W
            """
            tuolumne_river_npts  :                 6186 0.10%
            """
            tuolumne_river_sids = tuolumne_river_roi['sids'].values[0]
            sdf_intersect = snodas_w_sdf.stare_intersects(tuolumne_river_sids, n_partitions=N_PARTS, num_workers=N_CORES)
            tuolumne_river_gdf = snodas_w_sdf.loc[sdf_intersect.values]
            tuolumne_river_npts = int(tuolumne_river_gdf.shape[0])
            tuolumne_river_gids = tuolumne_river_gdf['gids'].tolist()
            if verbose:
                print(f"\ttuolumne_river_npts  : {tuolumne_river_npts:20d} {100 * (tuolumne_river_npts / conusw_npnts):.2f}%  of conusw_npnts")

            ##
            # Save to pickle
            odat = (sierra_nevada_gdf, sierra_nevada_gids, feather_river_gdf, feather_river_gids, tuolumne_river_gdf, tuolumne_river_gids)
            if verbose:
                print(f"\nSaved {grid_file}")
            with open(grid_file, 'wb') as f:
                pickle.dump(odat, f)
        else:
            if verbose:
                print(f"\nReading {grid_file}")
            with open(grid_file, 'rb') as f:
                odat = pickle.load(f)
            (sierra_nevada_gdf, sierra_nevada_gids, feather_river_gdf, feather_river_gids, tuolumne_river_gdf, tuolumne_river_gids) = odat
        del odat

        if make_plot:
            if verbose:
                print(f"\nMaking Plots ...")
            # ##
            # # Plot SNODAS_W
            # if mixed_qlev:
            #     pname = f"{ALT_DIR}snodas_w_masked_{QLEV_DGRID:02d}.png"
            # else:
            #     pname = f"{ALT_DIR}snodas_w_masked_{QLEV:02d}.png"
            # plot_conusw(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons, snodas_w_masked_gids)
            # if verbose:
            #     print(f"\nSaved {pname}")

            # # ##
            # # # Plot the Regions
            # # pname = f"{ALT_DIR}snodas_regions_{QLEV:02d}.png"
            # # plot_conus_regions(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons)
            # # if verbose:
            # #     print(f"Saved {pname}")

            # ##
            # # Plot the Sierra Nevada grids
            # if mixed_qlev:
            #     pname = f"{ALT_DIR}snodas_sierra_nevada_gids_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png"
            # else:
            #     pname = f"{ALT_DIR}snodas_sierra_nevada_gids_{QLEV:02d}.png"
            # plot_grids(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons, sierra_nevada_gids, sierra_nevada_roi, snodas_w_sdf,
            #            add_trixels=False, flip_trixels=False, trixel_double=False, zoomin=False)
            # if verbose:
            #     print(f"Saved {pname}")
            # if mixed_qlev:
            #     pname = f"{ALT_DIR}snodas_sierra_nevada_gids_zoomtrixel_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png" if use_trixels else f"{ALT_DIR}snodas_sierra_nevada_gids_zoom_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png"
            # else:
            #     pname = f"{ALT_DIR}snodas_sierra_nevada_gids_zoomtrixel_{QLEV:02d}.png" if use_trixels else f"{ALT_DIR}snodas_sierra_nevada_gids_zoom_{QLEV:02d}.png"
            # if trixel_flip:
            #    pname = pname.replace("zoomtrixel", "zoomfliptrixel")
            # elif double_trixel:
            #     pname = pname.replace("zoomtrixel", "zoom2trixel")
            # plot_grids(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons, sierra_nevada_gids, sierra_nevada_roi, snodas_w_sdf,
            #            add_trixels=use_trixels, flip_trixels=trixel_flip, trixel_double=double_trixel, zoomin=True)
            # if verbose:
            #     print(f"Saved {pname}")

            # ##
            # # Plot the Feather River grids
            # if mixed_qlev:
            #     pname = f"{ALT_DIR}snodas_feather_river_gids_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png"
            # else:
            #     pname = f"{ALT_DIR}snodas_feather_river_gids_{QLEV:02d}.png"
            # plot_grids(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons, feather_river_gids, feather_river_roi, snodas_w_sdf,
            #            add_trixels=False, flip_trixels=False, trixel_double=False, zoomin=False)
            # if verbose:
            #     print(f"Saved {pname}")
            if mixed_qlev:
                pname = f"{ALT_DIR}snodas_feather_river_gids_zoomtrixel_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png" if use_trixels else f"{ALT_DIR}snodas_feather_river_gids_zoom_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png"
            else:
                pname = f"{ALT_DIR}snodas_feather_river_gids_zoomtrixel_{QLEV:02d}.png" if use_trixels else f"{ALT_DIR}snodas_feather_river_gids_zoom_{QLEV:02d}.png"
            if trixel_flip:
               pname = pname.replace("zoomtrixel", "zoomfliptrixel")
            elif double_trixel:
                pname = pname.replace("zoomtrixel", "zoom2trixel")
            plot_grids(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons, feather_river_gids, feather_river_roi, snodas_w_sdf,
                      add_trixels=use_trixels, flip_trixels=trixel_flip, trixel_double=double_trixel, zoomin=True)
            if verbose:
                print(f"Saved {pname}")

            # # # Plot the Tuolumne River grids
            # if mixed_qlev:
            #     pname = f"{ALT_DIR}snodas_tuolumne_gids_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png"
            # else:
            #     pname = f"{ALT_DIR}snodas_tuolumne_gids_{QLEV:02d}.png"
            # # plot_grids(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons, tuolumne_river_gids, tuolumne_river_roi, snodas_w_sdf,
            #              add_trixels=False, flip_trixels=False, zoomin=False)
            # # if verbose:
            # #     print(f"Saved {pname}")
            # if mixed_qlev:
            #     pname = f"{ALT_DIR}snodas_tuolumne_gids_zoomtrixel_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png" if use_trixels else f"{ALT_DIR}snodas_tuolumne_gids_zoom_{QLEV_DGRID:02d}_{QLEV_BASIN:02d}.png"
            # else:
            #     pname = f"{ALT_DIR}snodas_tuolumne_gids_zoomtrixel_{QLEV:02d}.png" if use_trixels else f"{ALT_DIR}snodas_tuolumne_gids_zoom_{QLEV:02d}.png"
            # if trixel_flip:
            #    pname = pname.replace("zoomtrixel", "zoomfliptrixel")
            # elif double_trixel:
            #     pname = pname.replace("zoomtrixel", "zoom2trixel")
            # plot_grids(pname, snodas_lats2d, snodas_lons2d, snodas_npts, snodas_nlats, snodas_nlons, tuolumne_river_gids, tuolumne_river_roi, snodas_w_sdf,
            #            add_trixels=use_trixels, flip_trixels=trixel_flip, trixel_double=double_trixel, zoomin=True)
            # if verbose:
            #     print(f"Saved {pname}")
    return

#---Start of main code block.
if __name__=='__main__':

    # Send some informative output to standard I/O
    verbose     = [False, True][1]

    # Use QLEV_BASIN and QLEV_DGRID rather than unified QLEV encodings
    mixed_qlev = [False, True][0]

    # Use a narrow version of CONUS_W for faster graphics
    shrink_conusw = [False, True][1]

    # Make Basin DataFrames (rather than read)
    make_basins = [False, True][0]
    # Make Datagrid DataFrames (rather than read)
    make_dgrid  = [False, True][0]
    # Make Datagrid-Basin Intersection DataFrames (rather than read)
    make_grids  = [False, True][0]
    if shrink_conusw and make_grids:
        raise Exception("Warning can't use shrink_conusw and make_grids together!")

    # Make some plots
    make_plot   = [False, True][1]
    # Add basin trixels to some plots
    use_trixels = [False, True][1]
    # Do a reverse and show datagrid trixels and basin polygon (use_trixels must be True)
    trixel_flip = [False, True][0]
    # Show basin and datagrid trixels rather than datagrid grids
    double_trixel = [False, True][0]
    if double_trixel and trixel_flip:
        raise Exception("Warning can't use double_trixel and trixel_flip together!")
    if trixel_flip or double_trixel:
        use_trixels =  True

    ##
    # Main Routine
    demo_get_basin(mixed_qlev, shrink_conusw, make_basins, make_dgrid, make_grids, make_plot, use_trixels, trixel_flip, double_trixel, verbose)

# >>>> ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: <<<<
# >>>> END OF FILE | END OF FILE | END OF FILE | END OF FILE | END OF FILE | END OF FILE <<<<
# >>>> ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: <<<<
