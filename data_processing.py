import numpy as np
import os
import re
from datetime import datetime
from netCDF4 import Dataset
from pyresample import geometry, kd_tree
import dask.array as da
from pyresample.bucket import BucketResampler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Function to process files in a directory and organize them by date
def process_files(data_directory):
    """
    Processes netCDF files in a directory and organizes them by date.

    Parameters:
    - data_directory: str, directory containing netCDF files

    Returns:
    - daily_datasets: dict, datasets organized by date
    """
    daily_datasets = {}

    # Loop through each netCDF file in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith((".nc4", ".nc")):  # Check for both extensions
            # Extract the date from the filename using regular expression
            match = re.search(r'(\d{8})T', filename)
            if match:
                date_str = match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d').date()
                
                file_path = os.path.join(data_directory, filename)

                # Initialize the list for the date if it doesn't exist
                if date not in daily_datasets:
                    daily_datasets[date] = []

                daily_datasets[date].append(file_path)
            else:
                print(f"Filename {filename} does not contain a valid date.")

    return daily_datasets

# Function to filter and clean data based on quality and placeholder values
def filter_and_clean_data(combined_data, placeholder_value, variable_name, qa_threshold, min_value=None):
    """
    Filters and cleans the data based on quality and placeholder values.

    Parameters:
    - combined_data: dict, combined data from different dates
    - placeholder_value: value to be replaced with NaN
    - variable_name: name of the variable to be filtered
    - qa_threshold: threshold for quality assurance
    - min_value: minimum value to filter the data (optional)

    Returns:
    - filtered_data: dict, filtered data
    """
    filtered_data = {}

    for date, data in combined_data.items():
        variable_data = data[variable_name]
        qa_values = data["qa_value"]

        # Replace placeholder values with NaN
        variable_data = np.where(variable_data == placeholder_value, np.nan, variable_data)
        variable_data = np.where(qa_values <= qa_threshold, np.nan, variable_data)
        
        # Apply the minimum value filter only if min_value is provided
        if min_value is not None:
            variable_data = np.where(variable_data <= min_value, np.nan, variable_data)

        filtered_data[date] = {
            "longitude": data["longitude"],
            "latitude": data["latitude"],
            variable_name: variable_data
        }

    return filtered_data

import numpy as np
from netCDF4 import Dataset

import numpy as np
from netCDF4 import Dataset

def combine_data(daily_datasets, variable_name, qa_variable_name='qa_value', conversion_factor=None):
    """
    Combines data from different files and dates.

    Parameters:
    - daily_datasets: dict, datasets organized by date
    - variable_name: str, name of the variable to combine
    - qa_variable_name: str, name of the quality assurance variable (default is 'qa_value')
    - conversion_factor: float, optional conversion factor to apply to the variable values

    Returns:
    - combined_data: dict, combined data from different dates
    """
    combined_data = {}

    for date, file_paths in daily_datasets.items():
        longitudes = []
        latitudes = []
        variable_values = []
        qa_values = []

        for file_name in file_paths:
            try:
                nc_file = Dataset(file_name, 'r')
                try:
                    # Check if 'PRODUCT' exists
                    if 'PRODUCT' in nc_file.groups:
                        group = nc_file.groups['PRODUCT']
                    else:
                        print(f"'PRODUCT' group not found in {file_name}")
                        continue

                    # Extract necessary variables
                    longitude = group.variables["longitude"][0, :, :]
                    latitude = group.variables["latitude"][0, :, :]
                    variable_data = group.variables[variable_name][0, :, :]
                    qa_value = group.variables[qa_variable_name][0, :, :]

                    # Apply the conversion factor if provided
                    if conversion_factor is not None:
                        variable_data = variable_data * conversion_factor

                    longitudes.append(longitude)
                    latitudes.append(latitude)
                    variable_values.append(variable_data)
                    qa_values.append(qa_value)
                except KeyError as e:
                    print(f"Variable {e} not found in {file_name}")
                finally:
                    nc_file.close()
            except Exception as e:
                print(f"An error occurred while processing {file_name}: {e}")

        if longitudes and latitudes and variable_values and qa_values:
            # Determine the common shape for concatenation
            common_shape = longitudes[0].shape
            longitudes = [arr for arr in longitudes if arr.shape == common_shape]
            latitudes = [arr for arr in latitudes if arr.shape == common_shape]
            variable_values = [arr for arr in variable_values if arr.shape == common_shape]
            qa_values = [arr for arr in qa_values if arr.shape == common_shape]

            # Check for dimension mismatches and skip mismatched arrays
            if len(longitudes) == len(latitudes) == len(variable_values) == len(qa_values):
                try:
                    combined_data[date] = {
                        "longitude": np.concatenate(longitudes, axis=0),
                        "latitude": np.concatenate(latitudes, axis=0),
                        variable_name: np.concatenate(variable_values, axis=0),
                        qa_variable_name: np.concatenate(qa_values, axis=0)
                    }
                    print(f"Data combination successful for date {date}")
                except ValueError as e:
                    print(f"Error concatenating arrays for date {date}: {e}")
            else:
                print(f"Skipping date {date} due to mismatched array shapes.")
        else:
            print(f"No valid data to combine for date {date}.")

    return combined_data

# Function to regrid and reproject data
def regrid_and_reproject_data(lats, lons, data, area_def, method='nearest'):
    """
    Regrids and reprojects the data using different methods.

    Parameters:
    - lats: array, latitude values
    - lons: array, longitude values
    - data: array, data to be regridded and reprojected
    - area_def: AreaDefinition, area definition for the target grid
    - method: str, resampling method ('nearest', 'average', 'count', 'sum', 'min', 'max', 'median', 'abs_max')

    Returns:
    - reprojected_data: array, reprojected data
    - lons_new: array, new longitude values
    - lats_new: array, new latitude values
    """
    swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
    
    if method == 'nearest':
        reprojected_data = kd_tree.resample_nearest(
            swath_def, data, area_def, radius_of_influence=50000, epsilon=0.5, fill_value=None)
    elif method in ['average', 'count', 'sum', 'min', 'max', 'median', 'abs_max']:
        lons = da.from_array(lons, chunks=(lons.shape[0]//10, lons.shape[1]//10))  # Convert to Dask array
        lats = da.from_array(lats, chunks=(lats.shape[0]//10, lats.shape[1]//10))  # Convert to Dask array
        data = da.from_array(data, chunks=(data.shape[0]//10, data.shape[1]//10))  # Convert data to Dask array
        resampler = BucketResampler(area_def, lons, lats)
        
        if method == 'average':
            reprojected_data = resampler.get_average(data, skipna=True)
        elif method == 'count':
            reprojected_data = resampler.get_count(data, skipna=True)
        elif method == 'sum':
            reprojected_data = resampler.get_sum(data, skipna=True)
        elif method == 'min':
            reprojected_data = resampler.get_min(data, skipna=True)
        elif method == 'max':
            reprojected_data = resampler.get_max(data, skipna=True)
        elif method == 'median':
            reprojected_data = resampler.get_median(data, skipna=True)
        elif method == 'abs_max':
            reprojected_data = resampler.get_abs_max(data, skipna=True)
    else:
        raise ValueError("Unsupported resampling method")
    
    lons_new, lats_new = area_def.get_lonlats()
    return reprojected_data, lons_new, lats_new

# Function to plot reprojected data
def plot_reprojected_data(reprojected_datasets, area_def, plot_date=None, cmap='RdYlBu_r', vmin=0, vmax=30):
    """
    Plots the reprojected data for a specific date.

    Parameters:
    - reprojected_datasets: dict, reprojected datasets
    - area_def: AreaDefinition, area definition for the target grid
    - plot_date: str, date to plot (format: YYYY-MM-DD)
    - cmap: str, colormap for the plot
    - vmin: float, minimum value for colormap
    - vmax: float, maximum value for colormap
    """
    if plot_date:
        # Convert the plot_date string to a datetime.date object
        try:
            plot_date = datetime.strptime(plot_date, '%Y-%m-%d').date()
        except ValueError:
            print(f"Invalid date format: {plot_date}. Please use YYYY-MM-DD format.")
            return
        
        # If a specific date is provided, plot data for that date
        if plot_date in reprojected_datasets:
            data = reprojected_datasets[plot_date]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.NorthPolarStereo()})
            
            ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
            
            mesh = ax.pcolormesh(data["lons_new"], data["lats_new"], data["reprojected_data"], cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            plt.colorbar(mesh, ax=ax, orientation='vertical', label='Reprojected Data Values')
            plt.title(f'Reprojected Data on {plot_date}')
            plt.show()
        else:
            print(f"No data available for the specified date: {plot_date}")
    else:
        # If no date is provided, inform the user
        print("No date specified for plotting. Please provide a date to plot the data.")


