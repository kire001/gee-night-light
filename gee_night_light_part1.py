import ee
import matplotlib.pyplot as plt
import pandas as pd
import requests

# List of African countries
african_countries = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 
    'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 
    'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 
    'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 
    'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 
    'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 
    'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 
    'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 
    'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
]

def export_image_to_drive(image, name, scale=10000):
    """
    Exports an Earth Engine image to Google Drive as a GeoTIFF file.

    Args:
        image (ee.Image): The Earth Engine image to export.
        name (str): The prefix for the exported file name.
        scale (int, optional): The resolution in meters. Defaults to 10000.
    """
    africa_envelope = ee.Geometry.Rectangle([-20, -35, 55, 40]) 
    export_task = ee.batch.Export.image.toDrive(
        image=image,
        description=f'{name}_export',
        folder='EarthEngine',
        fileNamePrefix=name,
        region=africa_envelope,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    export_task.start()
    print(f'Started task: {export_task.config["description"]}')

def main():
    # Authenticate and initialize Earth Engine
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # ----------------------------- Get data from GEE ---------------------------- #
    countries_africa = ee.FeatureCollection('FAO/GAUL/2015/level0') \
        .filter(ee.Filter.inList('ADM0_NAME', african_countries))

    # Load VIIRS DNB data (2014–2023), with number of cloud-free observations higher than 3
    # radiance in nanoWatts/sr/cm^2
    viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
        .filterDate('2014-01-01', '2023-12-31') \
        .select(['avg_rad', 'cf_cvg']) \
        .map(lambda image: image.updateMask(image.select('cf_cvg').gte(2))) \
        .map(lambda image: image.updateMask(image.select('avg_rad').gte(0)))

    # Compute annual mean radiance for 2014 and 2023
    viirs_2014 = viirs.filterDate('2014-01-01', '2014-12-31').mean().select('avg_rad')
    viirs_2023 = viirs.filterDate('2023-01-01', '2023-12-31').mean().select('avg_rad')
    print('getting data from GEE')

    # Calculate mean radiance per country
    radiance_2014 = viirs_2014.reduceRegions(
        collection=countries_africa,
        reducer=ee.Reducer.mean(),
        scale=500
    ).select(['ADM0_NAME', 'mean'], ['ADM0_NAME', 'R_2014'])

    radiance_2023 = viirs_2023.reduceRegions(
        collection=countries_africa,
        reducer=ee.Reducer.mean(),
        scale=500
    ).select(['ADM0_NAME', 'mean'], ['ADM0_NAME', 'R_2023'])

    # Get results to a DataFrame
    list_2014 = radiance_2014.getInfo()['features']
    list_2023 = radiance_2023.getInfo()['features']

    # pixel value of change
    change = viirs_2023.subtract(viirs_2014).rename('Change')
    change = change.unmask(0)
    viirs_change = change.select('Change')
    print('getting data from GEE before export')

    # ----------------------- Process data from GEE locally ---------------------- #

    # Merge the two DataFrames on the country name (ADM0_NAME)
    df_results = pd.merge(
        pd.DataFrame([{
            'ADM0_NAME': feature['properties']['ADM0_NAME'],
            'R_2014': feature['properties']['R_2014']
        } for feature in list_2014]),
        pd.DataFrame([{
            'ADM0_NAME': feature['properties']['ADM0_NAME'],
            'R_2023': feature['properties']['R_2023']
        } for feature in list_2023]),
        on='ADM0_NAME'
    )

    # Calculate the percentage change
    df_results['Light_Change'] = ((df_results['R_2023'] - df_results['R_2014']) / df_results['R_2014']) * 100

    # --------------------------------- GDP data --------------------------------- #

    # Download World Bank GDP per capita (PPP, constant 2021 Int-$) data
    url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.KD?format=json&date=2014:2023&per_page=10000"

    response = requests.get(url)
    # Convert JSON response to DataFrame
    data = response.json()[1]  # The second element contains the data
    # Create a DataFrame from the JSON data
    df_gdp_all = pd.DataFrame(data)

    # Replace the 'indicator' column (which is a dictionary) with only the 'value' key
    df_gdp_all['indicator'] = df_gdp_all['indicator'].apply(lambda x: x['value'] if isinstance(x, dict) else None)
    df_gdp_all['country'] = df_gdp_all['country'].apply(lambda x: x['value'] if isinstance(x, dict) else None)

    # Pivot the GDP data so each country is on one row and all years are in columns
    df_gdp_all_pivot = df_gdp_all.pivot(index='country', columns='date', values='value').reset_index()
    # Rename columns for clarity
    df_gdp_all_pivot.rename(columns={'country': 'Country'}, inplace=True)

    # Standardize country names for merging
    df_results['ADM0_NAME'] = df_results['ADM0_NAME'].replace({
        'Congo': 'Congo, Rep.',
        'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
        'Egypt': 'Egypt, Arab Rep.',
        'Gambia': 'Gambia, The'
    })
    df_results = df_results.merge(df_gdp_all_pivot, left_on='ADM0_NAME', right_on='Country', how='left')
    df_results['GDP_change'] = (df_results['2023'] - df_results['2014']) / df_results['2014'] * 100
    df_results.to_csv('gdp_sum.csv', index=False)

    # --------------------------- Visualize the results -------------------------- #

    # Scatter plot: R_2014 vs 2014 with log scale
    fig, ax = plt.subplots(figsize=(10, 8))

    data_light = 'Light_Change'
    data_gdp = 'GDP_change'
    # Plot each country as a point
    for i, row in df_results.iterrows():
        ax.scatter(row[data_light], row[data_gdp], label=row['ADM0_NAME'])
        ax.text(row[data_light], row[data_gdp], row['ADM0_NAME'], fontsize=8)

    # Set axis labels and title
    ax.set_xlabel('Night Light Radiance change (2014-2023) [%]', fontsize=12)
    ax.set_ylabel('Real GDP change(2014 -2023) [%]', fontsize=12)
    ax.set_title('Change of Night Light Radiance and real GDP(2014)', fontsize=14)

    # Show grid and plot
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # save scatter for report
    plt.savefig("scatter_light_vs_gdp.png", dpi=300)
    logging.info("Scatter plot saved: scatter_light_vs_gdp.png")
    plt.show()    

    
    # mean values on map
    # # Load country shapefile
    
    africa_gdf = countries_gdf[countries_gdf["CONTINENT"] == "Africa"]
    
    # Merge with results
    gdf = africa_gdf.merge(df, left_on="ADMIN", right_on="ADM0_NAME")

    # Plot map
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="Change_", ax=ax, cmap="RdBu", vmin=-50, vmax=50, legend=True,
             legend_kwds={"label": "Relative Change (%)", "orientation": "horizontal"})
    plt.title("Night Light Change in Africa (2014–2023)")
    plt.axis('off')
    plt.savefig("night_light_change_africa.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
