"""
Generate a report on night-time light change vs GDP for African countries (2014–2023).
Exports VIIRS images to Google Drive and produces local CSV, PNG and HTML outputs.
"""
import ee
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from scipy.stats import pearsonr

# ─── Configuration ────────────────────────────────────────────────────────────
LOGGING_LEVEL    = logging.INFO
OUTPUT_DIR       = Path(".")
EXPORT_SCALE     = 5000  # m
EXPORT_REGION    = ee.Geometry.Rectangle([-20, -35, 55, 40])
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
GAUL_COLLECTION  = "FAO/GAUL/2015/level0"

AFRICAN_COUNTRIES = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde",
    "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo", "Côte d'Ivoire",
    "Democratic Republic of the Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
    "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya",
    "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
    "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe",
    "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
    "United Republic of Tanzania", "Togo", "Tunisia", "Uganda", "Western Sahara", "Zambia", "Zimbabwe"
]

# ─── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("night_light_project.log"),
        logging.StreamHandler()
    ]
)

def init_earth_engine():
    """Authenticate and initialize the Earth Engine API."""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()
    print("Initialized Earth Engine.")

def export_image(image: ee.Image, name: str, scale: int = EXPORT_SCALE):
    """Export an EE image to Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder="EarthEngine",
        fileNamePrefix=name,
        region=EXPORT_REGION,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13
    )
    task.start()
    print(f"Started export task: {name}")

def fetch_viirs_stats() -> pd.DataFrame:
    """Compute mean radiance (2014, 2023) and percent change per country."""
    fc = ee.FeatureCollection(GAUL_COLLECTION) \
           .filter(ee.Filter.inList("ADM0_NAME", AFRICAN_COUNTRIES))

    viirs = ee.ImageCollection(VIIRS_COLLECTION) \
        .filterDate("2014-01-01", "2023-12-31") \
        .select(["avg_rad", "cf_cvg"]) \
        .map(lambda img: img.updateMask(img.select("cf_cvg").gte(2))) \
        .map(lambda img: img.updateMask(img.select("avg_rad").gte(0)))

    mean2014 = viirs.filterDate("2014-01-01", "2014-12-31") \
                    .mean().select("avg_rad")
    mean2023 = viirs.filterDate("2023-01-01", "2023-12-31") \
                    .mean().select("avg_rad")

    # schedule exports
    export_image(mean2014, "Night_Light_2014")
    export_image(mean2023, "Night_Light_2023")
    export_image(mean2023.subtract(mean2014).rename("Change"), "Night_Light_Change")

    # reduceRegions to get stats
    def reducer(image, name):
        feats = image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=500
        ).select(["ADM0_NAME", "mean"], ["ADM0_NAME", name])
        return feats.getInfo()["features"]

    feats14 = reducer(mean2014, "R_2014")
    feats23 = reducer(mean2023, "R_2023")

    # build DataFrame
    df14 = pd.DataFrame([{"ADM0_NAME": f["properties"]["ADM0_NAME"],
                          "R_2014": f["properties"]["R_2014"]}
                         for f in feats14])
    df23 = pd.DataFrame([{"ADM0_NAME": f["properties"]["ADM0_NAME"],
                          "R_2023": f["properties"]["R_2023"]}
                         for f in feats23])
    df = pd.merge(df14, df23, on="ADM0_NAME")
    df["Light_Change"] = (df.R_2023 - df.R_2014) / df.R_2014 * 100
    print("Fetched VIIRS stats.")
    missing_countries = [c for c in AFRICAN_COUNTRIES if c not in df['ADM0_NAME'].values]
    if missing_countries:
        logging.warning(f"Missing countries in df: {missing_countries}")

     # standardize names to match other datasets
    df.ADM0_NAME.replace({
        "Congo": "Congo, Rep.",
        "Democratic Republic of the Congo": "Congo, Dem. Rep.",
        "Egypt": "Egypt, Arab Rep.",
        "Gambia": "Gambia, The"
    }, inplace=True)
    return df

def fetch_population_data() -> pd.DataFrame:
    """Download and pivot World Bank population data for 2014–2023."""
    url = ("https://api.worldbank.org/v2/country/all/indicator/"
           "SP.POP.TOTL?format=json&date=2014:2023&per_page=10000")
    resp = requests.get(url)
    data = resp.json()[1]
    df = pd.DataFrame(data)
    df["country"] = df.country.apply(lambda c: c["value"])
    df["value"] = df.value.astype(float)
    pivot = df.pivot(index="country", columns="date", values="value").reset_index()
    pivot.rename(columns={"country": "ADM0_NAME"}, inplace=True)

    # # standardize names to match other datasets
    # pivot.ADM0_NAME.replace({
    #     "Congo": "Congo, Rep.",
    #     "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    #     "Egypt": "Egypt, Arab Rep.",
    #     "Gambia": "Gambia, The"
    # }, inplace=True)
    
    # Add population change calculation
    pivot["Pop_Change"] = ((pivot["2023"] - pivot["2014"]) / pivot["2014"]) * 100
    
    # Add columns with clearer names
    pivot.rename(columns={
        "2014": "Pop_2014",
        "2023": "Pop_2023"
    }, inplace=True)
    
    print("Fetched and processed population data.")
    return pivot

def fetch_gdp_data() -> pd.DataFrame:
    """Download and pivot World Bank GDP (constant 2021 USD) for 2014–2023."""
    url = ("https://api.worldbank.org/v2/country/all/indicator/"
           "NY.GDP.MKTP.KD?format=json&date=2014:2023&per_page=10000")
    resp = requests.get(url)
    data = resp.json()[1]
    df = pd.DataFrame(data)
    print(df.head())
    df["country"]   = df.country.apply(lambda c: c["value"])
    df["value"]     = df.value.astype(float)
    pivot = df.pivot(index="country", columns="date", values="value").reset_index()
    pivot.rename(columns={"country": "ADM0_NAME"}, inplace=True)
 
    pivot["GDP_Change"] = ((pivot["2023"] - pivot["2014"]) / pivot["2014"]) * 100
    print("Fetched and processed GDP data.")
    return pivot

def merge_data(df_light: pd.DataFrame, df_gdp: pd.DataFrame, df_pop: pd.DataFrame) -> pd.DataFrame:
    """Combine VIIRS, GDP, and Population into a single table."""
    df = pd.merge(df_light, df_gdp, on="ADM0_NAME", how="left")
    df = pd.merge(df, df_pop, on="ADM0_NAME", how="left")
    out_csv = OUTPUT_DIR / "night_light_gdp_pop.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved merged CSV: {out_csv}")
    return df

def plot_scatter(df: pd.DataFrame, year: int = 2014):
    """Scatter plot of Radiance vs GDP (log–log) with correlations."""
    radiance_col = f"R_{year}"
    gdp_col = str(year)
    x = np.log10(df[radiance_col] + 1e-6)
    y = np.log10(df[gdp_col] + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=30)
    for _, r in df.iterrows():
        ax.text(np.log10(r[radiance_col]), np.log10(r[gdp_col]), r.ADM0_NAME, fontsize=6)
    ax.set_xlabel(f"log10 Radiance {year}")
    ax.set_ylabel(f"log10 GDP {year}")
    ax.set_title(f"Night Light vs GDP ({year})")
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xlim(-1.2, 0.5)
    r, p = pearsonr(x, y)

    rho, ps = spearmanr(x, y)
    print(f"Pearson r={r:.2f}, p={p:.2e}")
    print(f"Spearman ρ={rho:.2f}, p={ps:.2e}")

    out_png = OUTPUT_DIR / f"scatter_light_vs_gdp_{year}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved scatter plot: {out_png}")

def plot_changes_scatter(df: pd.DataFrame):
    """Scatter plot comparing Light Change vs GDP Change with correlations and regression line."""
    # Remove rows with NaN values
    df_clean = df.dropna(subset=['Light_Change', 'GDP_Change'])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points at once with same color (like plot_scatter)
    ax.scatter(df_clean['Light_Change'], df_clean['GDP_Change'], s=30, c='blue', edgecolors='w', alpha=0.7)

    # Add text labels separately
    for _, row in df_clean.iterrows():
        ax.text(row['Light_Change'], row['GDP_Change'], row['ADM0_NAME'], fontsize=8)

    # Linear regression
    x = df_clean['Light_Change']
    y = df_clean['GDP_Change']
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    x_vals = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_vals, trend(x_vals), color='red', lw=2, label=f"Trend: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")

    # Set axis labels and title
    ax.set_xlabel('Night Light Change (%)', fontsize=12)
    ax.set_ylabel('GDP Change (%)', fontsize=12)
    ax.set_title('Night Light Change vs GDP Change (2014-2023)', fontsize=14)

    # Show grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Calculate and log correlations
    r, p = pearsonr(x, y)
    rho, ps = spearmanr(x, y)
    print(f"Changes correlation - Pearson r={r:.2f}, p={p:.2e}")
    print(f"Changes correlation - Spearman ρ={rho:.2f}, p={ps:.2e}")

    # Save the plot
    out_png = OUTPUT_DIR / "scatter_light_vs_gdp_changes.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved changes scatter plot: {out_png}")
    plt.show()
def plot_scatter_trajectory(df: pd.DataFrame, prefix: str = ""):
    """Scatter plot showing 2014 and 2023 data points connected by lines for each country."""
    # Remove rows with NaN values for both years
    col_2014= f"{prefix}2014"
    col_2023= f"{prefix}2023"
    df_clean = df.dropna(subset=['R_2014', 'R_2023', col_2014, col_2023]).reset_index(drop=True)

    # Calculate log values
    x_2014 = np.log10(df_clean['R_2014'] + 1e-6)
    y_2014 = np.log10(df_clean[col_2014] + 1e-6)
    x_2023 = np.log10(df_clean['R_2023'] + 1e-6)
    y_2023 = np.log10(df_clean[col_2023] + 1e-6)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot lines connecting 2014 to 2023 for each country
    for i in range(len(df_clean)):
        x_vals = [x_2014.iloc[i], x_2023.iloc[i]]
        y_vals = [y_2014.iloc[i], y_2023.iloc[i]]
        ax.plot(x_vals, y_vals, 'k-', alpha=0.3, linewidth=1)
    
    # Plot 2014 data points
    ax.scatter(x_2014, y_2014, s=30, c='lightblue', alpha=0.7, label='2014', edgecolors='k')
    
    # Plot 2023 data points         
    ax.scatter(x_2023, y_2023, s=30,c='lightcoral', alpha=0.7, label='2023', edgecolors='k')

    # Add country labels (using 2023 positions to avoid overlap)
    for i in range(len(df_clean)):
        shift = 0.01  # Shift labels slightly to avoid overlap with points
        ax.text(x_2023.iloc[i]+ shift, y_2023.iloc[i]+ shift, 
                df_clean['ADM0_NAME'].iloc[i], fontsize=6, alpha=0.8)
    
    # Set axis labels and title
   
    if prefix:
        ax.set_title('Night Light vs Population Trajectory', fontsize=14)
        ax.set_xlabel('log10 Night Light Radiance', fontsize=12)
        ax.set_ylabel('log10 Population', fontsize=12)
    else:
        ax.set_title('Night Light vs GDP Trajectory', fontsize=14)
        ax.set_xlabel('log10 Night Light Radiance', fontsize=12)
        ax.set_ylabel('log10 GDP (constant 2021 USD)', fontsize=12)

    # Show grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_xlim(-1.2, 0.5)
    
    # Calculate correlations for both years
    r_2014, p_2014 = pearsonr(x_2014, y_2014)
    r_2023, p_2023 = pearsonr(x_2023, y_2023)
    print(f"2014 correlation - Pearson r={r_2014:.2f}, p={p_2014:.2e}")
    print(f"2023 correlation - Pearson r={r_2023:.2f}, p={p_2023:.2e}")
    
    # Save the plot
    out_png = OUTPUT_DIR / "scatter_light_vs_gdp_trajectory.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved trajectory scatter plot: {out_png}")
    plt.show()
def plot_correlation_matrix(df: pd.DataFrame):
    """Generate and save correlation matrix for change variables."""
    # Compute correlation matrix for only change columns
    corr_cols = ["Light_Change", "GDP_Change", "Pop_Change"]
    corr_df = df[corr_cols].corr(method="pearson")
    
    # Save correlation matrix as CSV
    out_csv = OUTPUT_DIR / "correlation_matrix.csv"
    corr_df.to_csv(out_csv)
    print(f"Saved correlation matrix: {out_csv}")
    print("Correlation matrix:\n", corr_df)

    # Plot the correlation matrix with values in cells
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson correlation')
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation Matrix")
    
    # Annotate values
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    
    plt.tight_layout()
    out_png = OUTPUT_DIR / "correlation_matrix.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved correlation matrix plot: {out_png}")
    plt.show()

def main():
    print("Starting night light analysis...")
    init_earth_engine()
    df_light = fetch_viirs_stats()
    
    df_pop   = fetch_population_data()
    df_gdp   = fetch_gdp_data()
    df_all   = merge_data(df_light, df_gdp, df_pop)
    
    plot_scatter(df_all, year=2023)
    plot_scatter(df_all, year=2014)
    plot_changes_scatter(df_all)
    plot_scatter_trajectory(df_all, prefix="")
    plot_scatter_trajectory(df_all, prefix="Pop_")
    plot_correlation_matrix(df_all)

if __name__ == "__main__":
    main()

