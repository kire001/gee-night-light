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

# ─── Configuration ────────────────────────────────────────────────────────────
LOGGING_LEVEL    = logging.INFO
OUTPUT_DIR       = Path(".")
EXPORT_SCALE     = 5000  # m
EXPORT_REGION    = ee.Geometry.Rectangle([-20, -35, 55, 40])
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
GAUL_COLLECTION  = "FAO/GAUL/2015/level0"
AFRICAN_COUNTRIES = [
    "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cabo Verde",
    "Cameroon","Central African Republic","Chad","Comoros","Congo",
    "Democratic Republic of the Congo","Djibouti","Egypt","Equatorial Guinea",
    "Eritrea","Eswatini","Ethiopia","Gabon","Gambia","Ghana","Guinea",
    "Guinea-Bissau","Ivory Coast","Kenya","Lesotho","Liberia","Libya",
    "Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco",
    "Mozambique","Namibia","Niger","Nigeria","Rwanda","Sao Tome and Principe",
    "Senegal","Seychelles","Sierra Leone","Somalia","South Africa","South Sudan",
    "Sudan","Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe"
]

# ─── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=LOGGING_LEVEL,
                    format="%(asctime)s %(levelname)s: %(message)s")

def init_earth_engine():
    """Authenticate and initialize the Earth Engine API."""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()
    logging.info("Initialized Earth Engine.")

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
    logging.info(f"Started export task: {name}")

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
    logging.info("Fetched VIIRS stats.")
    return df

def fetch_gdp_data() -> pd.DataFrame:
    """Download and pivot World Bank GDP (constant 2021 USD) for 2014–2023."""
    url = ("https://api.worldbank.org/v2/country/all/indicator/"
           "NY.GDP.MKTP.KD?format=json&date=2014:2023&per_page=10000")
    resp = requests.get(url)
    data = resp.json()[1]
    df = pd.DataFrame(data)
    df["country"]   = df.country.apply(lambda c: c["value"])
    df["value"]     = df.value.astype(float)
    pivot = df.pivot(index="country", columns="date", values="value").reset_index()
    pivot.rename(columns={"country": "ADM0_NAME"}, inplace=True)

    # standardize names
    pivot.ADM0_NAME.replace({
        "Congo": "Congo, Rep.",
        "Democratic Republic of the Congo": "Congo, Dem. Rep.",
        "Egypt": "Egypt, Arab Rep.",
        "Gambia": "Gambia, The"
    }, inplace=True)
    pivot["GDP_Change"] = ((pivot["2023"] - pivot["2014"]) / pivot["2014"]) * 100
    logging.info("Fetched and processed GDP data.")
    return pivot

def merge_data(df_light: pd.DataFrame, df_gdp: pd.DataFrame) -> pd.DataFrame:
    """Combine VIIRS and GDP into a single table."""
    df = pd.merge(df_light, df_gdp, on="ADM0_NAME", how="left")
    out_csv = OUTPUT_DIR / "night_light_gdp.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved merged CSV: {out_csv}")
    return df

def plot_scatter(df: pd.DataFrame):
    """Scatter plot of Radiance vs GDP (log–log) with correlations."""
    x = np.log10(df.R_2014 + 1e-6)
    y = np.log10(df["2014"] + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=30, color="steelblue")
    for _, r in df.iterrows():
        ax.text(np.log10(r.R_2014), np.log10(r["2014"]), r.ADM0_NAME, fontsize=6)
    ax.set_xlabel("log10 Radiance 2014")
    ax.set_ylabel("log10 GDP 2014")
    ax.set_title("Night Light vs Real GDP (2014)")
    ax.grid(True, ls="--", alpha=0.5)

    r, p = pearsonr(x, y)
    rho, ps = spearmanr(x, y)
    logging.info(f"Pearson r={r:.2f}, p={p:.2e}")
    logging.info(f"Spearman ρ={rho:.2f}, p={ps:.2e}")

    out_png = OUTPUT_DIR / "scatter_light_vs_gdp.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    logging.info(f"Saved scatter plot: {out_png}")

def generate_html_report(df: pd.DataFrame):
    """Generate a simple HTML report with summary table and figure."""
    table_html = df[["ADM0_NAME", "R_2014", "R_2023", "Light_Change", "GDP_Change"]] \
                 .to_html(index=False, float_format="%.1f")
    html = f"""<html>
<head><title>Night Light vs GDP Report</title></head>
<body>
  <h1>Night Light Change & GDP (2014–2023)</h1>
  <h2>Scatter Plot</h2>
  <img src="scatter_light_vs_gdp.png" alt="Scatter" width="600"/>
  <h2>Summary Table</h2>
  {table_html}
</body>
</html>"""
    out_file = OUTPUT_DIR / "night_light_gdp_report.html"
    out_file.write_text(html, encoding="utf-8")
    logging.info(f"Generated HTML report: {out_file}")

def main():
    init_earth_engine()
    df_light = fetch_viirs_stats()
    df_gdp   = fetch_gdp_data()
    df_all   = merge_data(df_light, df_gdp)
    plot_scatter(df_all)
    generate_html_report(df_all)

if __name__ == "__main__":
    main()