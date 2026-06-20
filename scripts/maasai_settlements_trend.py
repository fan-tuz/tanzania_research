"""
Trend multi-anno (2009-2022) di insediamenti nelle aree masai del nord Tanzania
=================================================================================

PERCHE' NON OVERPASS TURBO
---------------------------
Overpass (Turbo) puo' interrogare il passato solo tramite la sua "attic data"
(parametro [date:"..."]), ma con due limiti pesanti per questo caso d'uso:

1. L'attic data della maggior parte delle istanze pubbliche (incluso il server
   principale overpass-api.de) parte dal primo dump compatibile ODbL, cioe'
   2012-09-12T06:55:00Z. Qualsiasi data precedente (es. 2009, 2010, 2011, 2012
   fino a settembre) restituisce semplicemente lo stato del 2012-09-12: NON e'
   un dato storico reale per quegli anni.
2. Query multi-anno su un'area ampia con [date:...] sono lente, spesso vanno in
   timeout sul servizio pubblico condiviso, e le query "area" derivate da
   relation restituiscono sempre la geometria ATTUALE, non quella storica.

ohsome (sviluppato da HeiGIT, Universita' di Heidelberg) e' pensato apposta per
questo: usa l'intera cronologia OSM, che parte dall'8 ottobre 2007, quindi
copre senza problemi tutto il periodo 2009-2022 richiesto, con un endpoint
nativo per serie temporali e un client Python (ohsome-py) gia' pronto.

ATTENZIONE SUI DATI REALI (importante, leggere prima di interpretare i risultati)
----------------------------------------------------------------------------------
I "boma" masai - gli insediamenti pastorali informali, spesso temporanei, che
in pratica corrispondono a cio' che si intende per "villaggio masai" - NON sono
sistematicamente mappati in OSM. La letteratura scientifica li descrive
esplicitamente come "non documentati nei sistemi GIS esistenti" (e' per questo
che esistono progetti di mappatura via deep learning su immagini satellitari,
es. Anderson et al. sui boma masai in Tanzania). Quindi:

  - Questo script estrae il MIGLIOR PROXY disponibile in OSM: villaggi/hamlet
    amministrativi (place=village/hamlet) ed edifici/aree residenziali mappati.
  - NON e' un censimento dei boma reali. La crescita osservata nel tempo
    rifletterà tanto l'attivita' di mappatura (es. campagne HOT/Ramani Huria)
    quanto i cambiamenti reali sul terreno: i due effetti vanno scorporati.
  - Se l'interesse e' davvero la dinamica dei boma (insediamenti informali),
    valutare dataset di telerilevamento multi-temporale come GHSL Built-Up
    Surface (JRC, epoche 1975-2030) o World Settlement Footprint Evolution
    (DLR), oppure dataset pubblicati da progetti specifici di boma-mapping.

Installazione (Pop!_OS / Ubuntu):
    pip install ohsome geopandas pandas matplotlib requests --break-system-packages
"""

import time as time_module
from pathlib import Path

import pandas as pd
import requests
from ohsome import OhsomeClient

# ---------------------------------------------------------------------------
# 1. CONFIGURAZIONE
# ---------------------------------------------------------------------------

# Bounding box approssimativo che copre i distretti a maggioranza masai del
# nord Tanzania: Ngorongoro, Loliondo, Monduli, Longido (regione Arusha),
# Simanjiro e Kiteto (regione Manyara).
# Formato: (lon_min, lat_min, lon_max, lat_max)
AOI_BBOX = (34.6, -5.4, 37.5, -1.9)

# Per un'area di interesse precisa invece del rettangolo, sostituire AOI_BBOX
# con un poligono caricato da un file dei confini distrettuali (es. scaricato
# da HDX "Tanzania - Subnational Administrative Boundaries" o geoBoundaries.org):
#
#   import geopandas as gpd
#   distretti = gpd.read_file("confini_distretti_tanzania.geojson")
#   aoi = distretti[distretti["adm2_name"].isin(
#       ["Ngorongoro", "Monduli", "Longido", "Simanjiro", "Kiteto"]
#   )]
#   # poi passare bpolys=aoi invece di bboxes=AOI_BBOX nelle chiamate sotto

# Range temporale: uno snapshot al 1 gennaio di ogni anno dal 2009 al 2023
# (l'ultimo punto, 2023-01-01, rappresenta lo stato cumulato a fine 2022).
TIME_RANGE = "2009-01-01/2023-01-01/P1Y"

OUTPUT_DIR = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Filtri ohsome (sintassi "ohsome filter", simile a Overpass QL ma pensata
# per le query aggregate). Tre proxy diversi per leggere il fenomeno da
# angolazioni complementari:
FILTERS = {
    "villaggi_hamlet": "(place=village or place=hamlet) and type:node",
    "edifici_mappati": "building=* and type:way",
    "area_residenziale_m2": "landuse=residential and type:way and geometry:polygon",
}


def get_client() -> OhsomeClient:
    client = OhsomeClient()
    print(f"Copertura storica ohsome: {client.start_timestamp} -> {client.end_timestamp}")
    return client


# ---------------------------------------------------------------------------
# 2. SERIE TEMPORALI (conteggi / area per anno)
# ---------------------------------------------------------------------------

def build_yearly_dataframe(client: OhsomeClient) -> pd.DataFrame:
    """Interroga ohsome per ciascun filtro e unisce i risultati in un unico
    DataFrame con una riga per anno."""

    frames = []

    for label, filt in FILTERS.items():
        if "area" in label:
            response = client.elements.area.post(
                bboxes=AOI_BBOX, time=TIME_RANGE, filter=filt
            )
        else:
            response = client.elements.count.post(
                bboxes=AOI_BBOX, time=TIME_RANGE, filter=filt
            )

        df = response.as_dataframe()
        df = df.rename(columns={"value": label})[[label]]
        frames.append(df)

    combined = pd.concat(frames, axis=1)
    combined.index.name = "data"
    combined.index = pd.to_datetime(combined.index).year
    combined.index.name = "anno"
    return combined


# ---------------------------------------------------------------------------
# 3. ESTRAZIONE GEOMETRIE PER ANNI SPECIFICI (per uso in QGIS/GIS)
# ---------------------------------------------------------------------------

def export_yearly_snapshots(client: OhsomeClient, years=(2009, 2014, 2018, 2022)):
    """Salva un GeoJSON per ciascun anno scelto con le posizioni dei
    villaggi/hamlet cosi' come risultano mappati in OSM in quella data,
    utile per visualizzare la diffusione spaziale nel tempo in QGIS."""

    timestamps = [f"{y}-01-01" for y in years]

    response = client.elements.geometry.post(
        bboxes=AOI_BBOX,
        time=timestamps,
        filter=FILTERS["villaggi_hamlet"],
        properties="tags",
    )

    # multi_index=False per avere "@snapshotTimestamp" come colonna normale
    # invece che come parte dell'indice (vedi ohsome-py: response.py)
    gdf = response.as_dataframe(multi_index=False)
    gdf["@snapshotTimestamp"] = pd.to_datetime(gdf["@snapshotTimestamp"])

    for y in years:
        subset = gdf[gdf["@snapshotTimestamp"].dt.year == y]
        out_path = OUTPUT_DIR / f"villaggi_masai_{y}.geojson"
        subset.to_file(out_path, driver="GeoJSON")
        print(f"Salvato: {out_path} ({len(subset)} elementi)")


# ---------------------------------------------------------------------------
# 4. GRAFICO RIASSUNTIVO
# ---------------------------------------------------------------------------

def plot_trend(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    df["villaggi_hamlet"].plot(ax=axes[0], marker="o", color="#6b4226")
    axes[0].set_ylabel("n. villaggi/hamlet")

    df["edifici_mappati"].plot(ax=axes[1], marker="o", color="#8a9a5b")
    axes[1].set_ylabel("n. edifici mappati")

    (df["area_residenziale_m2"] / 1e6).plot(ax=axes[2], marker="o", color="#c1440e")
    axes[2].set_ylabel("area residenziale (km²)")
    axes[2].set_xlabel("anno")

    fig.suptitle("Aree masai, nord Tanzania - proxy OSM 2009-2022")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "trend_insediamenti_masai.png"
    fig.savefig(out_path, dpi=150)
    print(f"Grafico salvato in: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    client = get_client()

    df = build_yearly_dataframe(client)
    csv_path = OUTPUT_DIR / "trend_insediamenti_masai_2009_2022.csv"
    df.to_csv(csv_path)
    print(f"CSV salvato in: {csv_path}")
    print(df)

    plot_trend(df)
    export_yearly_snapshots(client)


if __name__ == "__main__":
    main()
