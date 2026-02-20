"""
Estonia Weather Maps  –  EMHI observations, cartopy, QML colormaps
===================================================================
Looks like the Polish reference map: smooth colour field, white outside
the country, sharp black coastlines/borders, small station value labels.

Install once:
    py -m pip install requests matplotlib cartopy numpy scipy pillow

QML files must be in the SAME folder as this script:
    temperature_color_table_high.qml
    wind_gust_color_table.qml
    pressure_color_table.qml
    precipitation_color_table.qml

Run:
    py estonia_weather_map.py

Output (one PNG per parameter, saved next to the script):
    map_1_temperature.png   map_2_wind_speed.png   map_3_wind_gust.png
    map_4_pressure.png      map_5_humidity.png     map_6_precipitation.png
"""

import os, time, datetime, threading, warnings
import requests, xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm, ListedColormap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

UPDATE_INTERVAL = 300
DPI             = 200
_EXTENT_EE   = [21.3, 28.7, 57.3, 60.05]
_EXTENT_LV   = [20.8, 28.3, 55.4, 58.2]
_EXTENT_BOTH = [20.8, 28.7, 55.4, 60.05]
EXTENT = _EXTENT_BOTH
GRID_STEP       = 0.004
SIGMA           = 2.2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_qml(filename, vmin, vmax):
    path = os.path.join(SCRIPT_DIR, filename)
    root = ET.parse(path).getroot()
    vals, cols = [], []
    for item in root.findall(".//colorrampshader/item"):
        v = float(item.get("value"))
        h = item.get("color").lstrip("#")
        r,g,b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
        vals.append(v); cols.append((r,g,b))
    vals = np.array(vals, dtype=float)
    cols = np.array(cols, dtype=float)
    idx  = np.argsort(vals)
    vals, cols = vals[idx], cols[idx]
    pos  = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "qml", list(zip(pos, cols)), N=2048)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm

def _bc(stops, v0, v1):
    """Built-in fallback colormap."""
    pos = [max(0.,min(1.,(v-v0)/(v1-v0))) for v,_ in stops]
    cm  = LinearSegmentedColormap.from_list(
        "fb", list(zip(pos, [c for _,c in stops])), N=2048)
    return cm, Normalize(vmin=v0, vmax=v1)

def load_cmaps():
    c = {}
    try:
        c["temp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except Exception as e:
        print(f"  [warn] temp QML: {e}")
        c["temp"] = _bc([(-40,"#ff6eff"),(-20,"#32007f"),(-10,"#259aff"),
                          (0,"#d9ecff"),(10,"#52ca0b"),(20,"#f4bd0b"),
                          (30,"#af0f14"),(45,"#c5c5c5")], -40, 45)
    try:
        c["wind"] = parse_qml("wind_gust_color_table.qml", 0, 50)
    except Exception as e:
        print(f"  [warn] wind QML: {e}")
        c["wind"] = _bc([(0,"#ffffff"),(10,"#3c96f5"),(20,"#ffa000"),
                          (30,"#e11400"),(50,"#8c645a")], 0, 50)
    c["gust"] = c["wind"]
    try:
        c["pres"] = parse_qml("pressure_color_table.qml", 890, 1064)
    except Exception as e:
        print(f"  [warn] pres QML: {e}")
        c["pres"] = _bc([(965,"#32007f"),(990,"#91ccff"),(1000,"#07a127"),
                          (1013,"#f3fb01"),(1030,"#f4520b"),(1050,"#f0a0a0")], 960, 1055)
    try:
        c["prec"] = parse_qml("precipitation_color_table.qml", 0, 125)
    except Exception as e:
        print(f"  [warn] prec QML: {e}")
        c["prec"] = _bc([(0,"#f0f0f0"),(1,"#0482ff"),(5,"#1acf05"),
                          (10,"#ff7f27"),(20,"#bf0000"),(50,"#64007f")], 0, 50)
    c["hum"]  = _bc([(0,"#ffffff"),(20,"#d0eaff"),(50,"#55a3e0"),
                      (80,"#084a90"),(100,"#021860")], 0, 100)
    try:
        c["dewp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except:
        c["dewp"] = c["temp"]
    return c

def _parse_emhi(content):
    """Parse Estonia EMHI XML observations."""
    root = ET.fromstring(content)
    ts = root.get("timestamp", "")
    try:
        dt = datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc)
        time_str = dt.strftime("%Y-%m-%d  %H:%M UTC")
    except:
        time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")

    def fv(s, tag):
        e = s.find(tag)
        if e is not None and e.text and e.text.strip() not in ("","null","None"):
            try: return float(e.text.strip())
            except: pass
        return np.nan

    stations = []
    for s in root.findall("station"):
        lat=fv(s,"latitude"); lon=fv(s,"longitude")
        if np.isnan(lat) or np.isnan(lon): continue
        ne = s.find("name")
        stations.append({
            "name"    : ne.text.strip() if ne is not None and ne.text else "?",
            "country" : "EE",
            "lat":lat, "lon":lon,
            "temp"    : fv(s,"airtemperature"),
            "wind"    : fv(s,"windspeed"),
            "wind_dir": fv(s,"winddirection"),
            "gust"    : fv(s,"windspeedmax"),
            "hum"     : fv(s,"relativehumidity"),
            "prec"    : fv(s,"precipitations"),
            "pres"    : fv(s,"airpressure"),
        })
        t  = stations[-1]["temp"]
        rh = stations[-1]["hum"]
        if not (np.isnan(t) or np.isnan(rh)):
            a, b = 17.625, 243.04
            g = np.log(rh / 100.0) + a * t / (b + t)
            stations[-1]["dewp"] = round(b * g / (a - g), 1)
        else:
            stations[-1]["dewp"] = np.nan
    return stations, time_str

def _parse_latvia(content):
    """Parse Latvia LVGMC XML observations.
    Latvia's feed uses the same basic structure as EMHI.
    URL: https://www.meteo.lv/meteorologija-un-klimats/meteorologiskie-novarojumi/noveroojumu-dati/xml-dati/
    Falls back gracefully if unavailable.
    """
    try:
        root = ET.fromstring(content)
    except Exception:
        return []

    def fv(s, tag):
        for t in [tag, tag.lower(), tag.upper()]:
            e = s.find(t)
            if e is not None and e.text and e.text.strip() not in ("","null","None","-"):
                try: return float(e.text.strip())
                except: pass
        return np.nan

    stations = []
    for s in list(root.iter("Station")) + list(root.iter("station")):
        lat = np.nan; lon = np.nan
        for lt in ["Latitude","latitude","lat"]:
            v = fv(s, lt)
            if not np.isnan(v): lat = v; break
        for ln in ["Longitude","longitude","lon"]:
            v = fv(s, ln)
            if not np.isnan(v): lon = v; break
        if np.isnan(lat) or np.isnan(lon): continue
        if not (20.5 <= lon <= 28.3 and 55.5 <= lat <= 58.1): continue

        name = ""
        for nt in ["Name","name","StationName","stationname"]:
            e = s.find(nt)
            if e is not None and e.text and e.text.strip():
                name = e.text.strip(); break

        stations.append({
            "name"    : name or "LV",
            "country" : "LV",
            "lat":lat, "lon":lon,
            "temp"    : fv(s,"AirTemperature") if not np.isnan(fv(s,"AirTemperature")) else fv(s,"airtemperature"),
            "wind"    : fv(s,"WindSpeed")      if not np.isnan(fv(s,"WindSpeed"))      else fv(s,"windspeed"),
            "wind_dir": fv(s,"WindDirection")  if not np.isnan(fv(s,"WindDirection"))  else fv(s,"winddirection"),
            "gust"    : fv(s,"WindGust")       if not np.isnan(fv(s,"WindGust"))       else fv(s,"maxwindspeedvalue"),
            "hum"     : fv(s,"RelativeHumidity") if not np.isnan(fv(s,"RelativeHumidity")) else fv(s,"relativehumidity"),
            "prec"    : fv(s,"Precipitation")  if not np.isnan(fv(s,"Precipitation"))  else fv(s,"precipitations"),
            "pres"    : fv(s,"AirPressure")    if not np.isnan(fv(s,"AirPressure"))    else fv(s,"airpressure"),
        })
    return stations

_LV_STATIONS = [
    ("Rīga",        56.946, 24.106),
    ("Liepāja",     56.505, 21.011),
    ("Ventspils",   57.401, 21.544),
    ("Jelgava",     56.652, 23.721),
    ("Jūrmala",     56.968, 23.771),
    ("Jēkabpils",   56.499, 25.877),
    ("Valmiera",    57.541, 25.426),
    ("Rēzekne",     56.510, 27.331),
    ("Daugavpils",  55.875, 26.536),
    ("Kolka",       57.748, 22.594),
    ("Mersrags",    57.335, 23.119),
    ("Ainažu",      57.865, 24.355),
    ("Sigulda",     57.153, 24.852),
    ("Alūksne",     57.432, 27.046),
    ("Zilupe",      56.386, 28.128),
    ("Pāvilosta",   56.887, 21.193),
    ("Saldus",      56.666, 22.493),
    ("Bauska",      56.410, 24.193),
    ("Stende",      57.165, 22.532),
    ("Priekuļi",    57.321, 25.352),
]

def _fetch_latvia_openmeteo():
    """Fetch current conditions for Latvia stations via Open-Meteo API (free, no key).
    Uses the 'pressure_msl' variable which is already sea-level pressure — no correction needed.
    """
    lats = ",".join(str(s[1]) for s in _LV_STATIONS)
    lons = ",".join(str(s[2]) for s in _LV_STATIONS)
    params = (
        "temperature_2m,wind_speed_10m,wind_direction_10m,"
        "wind_gusts_10m,relative_humidity_2m,precipitation,pressure_msl"
    )
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        f"&current={params}"
        f"&wind_speed_unit=ms"
        f"&timeformat=unixtime"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list):
        data = [data]

    stations = []
    for i, (name, lat, lon) in enumerate(_LV_STATIONS):
        if i >= len(data): break
        cur = data[i].get("current", {})
        stations.append({
            "name"    : name,
            "country" : "LV",
            "lat": lat, "lon": lon,
            "temp"    : cur.get("temperature_2m"),
            "wind"    : cur.get("wind_speed_10m"),
            "wind_dir": cur.get("wind_direction_10m"),
            "gust"    : cur.get("wind_gusts_10m"),
            "hum"     : cur.get("relative_humidity_2m"),
            "prec"    : cur.get("precipitation"),
            "pres"    : cur.get("pressure_msl"),
        })
        t  = stations[-1]["temp"]
        rh = stations[-1]["hum"]
        if not (np.isnan(t) or np.isnan(rh)):
            a, b = 17.625, 243.04
            gamma = np.log(rh / 100.0) + a * t / (b + t)
            stations[-1]["dewp"] = round(b * gamma / (a - gamma), 1)
        else:
            stations[-1]["dewp"] = np.nan
        for k in ["temp","wind","wind_dir","gust","hum","prec","pres","dewp"]:
            if stations[-1][k] is None:
                stations[-1][k] = np.nan
    return stations

def fetch():
    hdr = {"User-Agent": "Mozilla/5.0"}
    all_stations = []

    if EXTENT in (_EXTENT_EE, _EXTENT_BOTH):
        r = requests.get("https://www.ilmateenistus.ee/ilma_andmed/xml/observations.php",
                         headers=hdr, timeout=20)
        r.raise_for_status()
        ee_stations, time_str = _parse_emhi(r.content)
        print(f"  EE: {len(ee_stations)} stations  |  {time_str}")
        all_stations.extend(ee_stations)
    else:
        time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")

    if EXTENT in (_EXTENT_LV, _EXTENT_BOTH):
        try:
            lv_stations = _fetch_latvia_openmeteo()
            print(f"  LV: {len(lv_stations)} stations from Open-Meteo")
            all_stations.extend(lv_stations)
        except Exception as e:
            print(f"  LV: Open-Meteo fetch failed: {e} — continuing without Latvia")

    print(f"  Total: {len(all_stations)} stations")
    return all_stations, time_str

def make_grid():
    lons = np.arange(EXTENT[0], EXTENT[1]+GRID_STEP, GRID_STEP)
    lats = np.arange(EXTENT[2], EXTENT[3]+GRID_STEP, GRID_STEP)
    return np.meshgrid(lons, lats)

def interpolate(stations, key, gx, gy):
    ok  = [s for s in stations if not np.isnan(s.get(key, np.nan))]
    if len(ok) < 4: return None
    pts = np.array([(s["lon"],s["lat"]) for s in ok])
    vs  = np.array([s[key]              for s in ok])

    zi  = griddata(pts, vs, (gx, gy), method="linear")
    znn = griddata(pts, vs, (gx, gy), method="nearest")
    zi  = np.where(np.isnan(zi), znn, zi)
    zs = gaussian_filter(zi, sigma=SIGMA)
    return zs

import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
import matplotlib.patches as mpatches

_SHPCACHE = {}

def _get_country_geoms(iso_keep_tuple):
    if iso_keep_tuple in _SHPCACHE:
        return _SHPCACHE[iso_keep_tuple]
    iso_keep = set(iso_keep_tuple)
    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shp_path)
    keep_geoms  = []
    other_geoms = []
    for rec in reader.records():
        iso  = rec.attributes.get("ISO_A2", "")
        iso2 = rec.attributes.get("ADM0_A3", "")
        geom = rec.geometry
        if iso in iso_keep or iso2 in iso_keep:
            keep_geoms.append(geom)
        else:
            other_geoms.append(geom)
    keep_union  = unary_union(keep_geoms)  if keep_geoms  else None
    other_union = unary_union(other_geoms) if other_geoms else None
    _SHPCACHE[iso_keep_tuple] = (keep_union, other_union)
    return keep_union, other_union

_PE  = [mpe.withStroke(linewidth=3, foreground="white")]
_PE2 = [mpe.withStroke(linewidth=4, foreground="white")]

def render_one(stations, key, cmap, norm, title, unit, fmt,
               time_str, gx, gy, outfile, wind_arrows=False):
    grid = interpolate(stations, key, gx, gy)
    ok   = [s for s in stations if not np.isnan(s.get(key, np.nan))]
    if ok:
        vmin_obs = min(s[key] for s in ok)
        vmax_obs = max(s[key] for s in ok)
        s_min    = min(ok, key=lambda s: s[key])
        s_max    = max(ok, key=lambda s: s[key])
    else:
        vmin_obs = vmax_obs = 0
        s_min = s_max = None

    if EXTENT == _EXTENT_EE:
        iso_keep = ("EE",)
        region_label = "Eesti"
        credits = "Andmeallikas: EMHI vaatlusjaamad  •  ilmateenistus.ee"
    elif EXTENT == _EXTENT_LV:
        iso_keep = ("LV",)
        region_label = "Läti"
        credits = "Andmeallikas: LVĢMC / Open-Meteo  •  open-meteo.com"
    else:
        iso_keep = ("EE", "LV")
        region_label = "Eesti + Läti"
        credits = ("Andmeallikas: EMHI vaatlusjaamad (ilmateenistus.ee)  •  "
                   "LVĢMC / Open-Meteo (open-meteo.com)")

    keep_union, other_union = _get_country_geoms(iso_keep)

    fig = plt.figure(figsize=(16, 11), facecolor="white", dpi=DPI)
    ax  = fig.add_axes([0.04, 0.06, 0.82, 0.88],
                       projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")

    if grid is not None:
        masked = np.ma.masked_invalid(grid)
        ax.pcolormesh(gx, gy, masked,
                      cmap=cmap, norm=norm,
                      shading="auto",
                      transform=ccrs.PlateCarree(),
                      rasterized=True, zorder=2)

    ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                   facecolor="white", zorder=3)
    if other_union is not None:
        ax.add_geometries([other_union], ccrs.PlateCarree(),
                          facecolor="white", edgecolor="none", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("10m"),
                   facecolor="white", edgecolor="#666666",
                   linewidth=0.5, zorder=4)

    ax.coastlines(resolution="10m", linewidth=1.1,
                  color="#222222", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   linestyle="-", edgecolor="#333333",
                   linewidth=0.9, zorder=5)
    admin1 = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m",
        facecolor="none", edgecolor="#555555", linewidth=0.5)
    ax.add_feature(admin1, zorder=5)

    lon0,lon1,lat0,lat1 = EXTENT
    if wind_arrows:
        for s in ok:
            slon, slat = s["lon"], s["lat"]
            if not (lon0-0.1 <= slon <= lon1+0.1 and
                    lat0-0.1 <= slat <= lat1+0.1): continue
            wd = s.get("wind_dir", np.nan)
            if np.isnan(wd) or s[key] < 0.5: continue
            wr = np.radians(wd)
            u  = -s[key]*np.sin(wr)*0.05
            v  = -s[key]*np.cos(wr)*0.035
            ax.annotate("",
                xy    =(slon+u, slat+v),
                xytext=(slon,   slat),
                arrowprops=dict(arrowstyle="-|>", color="#111111",
                                lw=0.7, mutation_scale=6),
                transform=ccrs.PlateCarree(), zorder=9)

    margin = 0.05
    for s in ok:
        slon, slat = s["lon"], s["lat"]
        if not (lon0-margin <= slon <= lon1+margin and
                lat0-margin <= slat <= lat1+margin): continue
        ax.plot(slon, slat, "s",
                color="black", ms=3.2, mec="black", mew=0.0,
                transform=ccrs.PlateCarree(), zorder=10)
        ax.text(slon+0.025, slat+0.02,
                fmt.format(s[key]),
                fontsize=6.5, fontweight="bold", color="black",
                path_effects=_PE,
                transform=ccrs.PlateCarree(), zorder=11)

    if s_min and s_max:
        ax.text(0.012, 0.17, f"{fmt.format(vmax_obs)}{unit}",
                transform=ax.transAxes, fontsize=14,
                fontweight="bold", color="#cc0000",
                path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.12, s_max["name"],
                transform=ax.transAxes, fontsize=9,
                fontweight="bold", color="#cc0000",
                path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.07, f"{fmt.format(vmin_obs)}{unit}",
                transform=ax.transAxes, fontsize=14,
                fontweight="bold", color="#0055cc",
                path_effects=_PE2, zorder=20)
        ax.text(0.012, 0.02, s_min["name"],
                transform=ax.transAxes, fontsize=9,
                fontweight="bold", color="#0055cc",
                path_effects=_PE2, zorder=20)

    cax = fig.add_axes([0.875, 0.06, 0.022, 0.88])
    cb  = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax, orientation="vertical", extend="both")
    cb.set_label(unit, fontsize=9, rotation=270, labelpad=14)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)

    ax.set_title(
        f"{region_label}  •  {title}  •  1 km\n{time_str}",
        fontsize=13, fontweight="bold", pad=10,
        loc="center", color="#111111")
    fig.text(0.5, 0.005, credits, ha="center", fontsize=8, color="#888888")

    fig.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved  {os.path.basename(outfile)}")


PANELS = [
    ("temp","Õhutemperatuur 2 m (°C)",           "°C",  "{:.1f}", False, "map_1_temperature.png"),
    ("wind","Tuule kiirus (m/s)",                 "m/s", "{:.1f}", True,  "map_2_wind_speed.png"),
    ("gust","Tuule puhang – max (m/s)",           "m/s", "{:.1f}", True,  "map_3_wind_gust.png"),
    ("pres","Meretasemele taandatud rõhk (hPa)",  "hPa", "{:.1f}", False, "map_4_pressure.png"),
    ("hum", "Suhteline niiskus (%)",              "%",   "{:.0f}", False, "map_5_humidity.png"),
    ("prec","Sademete hulk (mm/h)",               "mm",  "{:.1f}", False, "map_6_precipitation.png"),
    ("dewp","Kastepunkt 2 m (°C)",                "°C",  "{:.1f}", False, "map_7_dewpoint.png"),
]

def run_once(cmaps, gx, gy):
    try:
        stations, time_str = fetch()
    except Exception as e:
        print(f"  Fetch error: {e}"); return
    for (key, title, unit, fmt, arrows, fname) in PANELS:
        outfile = os.path.join(SCRIPT_DIR, fname)
        try:
            render_one(stations, key, *cmaps[key],
                       title, unit, fmt, time_str, gx, gy, outfile,
                       wind_arrows=arrows)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  Error {fname}: {e}")

def loop(cmaps, gx, gy):
    while True:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] Updating …")
        run_once(cmaps, gx, gy)
        print(f"  Next update in {UPDATE_INTERVAL//60} min.")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          Ilmakaart  –  Vali piirkond                ║")
    print("║          Weather Map  –  Choose region               ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  1)  Eesti (Estonia only)                            ║")
    print("║  2)  Läti  (Latvia only)                             ║")
    print("║  3)  Eesti + Läti  (Both countries)                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            choice = input("  Vali / Choose [1/2/3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped."); raise SystemExit

        if choice == "1":
            EXTENT = _EXTENT_EE
            label  = "Eesti"
            break
        elif choice == "2":
            EXTENT = _EXTENT_LV
            label  = "Läti"
            break
        elif choice == "3":
            EXTENT = _EXTENT_BOTH
            label  = "Eesti + Läti"
            break
        else:
            print("  Palun sisesta 1, 2 või 3.  /  Please enter 1, 2 or 3.")

    print()
    print(f"  Piirkond / Region: {label}")
    print("  Auto-updates every 5 minutes. Press Ctrl+C to stop.")
    print()

    cmaps  = load_cmaps()
    gx, gy = make_grid()
    run_once(cmaps, gx, gy)
    t = threading.Thread(target=loop, args=(cmaps, gx, gy), daemon=True)
    t.start()
    try:
        while True: time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped.")