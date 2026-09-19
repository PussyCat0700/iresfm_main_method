# -*- coding: utf-8 -*-
"""
Pseudocode: Single-site dataset generation (power + paired weather)

Dataset specification
---------------------
Input (one site):
  - PLANT_NO: int
  - LONGITUDE: float
  - LATITUDE: float
  - CONFIRM_POWER_MW: float
  - MODE: "wind" or "solar"
  - START_UTC, END_UTC: UTC timestamps on the hour (minute=second=0), inclusive.

Output files:
  1) {PLANT_NO}.csv                      (power, 15-min)
       columns: time, power
       time: 15-min grid from START_UTC to END_UTC (inclusive)
       power: MW

  2) info.csv                            (site registry)
       one row per site/run:
         PLANT_NO, LONGITUDE, LATITUDE, CONFIRM_POWER_MW, MODE, START_UTC, END_UTC, SEED, POWER_META_JSON

  3) weather_{PLANT_NO}.npy              (paired weather tensor)
       shape: (hours, 64, 27), float32
         hours = number of hourly timestamps from START_UTC to END_UTC (inclusive)
         64    = flattened 8×8 patch around site (8×8 -> 64)
         27    = 11 (HRES) + 8 (Pangu) + 8 (FengWu)

Time convention
---------------
- All timestamps are UTC.
- Weather is hourly (UTC) and aligned to the same hourly index used for hourly power.
- HRES / Pangu / FengWu are assembled day-by-day using cycles at 00/06/12/18 UTC, with lead 0..5.

Variables
---------
Power meteorology (ERA5 point):
  Wind:  u10, v10, u100, v100, t2m(K), sp(Pa)
  Solar: u10, v10, t2m(K), sp(Pa), tp, fdir(J/m² accumulated hourly)

Solar radiation conversion for power:
  ghi(W/m²) = fdir(J/m²) / 3600

Paired weather (gridded 8×8 patch):
  HRES(11):
    10U,10V,100U,100V,2T,200U,200V,SSR,SP,TCC,TP
    SSR and TP are accumulated => instantaneous via first difference along time.

  Pangu(8) and FengWu(8) use the SAME set:
    u10,v10,t2m,z1000,q1000,t1000,u1000,v1000
"""

# -------------------------------------------------------------------
# 0) Configuration
# -------------------------------------------------------------------
class Config:
    seed = 42

    # Final power resolution
    power_output_freq = "15min"

    # Patch size for paired weather
    patch_size = 8  # 8×8 -> 64

    # Daily cycle assembly (UTC)
    cycles_utc = [0, 6, 12, 18]
    steps_per_cycle = 6  # lead 0..5

    # Paired weather variable lists
    hres_vars = ["10U","10V","100U","100V","2T","200U","200V","SSR","SP","TCC","TP"]  # 11
    hres_accum_vars = ["SSR","TP"]

    ai_vars = ["u10","v10","t2m","z1000","q1000","t1000","u1000","v1000"]  # 8 (Pangu + FengWu)

    # ERA5 grid definition for point extraction (example: 0.25° grid)
    lon_grid = [0, 0.25, 0.50, ..., 359.75]
    lat_grid = [90, 89.75, ..., -90.0]

    # Wind model retries (avoid rare all-zero outputs)
    wind_max_tries = 30


# -------------------------------------------------------------------
# 1) Public entry point
# -------------------------------------------------------------------
def GenerateDatasetForSite(PLANT_NO, LONGITUDE, LATITUDE, CONFIRM_POWER_MW,
                           MODE, START_UTC, END_UTC,
                           OUT_DIR, cfg):
    """
    End-to-end generation for a single site.

    Inputs:
      PLANT_NO, LONGITUDE, LATITUDE, CONFIRM_POWER_MW, MODE, START_UTC, END_UTC
      START_UTC and END_UTC must be on the hour and are inclusive.

    Outputs:
      - OUT_DIR/power/{PLANT_NO}.csv
      - OUT_DIR/info.csv
      - OUT_DIR/weather/weather_{PLANT_NO}.npy
    """
    SeedAll(cfg.seed)
    AssertHourlyEndpoints(START_UTC, END_UTC)

    # Build time indices
    time_hourly = DateRange(START_UTC, END_UTC, freq="1h")      # used by hourly power + hourly weather
    time_15min  = DateRange(START_UTC, END_UTC, freq="15min")   # used only by final power CSV

    # ---- (A) Power generation: hourly -> 15-min ----
    power_hourly, power_meta = SimulatePowerHourly(
        MODE, LATITUDE, LONGITUDE, CONFIRM_POWER_MW, time_hourly, cfg
    )
    power_15min = InterpolateHourlyTo15Min(power_hourly, time_15min)

    SaveCSV(PathJoin(OUT_DIR, "power", f"{PLANT_NO}.csv"),
            columns=["time", "power"],
            rows=[(t, v) for t, v in zip(power_15min.index, power_15min.values)])

    # ---- (B) Site registry ----
    AppendInfoRow(PathJoin(OUT_DIR, "info.csv"),
                  PLANT_NO, LONGITUDE, LATITUDE, CONFIRM_POWER_MW, MODE,
                  START_UTC, END_UTC, cfg.seed, power_meta)

    # ---- (C) Paired weather tensor (hours,64,27) ----
    weather = BuildPairedWeatherTensor(PLANT_NO, LATITUDE, LONGITUDE, time_hourly, cfg)
    SaveNPY(PathJoin(OUT_DIR, "weather", f"weather_{PLANT_NO}.npy"), weather)

    return


# -------------------------------------------------------------------
# 2) Power generation (ERA5 point -> physics -> hourly power -> 15-min)
# -------------------------------------------------------------------
def SimulatePowerHourly(mode, lat, lon, capacity_mw, time_hourly, cfg):
    """
    Hourly power simulation driven by ERA5 point meteorology.

    For each hour:
      1) read ERA5 grids for required variables
      2) bilinear interpolate to station point
      3) run physical power model (wind farm or PV system)

    Returns:
      power_hourly: series indexed by time_hourly, unit MW
      power_meta: dict describing sampled physical parameters (useful for reproducibility)
    """
    if mode == "wind":
        var_list = ["u100","v100","u10","v10","t2m","sp"]
    else:
        var_list = ["u10","v10","t2m","sp","tp","fdir"]

    met = FetchEra5PointTimeseries(lat, lon, time_hourly, var_list, cfg)

    if mode == "wind":
        power_hourly, meta = WindMeteorologyToPower(met, capacity_mw, cfg)
    else:
        power_hourly, meta = SolarMeteorologyToPower(met, lat, lon, capacity_mw, cfg)

    return power_hourly, meta


def FetchEra5PointTimeseries(lat, lon, time_hourly, var_list, cfg):
    """
    Read ERA5 hourly grids and interpolate each variable to a single station point.

    Inputs:
      - lat, lon: station coordinate
      - time_hourly: hourly UTC index
      - var_list: list of ERA5 variable names

    Output:
      met: table indexed by time_hourly, columns=var_list
    """
    lon_0_360 = NormalizeLonTo0_360(lon)

    table = {v: [] for v in var_list}
    for ts in time_hourly:
        for v in var_list:
            grid = ReadEra5Grid(v, ts)                # -> 2D grid[lat,lon] (or squeezable 3D)
            grid2d = SqueezeTo2D(grid)
            val = BilinearInterpToPoint(grid2d, cfg.lat_grid, cfg.lon_grid, lat, lon_0_360)
            table[v].append(val)

    return TableFromDict(table, index=time_hourly)


def InterpolateHourlyTo15Min(power_hourly, time_15min):
    """
    Time-based linear interpolation from hourly power to 15-min timestamps.
    """
    union_index = UnionIndex(power_hourly.index, time_15min)
    tmp = power_hourly.reindex(union_index)
    tmp = InterpolateTime(tmp)
    return tmp.reindex(time_15min)


def WindMeteorologyToPower(met, capacity_mw, cfg):
    """
    Wind physical model (windpowerlib-style).

    Inputs
    ------
    met: hourly table with columns:
         u100, v100, u10, v10, t2m, sp
         - u*, v*: wind components (m/s)
         - t2m   : 2m air temperature (K)
         - sp    : surface pressure (Pa)
    capacity_mw: installed capacity (MW)
    cfg.wind_max_tries: retry count for random configuration sampling

    Output
    ------
    power_hourly: time series (MW), indexed by met.index
    meta: dict of sampled parameters for reproducibility

    Model logic (high-level)
    ------------------------
    1) Convert wind components to wind speed magnitude:
         ws10  = sqrt(u10^2  + v10^2)
         ws100 = sqrt(u100^2 + v100^2)

    2) Prepare the weather input required by a farm model:
         - wind speed at 10 m
         - wind speed at 100 m
         - temperature at 2 m
         - pressure at surface
         - roughness length (sampled; affects vertical wind profile)

       In windpowerlib, this is typically represented as a DataFrame with MultiIndex columns:
         ('wind_speed','10'), ('wind_speed','100'), ('temperature','2'),
         ('pressure','0'), ('roughness_length','0')
       but the exact data structure is left to the implementation.

    3) Sample a plausible farm configuration and model options:
         (a) Turbine type:
             choose a turbine with an available power curve and hub-height options
         (b) Hub height:
             choose one supported hub height for that turbine
         (c) Roughness length:
             choose from typical land/sea roughness values with a prior distribution
         (d) Farm efficiency:
             scalar in [0.6, 0.95] to represent wake/availability losses
         (e) Density model:
             "barometric" or "ideal_gas" (both use pressure + temperature)
         (f) Wind speed vertical model:
             - "hellman" (power-law shear with Hellman exponent)
             - interpolation / log-interpolation alternatives
         (g) Hellman exponent:
             sample around onshore/offshore priors (e.g. 1/7, 1/9) with small noise
         (h) Optional smoothing:
             parameters to mimic spatial wind variability inside the farm

       Because parameters are randomized, a small fraction of configurations can produce
       degenerate outputs (e.g., all zeros). We retry up to cfg.wind_max_tries and accept
       the first non-all-zero result.

    4) Run the farm model:
         power_hourly = RunWindFarmModel(wind_weather, turbine, farm_params, modelchain_data)
       Output is clipped at 0 MW.
    """
    # 1) derive wind speed at reference heights
    ws100 = sqrt(met["u100"]**2 + met["v100"]**2)
    ws10  = sqrt(met["u10"]**2  + met["v10"]**2)

    # 2) conceptual weather object for the wind model
    wind_weather = {
        "wind_speed_10m": ws10,
        "wind_speed_100m": ws100,
        "temperature_2m": met["t2m"],      # K
        "pressure_surface": met["sp"],     # Pa
        # "roughness_length": will be added after sampling
    }

    # 3) retry loop for random configuration sampling
    for attempt in range(cfg.wind_max_tries):
        turbine_type, hub_height = SampleWindTurbine()
        roughness_length = SampleRoughnessLength()

        farm_efficiency = RandUniform(0.6, 0.95)

        density_model = RandChoice(["barometric", "ideal_gas"])
        wind_speed_model = RandChoice(["hellman", "interpolation_extrapolation", "log_interpolation_extrapolation"])

        base_exp = RandChoice([1/7, 1/9])
        hellman_exp = RandNormal(base_exp, 0.01)

        smoothing = RandChoice([True, False])
        block_width = RandNormal(0.5, 0.1)
        std_method = RandChoice(["turbulence_intensity", "Staffell_Pfenninger"])
        smoothing_order = RandChoice(["turbine_power_curves", "wind_farm_power_curves"])

        wind_weather["roughness_length"] = roughness_length

        modelchain_data = {
            "power_output_model": "power_curve",
            "density_model": density_model,
            "wind_speed_model": wind_speed_model,
            "hellman_exp": hellman_exp,
            "smoothing": smoothing,
            "block_width": block_width,
            "standard_deviation_method": std_method,
            "smoothing_order": smoothing_order,
            "wake_losses_model": None,
        }

        power_hourly = RunWindFarmModel(
            wind_weather=wind_weather,
            turbine_type=turbine_type,
            hub_height=hub_height,
            total_capacity_mw=capacity_mw,
            farm_efficiency=farm_efficiency,
            modelchain_data=modelchain_data,
        )

        power_hourly = ClipLower(power_hourly, 0.0)

        if not AllZero(power_hourly):
            meta = {
                "nonzero": True,
                "attempt_used": attempt + 1,
                "turbine_type": turbine_type,
                "hub_height": hub_height,
                "roughness_length": roughness_length,
                "farm_efficiency": farm_efficiency,
                "density_model": density_model,
                "wind_speed_model": wind_speed_model,
                "hellman_exp": hellman_exp,
                "smoothing": smoothing,
                "block_width": block_width,
                "standard_deviation_method": std_method,
                "smoothing_order": smoothing_order,
            }
            return power_hourly, meta

    # fallback (rare): all-zero output
    meta = {"nonzero": False, "attempt_used": cfg.wind_max_tries}
    return ZeroSeriesLike(met.index), meta


# Wind model hooks (implement with windpowerlib, etc.)
def SampleWindTurbine():
    """
    Return a turbine type and a hub height.

    Typical implementation:
      - load a turbine database (e.g., windpowerlib.data.get_turbine_types(filter_=True))
      - filter to turbines with power curves and available hub heights
      - sample turbine_type uniformly or with a custom prior
      - sample one hub height option for that turbine
    """
    pass


def SampleRoughnessLength():
    """
    Return a surface roughness length in meters.

    Typical implementation:
      roughness_level = [0.0002, 0.0024, 0.03, 0.055, 0.1, 0.2, 0.4, 0.6, 1.6]
      prob_rough = [0.95/6]*6 + [0.05/3]*3
      roughness_length = random.choices(roughness_level, weights=prob_rough, k=1)[0]
    """
    pass


def RunWindFarmModel(wind_weather, turbine_type, hub_height,
                     total_capacity_mw, farm_efficiency, modelchain_data):
    """
    Run a wind farm physical model and return MW series.

    Typical implementation with windpowerlib (conceptual):
      1) Build WindTurbine(turbine_type=..., hub_height=...)
      2) Build wind_turbine_fleet with total_capacity = total_capacity_mw * 1e6 (W)
      3) Build WindFarm(wind_turbine_fleet=...)
      4) Set farm.efficiency = farm_efficiency
      5) Convert wind_weather into windpowerlib MultiIndex format, including roughness_length
      6) Run TurbineClusterModelChain(farm, **modelchain_data).run_model(wind_weather)
      7) Return power_output / 1e6 as MW series
    """
    pass


def SolarMeteorologyToPower(met, latitude, longitude, capacity_mw, cfg):
    """
    PV physical model (pvlib-style).

    Inputs
    ------
    met: hourly table with columns:
         u10, v10, t2m, sp, tp, fdir
         - u10,v10: 10m wind components (m/s)
         - t2m: air temperature (K)
         - sp: surface pressure (Pa)
         - fdir: hourly accumulated radiation (J/m^2)
         - tp: precipitation (not required by default; can be used in advanced loss models)
    latitude, longitude: site coordinate
    capacity_mw: installed capacity (MW)

    Output
    ------
    power_hourly: time series (MW), indexed by met.index
    meta: dict of sampled PV configuration for reproducibility

    Model logic (high-level)
    ------------------------
    1) Derive meteorological drivers:
         wind_speed = sqrt(u10^2 + v10^2)
         temp_air_c = t2m - 273.15
         pressure_hpa = sp / 100

    2) Convert radiation:
         fdir is hourly accumulated energy J/m^2
         ghi = fdir / 3600  => mean irradiance W/m^2 over the hour (proxy)

    3) Solar position:
         solpos = SolarPosition(time, lat, lon, pressure_hpa, temp_air_c)
       provides zenith/azimuth and apparent zenith.

    4) Irradiance decomposition (GHI -> DNI and DHI):
       choose one model among:
         - disc
         - dirint
         - erbs
         - boland
       yielding:
         dni(t), dhi(t)

    5) Sample PV system configuration:
       (a) inverter and module from a public database (e.g., SAM CEC):
            inverter has rating (Paco), module has STC
       (b) choose DC/AC ratio proxy in [0.8, 0.9]
       (c) design a feasible string configuration:
            modules_per_string, strings_per_inverter
       (d) mounting:
            - fixed tilt: tilt ≈ |latitude|, azimuth depends on hemisphere
            - dual-axis tracker (optional)
       (e) loss / AOI / spectral models:
            losses_model ∈ {pvwatts, no_loss}
            aoi_model ∈ {physical, ashrae, martin_ruiz}
            spectral_model often set to "no_loss" unless correct precipitable water is provided

    6) Run PV model chain:
         ac_watt(t) = PVModelChain(weather, system_config)
       Scale to target installed capacity:
         scale_num = capacity_mw / base_capacity_mw_of_config
         power_mw(t) = ac_watt(t)/1e6 * scale_num
       Clip at 0.
    """
    lon_0_360 = NormalizeLonTo0_360(longitude)
    lon_m180_180 = Lon0_360_ToMinus180_180(lon_0_360)

    # 1) derived drivers
    wind_speed = sqrt(met["u10"]**2 + met["v10"]**2)
    temp_c = met["t2m"] - 273.15
    pressure_hpa = met["sp"] / 100.0

    # 2) radiation conversion (J/m^2 per hour -> W/m^2)
    ghi = met["fdir"] / 3600.0
    ghi = ClipLower(ghi, 0.0)

    # 3) solar position
    solpos = SolarPosition(
        time=met.index,
        lat=latitude,
        lon=lon_m180_180,
        pressure_hpa=pressure_hpa,
        temperature_c=temp_c
    )

    # 4) irradiance decomposition
    dec_model = RandChoice(["disc", "dirint", "erbs", "boland"])
    dni, dhi = DecomposeIrradiance(
        model=dec_model,
        ghi=ghi,
        solar_zenith=solpos.zenith,
        time=met.index,
        pressure_hpa=pressure_hpa
    )

    # 5) PV system sampling
    inverter = SampleInverter(min_ac_power_w=1e6)
    module = SampleModule()

    modules_coef = RandUniform(0.8, 0.9)
    modules_per_string, strings_per_inverter, total_modules_cfg = DesignPVStrings(inverter, module, modules_coef)

    mount = RandChoiceWithProb(
        options=[FixedTiltMount(latitude), DualAxisTrackerMount()],
        probs=[0.8, 0.2]
    )

    losses_model = RandChoice(["pvwatts", "no_loss"])
    aoi_model = RandChoice(["physical", "ashrae", "martin_ruiz"])
    spectral_model = "no_loss"

    base_capacity_mw = (total_modules_cfg * module.STC_watt) / 1e6
    scale_num = capacity_mw / max(base_capacity_mw, 1e-9)

    weather = {
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "temp_air": temp_c,
        "wind_speed": wind_speed
    }

    ac_watt = RunPVModelChain(
        weather=weather,
        lat=latitude, lon=lon_m180_180,
        inverter=inverter, module=module,
        mount=mount,
        modules_per_string=modules_per_string,
        strings_per_inverter=strings_per_inverter,
        losses_model=losses_model,
        aoi_model=aoi_model,
        spectral_model=spectral_model
    )

    power_hourly = (ac_watt / 1e6) * scale_num
    power_hourly = ClipLower(power_hourly, 0.0)

    meta = {
        "radiation_var": "fdir",
        "radiation_unit": "J/m^2 per hour",
        "ghi_convert": "ghi = fdir/3600",
        "decomposition_model": dec_model,
        "inverter": inverter.name,
        "module": module.name,
        "modules_coef": modules_coef,
        "modules_per_string": modules_per_string,
        "strings_per_inverter": strings_per_inverter,
        "total_modules_cfg": total_modules_cfg,
        "mount": mount.name,
        "losses_model": losses_model,
        "aoi_model": aoi_model,
        "spectral_model": spectral_model,
        "scale_num": scale_num
    }
    return power_hourly, meta


def Lon0_360_ToMinus180_180(lon_0_360):
    return lon_0_360 - 360 if lon_0_360 > 180 else lon_0_360


# Solar model hooks (implement with pvlib, etc.)
def SolarPosition(time, lat, lon, pressure_hpa, temperature_c):
    """
    Return solar position object providing at least:
      - zenith (or apparent_zenith)
      - azimuth (optional)
    Typical implementation: pvlib.solarposition.get_solarposition(...)
    """
    pass

def DecomposeIrradiance(model, ghi, solar_zenith, time, pressure_hpa):
    """
    Given GHI and solar zenith, compute DNI and DHI.

    Typical pvlib implementations:
      - disc
      - dirint
      - erbs
      - boland
    """
    pass

def SampleInverter(min_ac_power_w):
    """
    Sample an inverter from a database (e.g., SAM CEC inverter list),
    possibly filtering by Paco > min_ac_power_w.
    Must return an object with:
      - name
      - Paco (or equivalent rating)
    """
    pass

def SampleModule():
    """
    Sample a PV module from a database (e.g., SAM CEC module list).
    Must return an object with:
      - name
      - STC_watt
    """
    pass

def DesignPVStrings(inverter, module, modules_coef):
    """
    Design a feasible string configuration.

    Inputs:
      inverter: has AC rating
      module: has STC_watt
      modules_coef: DC/AC ratio proxy

    Output:
      modules_per_string (int)
      strings_per_inverter (int)
      total_modules_cfg (int)

    One practical approach:
      - compute target module count = inverter.Paco * modules_coef / module.STC
      - search mps in [10..40], spi >=1 to find configuration close to target
    """
    pass

def FixedTiltMount(latitude):
    """Return a mount object representing fixed tilt and azimuth based on hemisphere."""
    pass

def DualAxisTrackerMount():
    """Return a mount object representing dual-axis tracking."""
    pass

def RandChoiceWithProb(options, probs):
    """Random choice with explicit probabilities."""
    pass

def RunPVModelChain(weather, lat, lon, inverter, module, mount,
                    modules_per_string, strings_per_inverter,
                    losses_model, aoi_model, spectral_model):
    """
    Run pvlib ModelChain (or equivalent) and return AC power in Watts.
    """
    pass


# -------------------------------------------------------------------
# 3) Paired weather tensor: (hours, 64, 27)
# -------------------------------------------------------------------
def BuildPairedWeatherTensor(plant_no, lat, lon, time_hourly, cfg):
    """
    Build hourly paired weather aligned with `time_hourly`.

    Output:
      weather: float32 array of shape (H, 64, 27)
        H = len(time_hourly)

    Channel order (fixed):
      - HRES:  11 variables in cfg.hres_vars
      - Pangu:  8 variables in cfg.ai_vars
      - FengWu: 8 variables in cfg.ai_vars
      total: 27 channels
    """
    # Construct full-day blocks to respect cycle structure, then slice to requested hours.
    day0 = FloorDay(time_hourly[0])
    day1 = FloorDay(time_hourly[-1])
    days = DateRange(day0, day1, freq="1D")

    # Patch indices (8×8) are constant for the site and a given source grid.
    patch_hres = GetPatchIndices(lat, lon, cfg.patch_size, source="HRES")
    patch_pg   = GetPatchIndices(lat, lon, cfg.patch_size, source="PANGU")
    patch_fw   = GetPatchIndices(lat, lon, cfg.patch_size, source="FENGWU")

    blocks = []
    for d in days:
        hres = BuildHRESDay(d, cfg, patch_hres)      # [24,64,11]
        pg   = BuildPanguDay(d, cfg, patch_pg)       # [24,64,8]
        fw   = BuildFengWuDay(d, cfg, patch_fw)      # [24,64,8]
        blocks.append(ConcatOnChannel([hres, pg, fw]))  # [24,64,27]

    full = ConcatOnTime(blocks)  # [Ndays*24,64,27]
    full_index = DateRange(day0, day1 + Hours(23), freq="1h")

    pos = Indexer(full_index, time_hourly)
    return full[pos, :, :]


# -------------------------------------------------------------------
# 3.1) Spatial patch selection (8×8) and flattening (8×8 -> 64)
# -------------------------------------------------------------------
def GetPatchIndices(lat, lon, patch_size, source):
    """
    Select a patch_size×patch_size square on a regular lat/lon grid, centered "as well as possible"
    around the target coordinate (lat, lon). The output is a list of integer grid indices used to
    crop any gridded variable from the same source.

    Parameters
    ----------
    lat, lon : float
        Target coordinate of the site (in the longitude convention used by the source grid).
    patch_size : int
        Patch width/height. For this project patch_size=8, so 8×8=64 points.
    source : str
        Data source identifier, e.g. "HRES", "PANGU", "FENGWU".
        Different sources may use different grids (bounds/resolution/longitude convention).

    Returns
    -------
    patch_idx : list[tuple[int,int]] with length patch_size*patch_size
        Flattened list of (i_lat, j_lon) indices describing the selected square.
        The index ordering is row-major:
            (i0,j0),(i0,j1),...,(i0,j7),(i1,j0),...,(i7,j7)

    Determinism
    ----------
    This selection is deterministic for a fixed (lat, lon, patch_size, source_grid_meta).
    It can be cached per site/source and reused for all variables and all timestamps.

    Selection criterion (same as the provided GridFinder implementation)
    -------------------------------------------------------------------
    Among all possible patch_size×patch_size squares on the grid:
      1) take the four corner coordinates of the square,
      2) compute distances from target to each corner,
      3) compute the variance of these four distances,
      4) pick the square with the smallest variance.

    Intuition:
      Minimizing the variance of corner distances tends to place the target near the geometric
      center of the chosen square (balanced distances to corners), producing a stable local patch.
    """
    lat_range, lon_range, step = GetSourceGridMeta(source)
    finder = GridFinder(lat_range=lat_range, lon_range=lon_range, step=step)

    # GridFinder API expects a list of coordinates; one site => list of length 1.
    patch_idx = finder.find_optimal_square_grid(coord=[(lat, lon)], n=patch_size)[0]
    return patch_idx


def GetSourceGridMeta(source):
    """
    Return grid metadata for a given source.

    Output
    ------
    (lat_range, lon_range, step):
      lat_range: (lat_max, lat_min)  # latitude descending grid
      lon_range: (lon_min, lon_max)
      step: grid resolution in degrees

    Notes
    -----
    - lat_range and lon_range must match the arrays used to index the stored fields.
    - Longitude convention must be consistent:
        * [0, 360) degrees east  OR  [-180, 180)  (choose one per source)
    - For global 0.25° grids, a common choice is:
        lat_range = (90, -90), lon_range = (0, 360), step = 0.25
    """
    # Example placeholders:
    # if source in {"HRES","PANGU","FENGWU"}:
    #     return (90, -90), (0, 360), 0.25
    # else:
    #     raise ValueError("unknown source")
    pass


class GridFinder:
    """
    Patch selector on a regular lat/lon grid.

    Construction
    ------------
    Input:
      lat_range = (lat_max, lat_min)  # descending latitude grid
      lon_range = (lon_min, lon_max)
      step      = grid spacing in degrees

    Internal grid indexing
    ----------------------
    The grid is represented by:
      grid[i_lat, j_lon] = (lat_value, lon_value)

    Selection method
    ----------------
    find_optimal_square_grid(coord, n) searches all possible n×n squares and selects the one
    minimizing the variance of distances from the target coordinate to the four corners.

    Output format
    -------------
    For each target coordinate, returns a list of (i_lat, j_lon) indices of length n*n.
    """
    def __init__(self, lat_range, lon_range, step):
        # Build 2D coordinate grid:
        #   lats = [lat_max, lat_max-step, ..., lat_min]
        #   lons = [lon_min, lon_min+step, ..., lon_max]
        # Store grid_shape = (n_lat, n_lon)
        pass

    def find_optimal_square_grid(self, coord, n):
        """
        Parameters
        ----------
        coord : list[(lat,lon)]
            Target coordinates.
        n : int
            Patch size (n×n).

        Returns
        -------
        results : list[list[(i_lat,j_lon)]]
            One element per target coordinate.
        """
        # For each target:
        #   best_score = +inf
        #   best_square = None
        #   for i in range(n_lat - n + 1):
        #     for j in range(n_lon - n + 1):
        #       corners = (i,j), (i,j+n-1), (i+n-1,j), (i+n-1,j+n-1)
        #       distances = [dist(target, corner_k) for k in 1..4]
        #       score = var(distances)
        #       if score < best_score:
        #           best_score = score
        #           best_square = [(i+di, j+dj) for di in 0..n-1 for dj in 0..n-1]
        #   append best_square to results
        pass


def CropFlatten(grid2d, patch_idx):
    """
    Crop a patch from a 2D grid and flatten it to a 1D vector (length 64 for 8×8).

    Parameters
    ----------
    grid2d : array[lat,lon]
        A single-variable gridded field at one timestamp.
    patch_idx : list[(i_lat,j_lon)]
        Output of GetPatchIndices(...).

    Returns
    -------
    vec : array[patch_size*patch_size]
        Flattened patch, row-major order consistent with patch_idx.
    """
    # No need to pass patch_size: its implied by len(patch_idx) = patch_size^2
    vec = [grid2d[i, j] for (i, j) in patch_idx]
    return ToFloat32Array(vec)


def ToFloat32Array(vec):
    """
    Convert Python list-like to float32 array.

    Implementation note:
      In NumPy: np.asarray(vec, dtype=np.float32)
    """
    pass


# -------------------------------------------------------------------
# 3.2) HRES day block: [24,64,11]
# -------------------------------------------------------------------
def BuildHRESDay(day_utc, cfg, patch_idx):
    """
    Assemble 24 hourly HRES frames for one UTC day.

    For each cycle (00/06/12/18):
      - read steps 0..5
      - place them into hours [cycle..cycle+5]

    Then convert accumulated variables:
      SSR, TP are cumulative => instantaneous per hour via first difference.
    """
    out = zeros([24, cfg.patch_size*cfg.patch_size, len(cfg.hres_vars)], dtype="float32")

    for cyc in cfg.cycles_utc:
        base_time = day_utc + Hours(cyc)
        for step in range(cfg.steps_per_cycle):          # 0..5
            hour = cyc + step                            # 0..23
            for vidx, var in enumerate(cfg.hres_vars):
                grid = ReadHRESGrid(var, base_time, step)            # grid2d[lat,lon]
                out[hour, :, vidx] = CropFlatten(grid, patch_idx, cfg.patch_size)

    # accumulated -> instantaneous (SSR, TP)
    for var in cfg.hres_accum_vars:
        vidx = index_of(cfg.hres_vars, var)
        accum = out[:, :, vidx]                           # [24,64]
        inst = zeros_like(accum)
        inst[1:] = accum[1:] - accum[:-1]
        inst[0] = inst[1]                                 # boundary rule
        inst = clip_lower(inst, 0.0)                      # avoid negative artifacts
        out[:, :, vidx] = inst

    return out


# -------------------------------------------------------------------
# 3.3) Pangu day block: [24,64,8]  (vars = cfg.ai_vars)
# -------------------------------------------------------------------
def BuildPanguDay(day_utc, cfg, patch_idx):
    """
    Assemble 24 hourly Pangu-Weather frames for one UTC day.

    Reference implementation / model release
    ---------------------------------------
    Pangu-Weather (ONNX models and reference code):
      https://github.com/198808xc/Pangu-Weather/

    Output tensor
    -------------
    out: float32 array with shape [24, 64, 8]
      - time dimension: 24 hours of the UTC day (00..23)
      - spatial dimension: 8×8 patch flattened to 64
      - channel dimension (8): variables in cfg.ai_vars in fixed order:
            ["u10","v10","t2m","z1000","q1000","t1000","u1000","v1000"]

    Daily assembly rule (00/06/12/18 cycles)
    ----------------------------------------
    For each cycle time in {00,06,12,18} UTC:
      1) Initialize model state at base_time (lead=0)
         - base_time is a UTC timestamp: day_utc + cyc hours
         - initial state can be driven by analysis/ERA5/HRES fc0 (lead=0), depending on setup

      2) Produce 6 hourly frames for this cycle:
         lead=0 corresponds to valid time [cyc + 0]
         lead=1..5 correspond to valid times [cyc + 1 .. cyc + 5]

         To obtain hourly outputs, run the 1-hour model sequentially:
           state(t+1) = PanguStep1h(state(t))
           repeat 5 times.

      3) At each valid hour:
         - extract the 8 variables as 2D grids (lat×lon)
         - crop an 8×8 patch using patch_idx
         - flatten to length 64
         - write into out[valid_hour, :, :]

    Required hooks (environment-specific)
    ------------------------------------
    - PanguInitialState(base_time_utc) -> (pressure_state, surface_state)
        returns the model input state at lead=0 for that cycle.
    - PanguStep1h(pressure_state, surface_state) -> (next_pressure, next_surface)
        runs the 1-hour ONNX model once.
    - PanguExtractVars(pressure_state, surface_state) -> dict(name -> grid2d)
        returns 2D grids for each variable in cfg.ai_vars.

    """
    out = zeros([24, cfg.patch_size * cfg.patch_size, len(cfg.ai_vars)], dtype="float32")

    for cyc in cfg.cycles_utc:
        base_time = day_utc + Hours(cyc)

        # lead=0: initial state at the cycle time
        pressure_state, surface_state = PanguInitialState(base_time)

        # lead=0 extraction
        grids = PanguExtractVars(pressure_state, surface_state)  # dict ai_vars -> grid2d
        for k, name in enumerate(cfg.ai_vars):
            out[cyc + 0, :, k] = CropFlatten(grids[name], patch_idx)

        # lead=1..5: roll the 1-hour model sequentially
        for lead in range(1, 6):
            pressure_state, surface_state = PanguStep1h(pressure_state, surface_state)
            grids = PanguExtractVars(pressure_state, surface_state)
            for k, name in enumerate(cfg.ai_vars):
                out[cyc + lead, :, k] = CropFlatten(grids[name], patch_idx)

    return out


# -------------------------------------------------------------------
# 3.4) FengWu day block: [24,64,8]  (vars = cfg.ai_vars)
# -------------------------------------------------------------------
def BuildFengWuDay(day_utc, cfg, patch_idx):
    """
    Assemble 24 hourly FengWu frames for one UTC day.

    Reference implementation / model release
    ---------------------------------------
    FengWu (OpenEarthLab release):
      https://github.com/OpenEarthLab/FengWu

    Output tensor
    -------------
    out: float32 array with shape [24, 64, 8]
      - time dimension: 24 hours of the UTC day (00..23)
      - spatial dimension: 8×8 patch flattened to 64
      - channel dimension (8): variables in cfg.ai_vars in fixed order:
            ["u10","v10","t2m","z1000","q1000","t1000","u1000","v1000"]

    Daily assembly rule (00/06/12/18 cycles)
    ----------------------------------------
    FengWu inference is performed per 6-hour block, matching the same cycle structure:
      - lead=0 for a cycle uses the analysis (or fc0-equivalent) at the cycle time.
      - lead=1..5 are generated by rolling a 1-hour model sequentially.

    Normalization
    -------------
    FengWu models typically operate on normalized states:
        state_norm = (state_raw - mean) / std
        state_raw  = state_norm * std + mean
    The mean/std are per-channel statistics computed from training data and stored alongside the model.

    Required hooks (environment-specific)
    ------------------------------------
    - FengWuAnalysisState(ts_utc) -> raw_state[C,lat,lon]
        returns the unnormalized analysis state at ts_utc.
    - FengWuNormStats() -> (mean[C,1,1], std[C,1,1])
        returns normalization statistics.
    - FengWuStep1h(state_norm[1,C,lat,lon]) -> next_state_norm[1,C,lat,lon]
        runs the 1-hour FengWu model once (often needs to be trained by the user).
    - FengWuSelectVars(raw_state[C,lat,lon]) -> dict(name -> grid2d)
        maps the raw multichannel state to the 8 target variables in cfg.ai_vars
        and returns each as a 2D grid.

    """
    mean, std = FengWuNormStats()

    def norm(x):   return (x - mean) / std
    def denorm(x): return (x * std) + mean

    out = zeros([24, cfg.patch_size * cfg.patch_size, len(cfg.ai_vars)], dtype="float32")

    for cyc in cfg.cycles_utc:
        t0 = day_utc + Hours(cyc)

        # lead=0: analysis at cycle time
        s0_raw = FengWuAnalysisState(t0)          # raw [C,lat,lon]
        grids0 = FengWuSelectVars(s0_raw)         # dict ai_vars -> grid2d
        for k, name in enumerate(cfg.ai_vars):
            out[cyc + 0, :, k] = CropFlatten(grids0[name], patch_idx)

        # lead=1..5: roll 1-hour model sequentially
        s_prev_n = norm(s0_raw)[None, ...]        # [1,C,lat,lon]
        for lead in range(1, 6):
            s_next_n = FengWuStep1h(s_prev_n)     # [1,C,lat,lon] normalized
            s_next_raw = denorm(s_next_n[0])      # raw [C,lat,lon]
            grids = FengWuSelectVars(s_next_raw)
            for k, name in enumerate(cfg.ai_vars):
                out[cyc + lead, :, k] = CropFlatten(grids[name], patch_idx)
            s_prev_n = s_next_n

    return out


# -------------------------------------------------------------------
# 4) Helper stubs used above (explicitly documented)
# -------------------------------------------------------------------
def AssertHourlyEndpoints(start_utc, end_utc):
    """
    start_utc and end_utc must be UTC timestamps with minute=second=0.
    They are treated as inclusive endpoints.
    """
    pass

def SeedAll(seed):
    """
    Set random seeds for:
      - Python random
      - NumPy
      - any ML frameworks you use (optional)
    """
    pass

def NormalizeLonTo0_360(lon):
    """
    Convert any longitude convention to [0,360) degrees east for ERA5 grid matching.
    """
    pass

def ReadEra5Grid(var_name, ts_utc):
    """
    Return ERA5 grid for var_name at ts_utc.
    Must be consistent with cfg.lat_grid/cfg.lon_grid ordering.
    """
    pass

def BilinearInterpToPoint(grid2d, lat_grid, lon_grid, lat, lon_0_360):
    """
    Return the interpolated value at (lat, lon_0_360) using bilinear interpolation.
    """
    pass

def ReadHRESGrid(var_name, base_time_utc, step):
    """
    Return HRES grid valid at base_time_utc + step hours.
    """
    pass

def PanguInitialState(base_time_utc):
    """
    Create/Load Pangu initial state at base_time_utc (lead=0).
    """
    pass

def PanguStep1h(pressure_state, surface_state):
    """
    Run Pangu 1-hour ONNX model once.
    """
    pass

def PanguExtractVars(pressure_state, surface_state):
    """
    Return dict of 8 grids with keys cfg.ai_vars.
    """
    pass

def FengWuAnalysisState(ts_utc):
    """
    Return FengWu raw analysis state [C,lat,lon] at ts_utc.
    """
    pass

def FengWuNormStats():
    """
    Return mean/std for normalization:
      mean[C,1,1], std[C,1,1]
    """
    pass

def FengWuStep1h(state_norm_bchw):
    """
    Run FengWu 1-hour ONNX model once.
    """
    pass

def FengWuSelectVars(raw_state_chw):
    """
    Return dict of 8 grids with keys cfg.ai_vars.
    """
    pass

def SaveCSV(path, columns, rows):
    """
    Write a CSV file to `path`.
    - Ensure parent directories exist.
    - Use UTF-8 and newline-safe writer.
    """
    pass

def SaveNPY(path, array):
    """
    Save NumPy array to .npy (float32).
    """
    pass

def AppendInfoRow(info_csv_path, PLANT_NO, LONGITUDE, LATITUDE, CONFIRM_POWER_MW,
                  MODE, START_UTC, END_UTC, SEED, power_meta):
    """
    Append one row to info.csv.
    If file does not exist, create it with header.

    POWER_META_JSON is recommended to store the power model sampling choices compactly.
    """
    pass

def PathJoin(*parts):
    """
    OS-agnostic path join.
    """
    pass

def FloorDay(ts):
    """
    Return UTC day boundary timestamp for ts (00:00:00).
    """
    pass

def Hours(h):
    """
    Return a time delta of h hours.
    """
    pass

def DateRange(start, end, freq):
    """
    Inclusive date range generator.

    For hourly: includes both endpoints if they fall on the grid.
    For daily:  includes both day endpoints.
    """
    pass

def UnionIndex(a, b):
    """
    Return sorted union of two time indices.
    """
    pass

def InterpolateTime(series):
    """
    Time-based interpolation (linear in time).
    """
    pass

def TableFromDict(d, index):
    """
    Create a table (DataFrame-like):
      rows aligned to `index`,
      columns from dict keys.
    """
    pass

def SqueezeTo2D(arr):
    """
    If arr is [1,lat,lon], return arr[0].
    If arr is [lat,lon], return arr.
    """
    pass

def Indexer(full_index, target_index):
    """
    Return integer positions mapping target_index into full_index (exact match).
    """
    pass

def ConcatOnChannel(arrays):
    """
    Concatenate arrays along the last dimension (channel dimension).
    """
    pass

def ConcatOnTime(arrays):
    """
    Concatenate arrays along the first dimension (time dimension).
    """
    pass

def ToFloat32Array(vec):
    """
    Convert list-like into float32 numpy array.
    """
    pass

def zeros(shape, dtype): pass
def zeros_like(x): pass
def clip_lower(x, lo): pass
def index_of(lst, item): pass
def ndim(arr): pass