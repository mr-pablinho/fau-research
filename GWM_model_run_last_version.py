
# %% Import libraries
import os
import math
import flopy
import pandas as pd
import geopandas as gpd
import numpy as np
import flopy.utils.binaryfile as bf
from flopy.utils import HeadFile
import matplotlib.pyplot as plt
from shapely.geometry import Point

def GWM(hk1, hk2, hk3, hk4, hk5, sy1, sy2, sy3, sy4, sy5, D_Isar, Kriv_Isar, Kriv_Muhlbach, Kriv_Giessen, Kriv_Griesbach, Kriv_Schwabinger_Bach, Kriv_Wiesackerbach, D_rch1, D_rch2, custom_out_dir=None):


    """ #%% Name and directories of the model """
    modelname       = "Garching_model"
    start_date      = '28/04/2024' # warm up + calibration period

    in_dir          = os.path.join('Output1_Input2')
    
    if custom_out_dir is not None:
        out_dir = custom_out_dir
    else:
        out_dir = os.path.join('Output2')

    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)


    """ #%% Model parameters """
    # UPW
    hk1             = hk1  # hydraulic conductivity (m/d)
    hk2             = hk2  
    hk3             = hk3 
    hk4             = hk4  
    hk5             = hk5 
    sy1             = sy1 # specific yield (-)
    sy2             = sy2 
    sy3             = sy3
    sy4             = sy4 
    sy5             = sy5

    # RIV
    D_Isar = D_Isar # Stage 
    
    Kriv_Isar = Kriv_Isar # river bed conductance (m2/d)
    Kriv_Muhlbach = Kriv_Muhlbach
    Kriv_Giessen = Kriv_Giessen
    Kriv_Griesbach = Kriv_Griesbach
    Kriv_Schwabinger_Bach = Kriv_Schwabinger_Bach
    Kriv_Wiesackerbach = Kriv_Wiesackerbach
    Kriv_Schleissheimer_Kanal = 0
    
    # RCH
    D_rch1 = D_rch1
    D_rch2 = D_rch2

    """ #%% DIS discretization """    
    shape_name      = 'Area_GWM_Garching.shp' # import polygon shapefile of the area
    grid_size       = 50 # 50 m
    nlay            = 1 

    model           = flopy.modflow.Modflow(modelname,version="mfnwt", exe_name="MODFLOW-NWT_64", model_ws=out_dir)
    domain_shpname  = os.path.join(in_dir,shape_name)
    gdf             = gpd.read_file(domain_shpname)
    bounds          = gdf.bounds
    x_min, y_min, x_max, y_max = bounds.minx[0], bounds.miny[0], bounds.maxx[0], bounds.maxy[0]
    domain_polygon  = gdf.geometry[0] 

    # Add Spatial discretization
    x_min           = math.floor(x_min / grid_size) * grid_size
    y_min           = math.floor(y_min / grid_size) * grid_size
    x_max           = math.ceil(x_max / grid_size) * grid_size + 100 # to have the grid correctly placed
    y_max           = math.ceil(y_max / grid_size) * grid_size
    Lx              = x_max - x_min
    Ly              = y_max - y_min 
    nrow            = int(Ly/grid_size)
    ncol            = int(Lx/grid_size)
    delr            = Lx / ncol
    delc            = Ly / nrow

    # Add temporal discretization
    nper         = 139 # number of stress periods (0-365). must be an integer.
    perlen       = np.ones(nper, dtype=int) # length of stress period in days. integer or array or list. 
    timestep     = 1
    itmuni       = 4 # ime units for the simulation (4 = daily)
    nstp         = timestep*np.ones(nper, dtype=int) # number of timesteps in a stress period (set to 1). integer or array or list. 
    tsmult       = 1 # "time step multiplier." It is a parameter used to control the progression of time steps within a stress period
    steady_array = np.full(nper, False)
    steady_array[0] = True #  NumPy array called steady_array with a length equal to nper (the number of stress periods). False indicates transient conditions, True steady state
        

    dis  = flopy.modflow.ModflowDis(model, nlay, nrow, ncol, delr=delr, delc=delc, 
                                    nper = nper, perlen=perlen,tsmult = tsmult,itmuni=itmuni, nstp=nstp,steady=steady_array,
                                    xul=x_min, yul=y_max,crs = 25832 ,start_datetime=start_date)

    # Import top and bot
    grid_top =  np.loadtxt(os.path.join(in_dir, 'Cell_Top_ly1.csv'), delimiter=',')
    grid_botm =  np.loadtxt(os.path.join(in_dir, 'Cell_Bottom_ly1.csv'), delimiter=',')

    # Assign top and bot to DIS
    dis.top  = grid_top + 5
    dis.botm = grid_botm 

    """ #%% UPW """
    # UPW Package
    grid_soil_cell_values_1  = pd.read_csv(os.path.join(in_dir,'UPW_cellid_1.csv'))
    grid_soil_cell_values_1  = grid_soil_cell_values_1.to_numpy()

    grid_soil_cell_values_2  = pd.read_csv(os.path.join(in_dir,'UPW_cellid_2.csv'))
    grid_soil_cell_values_2  = grid_soil_cell_values_2.to_numpy()

    grid_soil_cell_values_3  = pd.read_csv(os.path.join(in_dir,'UPW_cellid_3.csv'))
    grid_soil_cell_values_3  = grid_soil_cell_values_3.to_numpy()

    grid_soil_cell_values_4  = pd.read_csv(os.path.join(in_dir,'UPW_cellid_4.csv'))
    grid_soil_cell_values_4  = grid_soil_cell_values_4.to_numpy()

    laytyp = np.ones(nlay) # Array to store layer type values. # 0 = confined. 1 = convertible. -1 = Special. Explaination in documentation

    hk = np.ones((nlay, nrow, ncol))  # m/day 
    hk[:, :, :] = hk5 

    for layer in range(hk.shape[0]):                              # loop through model layers
        for index in range(0,len(grid_soil_cell_values_1[:,0])-9):  # change values of cells indicated in array 'grid_cell_values' to hk1
            row=grid_soil_cell_values_1[index,0]
            col=grid_soil_cell_values_1[index,1]
            hk[layer][row,col]=hk1
        for index in range(0,len(grid_soil_cell_values_2[:,0])-1):  # change values of cells indicated in array 'grid_cell_values' to hk2
            row=grid_soil_cell_values_2[index,0]
            col=grid_soil_cell_values_2[index,1]
            hk[layer][row,col]=hk2
        for index in range(0,len(grid_soil_cell_values_3[:,0])-1):  # change values of cells indicated in array 'grid_cell_values' to hk3
            row=grid_soil_cell_values_3[index,0]
            col=grid_soil_cell_values_3[index,1]
            hk[layer][row,col]=hk3
        for index in range(0,len(grid_soil_cell_values_4[:,0])-1):  # change values of cells indicated in array 'grid_cell_values' to hk4
            row=grid_soil_cell_values_4[index,0]
            col=grid_soil_cell_values_4[index,1]
            hk[layer][row,col]=hk4

    sy = np.zeros((nlay, nrow, ncol)) 
    sy[:, :, :] = sy5

    for layer in range(sy.shape[0]):                              # loop through model layers
        for index in range(0,len(grid_soil_cell_values_1[:,0])-9):  # change values of cells indicated in array 'grid_cell_values' to sy1
            row=grid_soil_cell_values_1[index,0]
            col=grid_soil_cell_values_1[index,1]
            sy[layer][row,col]=sy1
        for index in range(0,len(grid_soil_cell_values_2[:,0])-1):  # change values of cells indicated in array 'grid_cell_values' to sy2
            row=grid_soil_cell_values_2[index,0]
            col=grid_soil_cell_values_2[index,1]
            sy[layer][row,col]=sy2
        for index in range(0,len(grid_soil_cell_values_3[:,0])-1):  # change values of cells indicated in array 'grid_cell_values' to sy3
            row=grid_soil_cell_values_3[index,0]
            col=grid_soil_cell_values_3[index,1]
            sy[layer][row,col]=sy3
        for index in range(0,len(grid_soil_cell_values_4[:,0])-1):  # change values of cells indicated in array 'grid_cell_values' to sy4
            row=grid_soil_cell_values_4[index,0]
            col=grid_soil_cell_values_4[index,1]
            sy[layer][row,col]=sy4
        
    upw = flopy.modflow.ModflowUpw(model, hk=hk, vka=hk, sy=sy,laytyp=laytyp)    

    
    """ #%% BAS6 """
    # %% Create the BAS6 package object
    ### Load previously created ibound (containing active cells, and inactive rigth side of teh Isar)
    ibound=pd.read_csv(os.path.join(in_dir, 'ibounds.csv'),header=None)
    ibound=ibound.to_numpy('float32') 

    # Set the CHD cells in the ibound array = -1 for both Upper and Downstream Boundary conditions
    chd_cellid_UB = pd.read_csv(os.path.join(in_dir, 'UB_cellid.csv'))
    chd_cellid_DB = pd.read_csv(os.path.join(in_dir, 'DB_cellid.csv'))

    # Process the Upper Boundary CHD cells
    chd_ly_UB = chd_cellid_UB['Layer'] - 1
    chd_R_UB = chd_cellid_UB['Row'] - 1
    chd_C_UB = chd_cellid_UB['Column'] - 1

    # Process the Downstream Boundary CHD cells
    chd_ly_DB = chd_cellid_DB['Layer'] - 1
    chd_R_DB = chd_cellid_DB['Row'] - 1
    chd_C_DB = chd_cellid_DB['Column'] - 1

    ibound[chd_R_UB, chd_C_UB] = -1
    ibound[chd_R_DB, chd_C_DB] = -1

    ### Initialize the starting head array
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    from_output_file = False
    if from_output_file:
        inhead =  475 #dis.top  #pd.read_csv(os.path.join(in_dir, 'Initial_head.csv'))
        strt[:] = inhead
    else:
        if grid_top.shape == (nrow, ncol):
            for ilay in range(nlay):
                strt[ilay, :, :] = grid_top


    ### Create the BAS6 package object
    bas = flopy.modflow.ModflowBas(model, ibound=ibound, strt=strt, hnoflo=-1e30, stoper=10)

    """ #%% RCH Recharge package """
    # %% Create the Recharge package
    # Read recharge time series (estomated with WTF) and cell locations corresponding to urban area
    pp_data    = pd.read_csv(os.path.join(in_dir,'RCH_data.csv')) 
    rch_cellid = pd.read_csv(os.path.join(in_dir,'RCH_cellid.csv'))

    # Prepare indices for RCH cells (0-based)
    rows = np.array(rch_cellid['Row'] - 1)
    cols = np.array(rch_cellid['Column'] - 1)

    # Initialize recharge dictionary
    rch_data = {}

    # Loop through each stress period
    for index, row in pp_data.iterrows():
        SP = row.iloc[0]               # Stress period
        base_val = row.iloc[1]         # Base recharge rate applied to the entire grid

        # Apply D_rch1 to the entire grid
        rch_array = np.full((nrow, ncol), base_val * D_rch1)

        # Override RCH cell values with D_rch2-scaled recharge (urban area)
        rch_array[rows, cols] = base_val * D_rch2

        # Store in dictionary
        rch_data[SP] = rch_array

    # Create the Recharge package
    rch = flopy.modflow.ModflowRch(model, nrchop=3, rech=rch_data)
 
    
    """ #%% CHD Constant Head Boundary """
    # Read the CSV files for CHB cell IDs
    chb_cellid_UB = pd.read_csv(os.path.join(in_dir, 'UB_cellid.csv'))
    chb_cellid_DB = pd.read_csv(os.path.join(in_dir, 'DB_cellid.csv'))

    # Adjust the cell IDs to zero-based indexing for Upper Boundary
    chb_cellid_UB['Layer'] = chb_cellid_UB['Layer'] - 1
    chb_cellid_UB['Row'] = chb_cellid_UB['Row'] - 1
    chb_cellid_UB['Column'] = chb_cellid_UB['Column'] - 1

    # Adjust the cell IDs to zero-based indexing for Downstream Boundary
    chb_cellid_DB['Layer'] = chb_cellid_DB['Layer'] - 1
    chb_cellid_DB['Row'] = chb_cellid_DB['Row'] - 1
    chb_cellid_DB['Column'] = chb_cellid_DB['Column'] - 1

    # Read the GHB data
    chb_data_UB = pd.read_csv(os.path.join(in_dir, 'CHB_data_UB.csv'))
    chb_data_DB = pd.read_csv(os.path.join(in_dir, 'CHB_data_DB.csv'))

    chb_dict = {}

    # Create the stress period data dictionary for Upper Boundary
    for index, row in chb_data_UB.iterrows():
        SP, shead, ehead = row.iloc[0], float(row.iloc[3]), float(row.iloc[4])
        chb_sp = chb_cellid_UB[['Layer', 'Row', 'Column']].copy()
        chb_sp['shead'] = np.ones(chb_cellid_UB.shape[0]) * shead
        chb_sp['ehead'] = np.ones(chb_cellid_UB.shape[0]) * ehead
        if SP not in chb_dict:
            chb_dict[SP] = []
        chb_dict[SP].extend(chb_sp.values.tolist())

    # Create the stress period data dictionary for Downstream Boundary
    for index, row in chb_data_DB.iterrows():
        SP, shead, ehead = row.iloc[0], float(row.iloc[3]), float(row.iloc[4])
        chb_sp = chb_cellid_DB[['Layer', 'Row', 'Column']].copy()
        chb_sp['shead'] = np.ones(chb_cellid_DB.shape[0]) * shead
        chb_sp['ehead'] = np.ones(chb_cellid_DB.shape[0]) * ehead
        if SP not in chb_dict:
            chb_dict[SP] = []
        chb_dict[SP].extend(chb_sp.values.tolist())

    # Add the Costant Head Boundary (CHB) package to the model
    chb = flopy.modflow.ModflowChd(model, stress_period_data=chb_dict)


    """ #%% RIV River package """
    # Load the combined RIV data 
    riv_data_Isar = pd.read_csv(os.path.join(in_dir, 'RIV_data_Isar.csv'))
    riv_data_Muhlbach = pd.read_csv(os.path.join(in_dir, 'RIV_data_Muhlbach.csv'))
    riv_data_Giessen = pd.read_csv(os.path.join(in_dir, 'RIV_data_Giessen.csv'))
    riv_data_Griesbach = pd.read_csv(os.path.join(in_dir, 'RIV_data_Griesbach.csv'))
    riv_data_Schleissheimer_Kanal = pd.read_csv(os.path.join(in_dir, 'RIV_data_Schleissheimer_Kanal.csv'))
    riv_data_Schwabinger_Bach = pd.read_csv(os.path.join(in_dir, 'RIV_data_Schwabinger_Bach.csv'))
    riv_data_Wiesackerbach = pd.read_csv(os.path.join(in_dir, 'RIV_data_Wiesackerbach.csv'))
    
    # %% Create the River package using multiple river datasets with per-river conductance and D_riv

    # --- Define conductance replacement dictionaries per river
    riv_par_dict = {
        'Isar': {'Kriv1': Kriv_Isar},
        'Muhlbach': {'Kriv1': Kriv_Muhlbach},
        'Giessen': {'Kriv1': Kriv_Giessen},
        'Griesbach': {'Kriv1': Kriv_Griesbach},
        'Schleissheimer_Kanal': {'Kriv1': Kriv_Schleissheimer_Kanal},  # Custom value
        'Schwabinger_Bach': {'Kriv1':  Kriv_Schwabinger_Bach},
        'Wiesackerbach': {'Kriv1': Kriv_Wiesackerbach}
    }

    # --- Define vertical shift (D_riv) per river
    D_riv_dict = {
        'Isar': D_Isar,
        'Muhlbach': 0.0,
        'Giessen': 0.0,
        'Griesbach': 0.0,
        'Schleissheimer_Kanal': 0,  
        'Schwabinger_Bach': 0.0,
        'Wiesackerbach': 0.0
    }

    # --- Load individual RIV data sources (already read into memory)
    riv_sources = {
        'Isar': riv_data_Isar,
        'Muhlbach': riv_data_Muhlbach,
        'Giessen': riv_data_Giessen,
        'Griesbach': riv_data_Griesbach,
        'Schleissheimer_Kanal': riv_data_Schleissheimer_Kanal,
        'Schwabinger_Bach': riv_data_Schwabinger_Bach,
        'Wiesackerbach': riv_data_Wiesackerbach
    }

    # --- Process and merge all river datasets
    processed_riv_data = []

    for river_name, df in riv_sources.items():
        df = df.copy()

        # --- Apply river-specific vertical shift
        D_riv_val = D_riv_dict[river_name]
        df['Stage'] = df['Stage'] + D_riv_val
        df['Rbott'] = df['Rbott'] + D_riv_val

        # --- Replace symbolic conductance with numeric value
        df['Cond'] = df['Cond'].replace(riv_par_dict[river_name])
        df['Cond'] = df['Cond'].astype(float)

        # --- Convert to zero-based indexing for MODFLOW
        df['Layer']  = df['Layer'] - 1
        df['Row']    = df['Row'] - 1
        df['Column'] = df['Column'] - 1

        processed_riv_data.append(df)

    # --- Combine all processed river data
    riv_data = pd.concat(processed_riv_data, ignore_index=True)

    # --- Build stress period dictionary for Flopy
    riv_dict = {}
    for sp in riv_data['SP'].unique():
        sp_data = riv_data[riv_data['SP'] == sp][['Layer', 'Row', 'Column', 'Stage', 'Cond', 'Rbott']]
        riv_dict[int(sp)] = sp_data.values.tolist()

    # --- Create the RIV package in the MODFLOW model
    riv = flopy.modflow.ModflowRiv(model, stress_period_data=riv_dict, ipakcb=1)

    """ ##% OC output control package """
    oc_sp_data = {}
    for kper in range(nper):
        oc_sp_data[(kper, nstp[kper]-1)] = ["save head","save drawdown","save budget"]
    oc = flopy.modflow.ModflowOc(model, stress_period_data=oc_sp_data, compact=True)
    
    """ #%% Observation wells with HOB package """
    obs_df = pd.read_csv(os.path.join(in_dir, 'obs.csv'))
    obsvals_df = pd.read_csv(os.path.join(in_dir, 'obs_values.csv'))

    layervals = (obs_df['Layer'] - 1).tolist()
    rowvals = (obs_df['Row'] - 1).tolist()
    colvals = (obs_df['Column'] - 1).tolist()

    obsvals = obsvals_df.values.tolist()  # each row = time step, each column = obs point

    obs_data = []
    for i in range(len(layervals)):
        time_series_data = [[j, obsvals[j][i]] for j in range(len(obsvals))]  # [[stress_period, value], ...]
        obs = flopy.modflow.HeadObservation(model, layer=layervals[i], row=rowvals[i], column=colvals[i], time_series_data=time_series_data)
        obs_data.append(obs)

    hob = flopy.modflow.ModflowHob(model, iuhobsv=7, hobdry=-999, obs_data=obs_data)

    """ #%% WEL - Extraction and Injection Wells (balanced by stress period) """
    # Manually define extraction wells: [(layer, row, col), ...]
    extraction_wells = [
        (0, 55, 128) # Layer 1-1, Row 56-1, Column 129-1
        
    ]

    # Define constant extraction rate (negative for pumping)
    extraction_rate = -25920 # m3/day = 0.3 m3/s = 300 l/s

    # Build stress period data
    wel_spd = {}
    for sp in range(nper):
        wel_data = []
        for layer, row, col in extraction_wells:
            wel_data.append([layer, row, col, extraction_rate])
        wel_spd[sp] = wel_data

    # Create the WEL package
    wel = flopy.modflow.ModflowWel(model, stress_period_data=wel_spd)

    """ Numerical solver """
    n   = flopy.modflow.ModflowNwt(model, maxiterout=500, headtol=1e-3, fluxtol=1e-0,iprnwt=1,options="COMPLEX",linmeth=2)

    """ Generate input files readable by modflow """

    model.write_input()  
    
    """ Run modflow """

    model.run_model()
    
    """ # Cleanup: delete all non-output files after model run
    # You can customize which files you consider "output" vs "temporary input" """
    keep_exts = ['.hds', '.cbc', '.hob.out', '.list']  
    for f in os.listdir(out_dir):
        full_path = os.path.join(out_dir, f)
        if os.path.isfile(full_path) and not any(f.endswith(ext) for ext in keep_exts):
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"⚠️ Could not delete {f}: {e}")

    return model, out_dir  # model and workspace



def get_head_value_at_point(model_ws, point, timestep):
    """
    Extract head value at a given model point and time step.

    Parameters:
    - model_ws: Path to the model workspace (where MODFLOW ran)
    - point: A tuple (layer, row, col)
    - timestep: Integer time step index (0-based; 0 = first, -1 = last)

    Returns:
    - Head value at the specified location and time
    """
    try:
        hds_path = os.path.join(model_ws, 'model.hds')  # Change if your file is named differently
        hds = flopy.utils.HeadFile(hds_path)

        head_array = hds.get_data(kstpkper=timestep)
        return head_array[point]  # point is (layer, row, col)
    
    except Exception as e:
        print(f"Failed to get head at {point}, timestep {timestep}: {e}")
        return np.nan
        
def get_heads_from_obs_csv(model_ws, obs_csv_path='Output1_Input2/obs.csv'):
    """
    Extracts simulated head values for all observation points and stress periods.

    Parameters:
    - model_ws: Path to the model workspace (where MODFLOW ran)
    - obs_csv_path: Full path to obs.csv file

    Returns:
    - sim_heads: 2D array [n_stress_periods x n_obs_points]
    """
    obs_df = pd.read_csv(obs_csv_path)
    layervals = (obs_df['Layer'] - 1).tolist()
    rowvals = (obs_df['Row'] - 1).tolist()
    colvals = (obs_df['Column'] - 1).tolist()

    hds_path = os.path.join(model_ws, 'Garching_model.hds')
    hds = flopy.utils.HeadFile(hds_path)

    available_kstpkper = hds.get_kstpkper()
    print(f"\n? Available head time steps in output: {available_kstpkper}")

    sim_heads = []
    for kstpkper in available_kstpkper:
        head_array = hds.get_data(kstpkper=kstpkper)
        sp_heads = [head_array[lay, row, col] for lay, row, col in zip(layervals, rowvals, colvals)]
        sim_heads.append(sp_heads)

    return np.array(sim_heads)
