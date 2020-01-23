### Model Evaluation: NAM-CMAQ vs. FV3GFS-CMAQ
### NCEP Internship - Summer 2019
### Author: Benjamin Yang

############################################
##########     IMPORT MODULES     ##########
############################################

import os
import numpy as np
import pandas as pd
import datetime as dts
from datetime import date, time, timedelta
from netCDF4 import Dataset
#import met_functions as met
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from MesoPy import Meso
from matplotlib import dates
import re

######################################################
##########     MODIFY FIGURE PROPERTIES     ##########
######################################################

# Modify default figure properties to make them more legible
# https://matplotlib.org/users/customizing.html
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.markersize'] = 4
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = 'bold'      
plt.rcParams.update({'mathtext.default': 'regular' })
plt.rcParams['axes.xmargin'] = 0
plt.close('all') # close all figures 

#############################################
##########     DEFINE FUNCTIONS    ##########
#############################################

# Get all dates between start and end dates inclusive
def perdelta(start, end, delta):
    curr = start
    while curr <= end:
        yield curr
        curr += delta

# Read latitude and longitude from file into numpy arrays
def naive_fast(latvar,lonvar,lat0,lon0):
    latvals = latvar[:]
    lonvals = lonvar[:]
    ny,nx = latvals.shape
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return int(iy_min),int(ix_min)

# Normalized Mean Bias (NMB)
def nmb(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']=df_new[name_var1]-df_new[name_var2]
    NMB=round((df_new['dif_var'].sum()/df_new[name_var2].sum())*100)
    return NMB

# Normalized Mean Error (NME)
def nme(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']= abs(df_new[name_var1]-df_new[name_var2])
    NME=round((df_new['dif_var'].sum()/df_new[name_var2].sum())*100)
    return NME

# Root Mean Squared Error (RMSE)
def rmse(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']= (df_new[name_var1]-df_new[name_var2])**(2)
    RMSE=round((df_new['dif_var'].sum()/len(df_new.index))**(0.5))
    return RMSE

# Coefficient of Determination (r^2)
def r2(df,name_var1,name_var2):
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    top_var= ((df_new[name_var1]-np.mean(df_new[name_var1])) * (df_new[name_var2]-np.mean(df_new[name_var2]))).sum()
    bot_var= (((df_new[name_var1]-np.mean(df_new[name_var1]))**2).sum() * ((df_new[name_var2]-np.mean(df_new[name_var2]))**2).sum())**(.5)
    r_squared=round(((top_var/bot_var)**2),2)
    return r_squared

# Combine statistics into a labeled dataframe
def stats(df,name_var1,name_var2):
    NMB = nmb(df,name_var1,name_var2)
    NME = nme(df,name_var1,name_var2)
    RMSE = rmse(df,name_var1,name_var2)
    r_squared = r2(df,name_var1,name_var2)
    g = pd.DataFrame([NMB,NME,RMSE,r_squared])
    g.index = ['NMB', 'NME', 'RMSE', 'R^2']
    g.columns = [name_var1]
    return g

######################################################
##########     SPECIFY DATA OF INTEREST    ###########
######################################################

# Set base directory containing python scripts, data, and output
base_dir = '/Users/Ben/Documents/NCEP/'

# Enter start/end dates & model information
start = date(2019,6,10) # June 1, 2019 (2019,6,1)
end = date(2019,6,10) # June 28, 2019 (2019,6,28)
models = ['NAM','FV3'] # NAM-CMAQ and FV3GFS-CMAQ (NAM and FV3)
# model variables (pm25, temp, rh, ws, wd, pblh)
#mod_var = ['pm25','temp','rh','ws','wd','pblh'] 
run = '12' # model run time (00z, 06z, 12z, or 18z)
daily = 'max' # PM2.5 daily 1-hr maximum ('max') or 24-hr average ('ave') 
hr_start = 1 # starting forecast hour (1-48)
hr_end = 48 # ending forecast hour (1-48)

# Enter mesonet station variables names, network name, and bounding box limits 
# https://developers.synopticdata.com/mesonet/v2/api-variables/
variables = 'air_temp,relative_humidity,wind_speed,wind_direction'
var_list = variables.split(',')
var_all = ('pm25,pblh,'+variables).split(',')
network = 1 # NWS/FAA network
bbox = [-131.77422,21.081833,-58.66989,53.096016] # CONUS domain

# Calculate and save all-inclusive dates in a list
alldates=[]
for result in perdelta(start, end, timedelta(days=1)):
    alldates.append(re.sub('-','',str(result)))

# Get list of forecast hours for a given run
#hours = np.arange(1,hr_out+1,1)

####################################################
##########     MODEL GRID INFORMATION    ###########
####################################################

# Obtain longitude/latitude data for dot points
grddot2d = Dataset(base_dir + 'Data/grddot2d.nc','r')
lon_dot = grddot2d.variables['LON'][:]
lat_dot = grddot2d.variables['LAT'][:]                            
grddot2d.close()

# Obtain longitude/latitude data for cross points
grdcro2d = Dataset(base_dir + 'Data/grdcro2d.nc','r')
lon_cro = grdcro2d.variables['LON'][:]
lat_cro = grdcro2d.variables['LAT'][:]
grdcro2d.close()

#######################################################
##########     AIRNOW PM2.5 OBSERVATIONS     ##########
#######################################################

# Iterate over files in directory 
d_str1 = base_dir+'Data/Airnow/201906_T13_T12/'
airnow_obs = [f for f in os.listdir(d_str1) if f.endswith('.csv')] 

# Load data from CSV files
airnow = []
colnames=['LAT', 'LON', 'DATETIME', 'PARAM', 'PM2.5', 'UNITS']
usecol = ['LAT','LON','DATETIME','PM2.5'] 
for a in airnow_obs:
    data = pd.read_csv(d_str1+a,names=colnames,usecols=usecol,header=None)
    data[data['PM2.5']<0]=np.nan # ignore negative PM2.5 values
    data['DATETIME'] = pd.to_datetime(data['DATETIME'],format='%Y-%m-%dT%H:%M') # convert dates into datetime format
    airnow.append(data)
airnow = pd.concat(airnow, ignore_index=True) # combine list of dataframes into one

###############################################
##########     PBLH OBSERVATIONS     ##########
###############################################

# Iterate over files in directory 
d_str2 = base_dir+'Data/PBLH_OBS/'
pblh_dat = [f for f in os.listdir(d_str2) if f.endswith('.txt')] 

# Regular expression for finding ID, datetime, latitude, longitude, and PBLH   
obs_re = r'(?P<ID>C\d{3})\s+(?P<DATETIME>\d{6}\/\d{4})\s+(?P<LAT>\d{2}\.\d{2})\s+(?P<LON>-?\d{2,3}\.\d{2})\s+(?P<PBLH>\d{,4}\.\d{2})'    

# Load data from text files 
pblh_df = []
for p in pblh_dat:
    with open(d_str2+p, 'r') as pf: # open file
        lines = pf.readlines()
    for l in lines: # search line by line for matches
        obs_match = re.search(obs_re,l) 
        if obs_match: # store matched data in dictionary and add to list
            match = obs_match.groupdict()
            pblh_df.append(match)
pblh_df = pd.DataFrame(pblh_df) # convert list to data frame
pblh_df['DATETIME'] = pd.to_datetime(pblh_df['DATETIME'],format='%y%m%d/%H%M') # convert dates into datetime format
pblh_df.PBLH = pblh_df.PBLH.astype(float) # convert PBLH values from string to float

##################################################
##########     MESONET OBSERVATIONS     ##########
##################################################

'''
# Enter Mesonet API token (generate using API key)
m = Meso(token = '12bf4ba1ccbb4c79a57a02f36b8ec11f')

# Enter start and end datetime ('%Y%m%d%H%M')
starting = '201906120000' # 201906100000
ending = '201906122359'# 201906102359

# Enter mesonet station variables names, network name, and bounding box limits 
# https://developers.synopticdata.com/mesonet/v2/api-variables/
variables = 'air_temp,relative_humidity,wind_speed,wind_direction'
var_list = variables.split(',')
network = 1 # NWS/FAA network
bbox = [-131.77422,21.081833,-58.66989,53.096016] # CONUS domain

# Get station data based on start/end dates, selecting UTC, all variables of 
# interest, and stations within the NWS/FAA network & CMAQ domain
# NOTE: REQUESTING LARGE AMOUNTS OF DATA MAY REQUIRE PAYMENTS TO SYNOPTIC DATA!
mesonet_api = m.timeseries(start=starting,end=ending,obtimezone='utc',vars=var_list,varoperator='and',network=network,bbox=bbox)

# Save dictionary as binary NumPy file 
np.save(base_dir+'Data/Mesonet/meso_%s.npy'%starting[:8],mesonet_api)
'''

# Iterate over files in directory 
d_str3 = base_dir+'Data/Mesonet/'
mesonet_obs = [f for f in os.listdir(d_str3) if f.endswith('.npy')] 

obs_meso_all = []
for mo in mesonet_obs:
    # Load NumPy array
    mesonet = np.load(d_str3+mo).flat[0]
    
    # Find latitudes, longitudes, and observations for all stations 
    lat_meso = [s['LATITUDE'] for s in mesonet['STATION']]
    lon_meso = [s['LONGITUDE'] for s in mesonet['STATION']]
    obs_meso = [s['OBSERVATIONS'] for s in mesonet['STATION']]
    
    # Reorganize and filter out station data 
    obs_meso_new = []
    for s,stations in enumerate(obs_meso):
        dnew = []
        for dt in stations['date_time']: # convert dates into datetime format 
            dnew.append(dts.datetime.strptime(dt,'%Y-%m-%dT%H:%M:%SZ'))
        df_meso = pd.DataFrame({'date':dnew})
        for v in var_list: # add met variables to dataframe
            if v+'_set_1' in stations:
                df_meso[v] = stations[v+'_set_1']
                if v=='wind_direction': # get cosine of wind direction
                    df_meso[v] = np.cos(df_meso[v]*np.pi/180)
        try: # try to get first data value of each hour
            df_meso = df_meso.resample(rule='H',on='date').first()
            #df_meso = df_meso.resample(rule='H',on='date').mean() # try to average data by hour
        except: # skip station if no numeric types to aggregate
            continue
        # Skip station if missing hours of data and/or the first hour doesn't round to 0
        if len(df_meso)!=24 or df_meso['date'][0].round('60min').to_pydatetime().hour!=0: 
            continue
        df_meso['LAT'] = mesonet['STATION'][s]['LATITUDE']
        df_meso['LON'] = mesonet['STATION'][s]['LONGITUDE']
        obs_meso_new.append(df_meso)
        
    obs_meso_new = pd.concat(obs_meso_new,sort=True)
    obs_meso_all.append(obs_meso_new) # combine list of dataframes into one

obs_meso_all = pd.concat(obs_meso_all,sort=True) # combine list of dataframes into one 

#######################################################
##########     HOURLY MODEL VERIFICATION     ##########
#######################################################   

# Loop through each day of model output
# NOTE: Run loop multiple times for >24 forecast hours (e.g. 1-24, then 25-48)
bias_all = []      
for i,d in enumerate(alldates): # i = index, d = date string
    
    # Set initialization & start/end forecast times for given model run
    init = dts.datetime.combine(dts.datetime.strptime(d,'%Y%m%d').date(),dts.time(hour=int(run)))
    start_t = init + timedelta(hours=hr_start)
    end_t = init + timedelta(hours=hr_end)
    
    # Calculate and save all-inclusive datetime objects in a list
    alldatetimes=[]
    for result in perdelta(start_t, end_t, timedelta(hours=1)):
        alldatetimes.append(result)

    # total hours (will be used to filter out stations with missing hour(s))
    #time_diff = end_t - start_t
    #allhr = round(time_diff.total_seconds() / 3600)+1
    allhr = len(alldatetimes)
    print(allhr)
    
    # Loop through each model
    bias_df = {}
    stat_df = {}
    for mod in models:
        
        #########################################
        ##########     MODEL OUTPUT    ##########  
        #########################################
        
        # Change directory
        os.chdir(base_dir+'Data/'+'%sCMAQ'%mod+'/'+'aqm.'+d) 
        
        # Loop through each model variable
        # Open, read, and close model output files (for given forecast hours)
        # Convert/calculate variables and set variable labels, units, levels
        for v in var_all:
            if v=='pm25': # total particulate matter 2.5 (ug/m^3)
                nc = Dataset('aqm.t%sz.aconc_sfc.nc'%run,'r')
                pm = nc.variables['PM25_TOT'][hr_start-1:hr_end,0,:,:]
                var = pm
                nc.close()
            if v=='air_temp': # air temperature at 2 m (C°)
                nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
                temp = nc.variables['TEMP2'][hr_start:hr_end+1,0,:,:] - 273.15 # K to C°
                var = temp
                nc.close()
            if v=='relative_humidity': # relative humidity (%) 
                nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
                temp = nc.variables['TEMP2'][hr_start:hr_end+1,0,:,:] # air temperature at 2 m (K)
                psfc = nc.variables ['PRSFC'][hr_start:hr_end+1,0,:,:] # surface pressure (Pa)
                q2 = nc.variables['Q2'][hr_start:hr_end+1,0,:,:] # water vapor mixing ratio at 2 m (kg/kg)
                # Calculate 2-m relative humidity using Q2, T2, PSFC, & constants
                # http://mailman.ucar.edu/pipermail/wrf-users/2012/002546.html
                rh = q2/((379.90516/psfc)*np.exp(17.2693882*(temp-273.16)/(temp-35.86)))*100  
                var = rh
                nc.close()
            if v=='wind_speed': # wind speed at 10 m (m/s)
                nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
                ws10 = nc.variables['WSPD10'][hr_start:hr_end+1,0,:,:] # wind speed at 10 m (m/s)
                var = ws10
                nc.close()
            if v=='wind_direction': # cosine of wind direction at 10 m
                nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
                wd10 = nc.variables['WDIR10'][hr_start:hr_end+1,0,:,:]
                var = np.cos(wd10*np.pi/180) # degrees to cosine of wind direction
                nc.close()
            if v=='pblh': # planetary boundary layer height (m)
                nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
                pblh = nc.variables['PBL2'][hr_start:hr_end+1,0,:,:] 
                var = pblh
                nc.close()
            
            '''
            # Choose point or cross point model grid lat/lon, depending on variable
            if v=='wind_speed' or v=='wind_direction':
                lon_mod = lon_dot[0,0,:,:]
                lat_mod = lat_dot[0,0,:,:]
            else:
                lon_mod = lon_cro[0,0,:,:]
                lat_mod = lat_cro[0,0,:,:]
            '''
            
            # Loop through each forecast hour
            # Choose station lat, lon, and met variable observations
            bias_dt = []
            stat_dt = []
            for idt,dt in enumerate(alldatetimes):    
                if v=='pm25':
                    lon_obs = airnow.loc[airnow['DATETIME']==dt]['LON']
                    lat_obs = airnow.loc[airnow['DATETIME']==dt]['LAT']
                    var_obs = airnow.loc[airnow['DATETIME']==dt]['PM2.5']
                elif v=='pblh':
                    lon_obs = pblh_df.loc[pblh_df['DATETIME']==dt]['LON']
                    lat_obs = pblh_df.loc[pblh_df['DATETIME']==dt]['LAT']
                    var_obs = pblh_df.loc[pblh_df['DATETIME']==dt]['PBLH']
                else:
                    lon_obs = obs_meso_all.loc[dt,'LON']
                    lat_obs = obs_meso_all.loc[dt,'LAT']
                    var_obs = obs_meso_all.loc[dt,v]
            
                # Sample lat/lon grid cell information
                var_mod = []
                for k in np.arange(0,len(lat_obs)):
                    iy,ix = naive_fast(lat_cro[0,0,:,:],lon_cro[0,0,:,:], float(list(lat_obs)[k]), float(list(lon_obs)[k]))
                    var_mod.append(var[idt,iy,ix])
        
                # Calculate model/observation difference (bias)
                var_diff = var_mod-var_obs
                
                # Add biases at each station to list for all forecast hours
                bias_dt.append(var_diff)
                
                # Compute statistics 
                df = pd.DataFrame({'MOD':var_mod,'OBS':var_obs})
                stat_dt.append(stats(df,'MOD','OBS'))
            
            # Create dictionary keys and fill with bias values for each variable/model 
            bias_df[v+'_'+mod] = bias_dt
            #bias_df.setdefault(v+'_'+mod,[]).append(bias_dt)
            # Create dictionary keys and fill with stat values for each variable/model 
            stat_df[v+'_'+mod] = stat_dt
            
    # Add bias dataframe to list for all days      
    bias_all.append(bias_df)
    print(d)  
              
# Compute statistics 
#df = pd.DataFrame({'%s_%s'%(mod,v):var_mod,'OBS_%s'%v:var_obs,'DIFF_%s'%v:var_diff})
#stats_df = stats(df,'%s_%s'%(mod,v),'OBS_%s'%v,var_units)

#################################################################
##########     Calculate Mean Bias by Forecast Hour    ##########
#################################################################

'''
# Try using this section instead if looking at multiple days of model output
# Create list of forecast hours
fh = np.arange(1,len(alldatetimes)+1)

# Calculate mean bias for each forecast hour for each variable/model
mean_bias = {}
for item in bias_all:
    for key_var,item_var in item.items():
        item_mean = [np.mean(item_bias) for item_bias in item_var[0]]
        mean_bias.setdefault(key_var,[]).append(item_mean)
mean_bias_all = {}
for k_mean,v_mean in mean_bias.items():
    for fhi in fh-1:
        item_mean = np.mean([vm[fhi] for vm in v_mean])
        mean_bias_all.setdefault(k_mean,[]).append(item_mean)
'''

# Create list of forecast hours
fh = np.arange(1,len(alldatetimes)+1)

# Calculate mean bias for each forecast hour for each variable/model
mean_bias_all = {}
for key_var,value_var in bias_all[0].items(): # remove 0 if loading .npy file
    for f_ind in value_var:
        f_mean = np.mean(f_ind)
        mean_bias_all.setdefault(key_var,[]).append(f_mean)

'''
########################################################
##########     Bias Maps by Forecast Hour     ##########
########################################################

# projection, lat/lon extents and resolution of polygons to draw
# resolutions: c - crude, l - low, i - intermediate, h - high, f - full
m1 = Basemap(llcrnrlon=-121,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

# Convert the lat/lon values to x/y projections.
x, y = m1(list(lon_obs),list(lat_obs))

# Insert hyphens into date string
dt_str = dts.datetime.strptime(d,'%Y%m%d').strftime('%Y-%m-%d')

# Create bias maps
fig1 = plt.figure(figsize=(8,6.5))
ax1 = plt.subplot(1,1,1)

# Add states, counties, coasts, national borders,& land-sea colored mask:
m1.drawcoastlines()
m1.drawstates() 
m1.drawcountries()
m1.drawlsmask(land_color='white', ocean_color='#CCFFFF') # land_color = linen               
 
#cmap = plt.get_cmap('seismic',12) # 12 discrete colors
cmap = plt.get_cmap('bwr')
cmap.set_over('maroon',3)
cmap.set_under('navy',3)

# Draw the contours and filled contours
#sc = m1.scatter(x,y,s=ws_diff,c=ws_diff,edgecolors='k',cmap='RdYlBu',vmin=-5,vmax=5)
clevs = [-3,-2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,2,3]

sc = m1.scatter(x,y,s=(abs(var_diff)*1.3)+100,c=var_diff,cmap=cmap,vmin=-3,vmax=3,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))

ax1.set_title('%sZ %s-CMAQ | F%i | Valid: %s' %(run,mod,dt.hour,dt_str), loc='right')
plt.title('%sZ %s-CMAQ PBLH Bias (m): %s'%(run,mod,dt_str),fontsize=16)

#clevs = np.arange(-30, 30+5, 5)
#cb = plt.colorbar(sc,orientation='horizontal',ticks=clevs,pad=0.05, aspect=50)
cb = plt.colorbar(sc,orientation='horizontal',pad=0.05,aspect=50,extend='both')
  
#plt.show()
fig1.tight_layout()

plt.savefig(base_dir+'Output/%s_CMAQ_PBLH/'%mod + d + '_%sz_bias.png'%run, dpi=300, bbox_inches='tight') # save figure
'''

##############################################################
##########     Time Series Plots - PM2.5 & PBLH     ##########
##############################################################

# Create time series plots of biases averaged by forecast hour
for v in var_all:
    fig3,ax3 = plt.subplots(figsize=(6,4))
    
    ax3.plot(fh,mean_bias_all[v+'_NAM'],c='b',label='NAM-CMAQ')
    ax3.plot(fh,mean_bias_all[v+'_FV3'],c='r',label='FV3GFS-CMAQ')
    plt.axhline(y=0, color='k', linestyle='-')
    ax3.set_title('Time Series (CONUS) | Initial Time: 2019-06-09 12z',fontsize='14')
    plt.xlabel('Forecast Hour (%s UTC Cycle)'%run,labelpad=10)
    if v=='pm25':
        ax3.set(ylabel='PM2.5 Bias ($\u03bcg/m^3$)',ylim=(-4,4))
    if v=='air_temp':
        ax3.set(ylabel='Temperature Bias (C°)',ylim=(-2,2))
    if v=='relative_humidity':
        ax3.set(ylabel='Relative Humidity Bias (%)',ylim=(-15,15))
    if v=='wind_speed':
        ax3.set(ylabel='Wind Speed Bias (m/s)',ylim=(-2,2))
    if v=='wind_direction': 
        ax3.set(ylabel='Cosine of Wind Direction Bias',ylim=(-1,1))
    if v=='pblh':
        ax3.set(ylabel='PBLH Bias (m)',ylim=(-300,300))
    ax3.set_xticks(np.arange(2,len(fh)+2,2))
    ax3.legend(loc='best',fancybox=True, shadow=True)
    
    #plt.show()
    fig3.tight_layout()
    fig3.savefig(base_dir+'Output/Case_Study/%s_bias_timeseries.png'%(v),dpi=300,bbox_inches='tight')

#####################################################################
##########     Scatter Plots - PM2.5 Bias vs. PBLH Bias    ##########
#####################################################################
 
fig4,ax4 = plt.subplots(figsize=(6,4))

# Ignore nan values (keep "good" points) 
pblh_NAM = np.array(mean_bias_all['pblh_NAM'])[~np.isnan(mean_bias_all['pblh_NAM'])]
pm25_NAM = np.array(mean_bias_all['pm25_NAM'])[~np.isnan(mean_bias_all['pblh_NAM'])]
pblh_FV3 = np.array(mean_bias_all['pblh_FV3'])[~np.isnan(mean_bias_all['pblh_FV3'])]
pm25_FV3 = np.array(mean_bias_all['pm25_FV3'])[~np.isnan(mean_bias_all['pblh_FV3'])]

# Calculate correlation coefficients (NAM-CMAQ and FV3GFS-CMAQ)
cc_nam = round(np.corrcoef(pblh_NAM,pm25_NAM)[0,1],2) 
cc_fv3 = round(np.corrcoef(pblh_FV3,pm25_FV3)[0,1],2) 
ax4.scatter(pblh_NAM,pm25_NAM,color='b',marker='o',label='NAM-CMAQ: r = %.2f'%cc_nam)
ax4.scatter(pblh_FV3,pm25_FV3,color='r',marker='s',label='FV3GFS-CMAQ: r = %.2f'%cc_fv3)

ax4.set_title('PM2.5 Bias vs. PBLH Bias')
ax4.set_xlabel('PBLH Bias (m)')
ax4.set_ylabel('PM2.5 Bias ($\u03bcg/m^3$)')
plt.ylim([-4,4])
ax4.set_yticks(np.arange(-4, 4+1, 1))
plt.xlim([-300,300])
ax4.set_xticks(np.arange(-300, 300+50, 50))
ax4.legend(loc='best',fancybox=True,shadow=True)

# Perform linear regression (ignore nan)
m1,b1 = np.polyfit(pblh_NAM,pm25_NAM,deg=1) 
m2,b2 = np.polyfit(pblh_FV3,pm25_FV3,deg=1) 
ax4.plot(pblh_NAM, [m1*l + b1 for l in pblh_NAM], '-b')
ax4.plot(pblh_FV3, [m2*l + b2 for l in pblh_FV3], '-r')

#plt.show()
fig4.tight_layout()
fig4.savefig(base_dir+'Output/Case_Study/correlation_pm25_vs_pblh.png',dpi=300) # save figure            

'''                
# Calling Linux commands to convert png files to a gif file
cmd = 'convert -delay 100 -loop 0 *.png animation.gif'
os.chdir('/home/meteo/bvy5062/Desktop/')
os.system(cmd)

# Grab AirNow PM2.5 site observations from website
url = 'http://www.airnowapi.org/aq/data/?startDate=2019-01-01T13&endDate= \
        2019-01-02T12&parameters=PM25&BBOX=-131.77422,21.081833,-58.66989, \
        53.096016&dataType=C&format=text/csv&verbose=0&nowcastonly=0&API_KEY= \
        3304CB9C-8FF8-4532-89B7-B6231FDF7CB8'
airnow = urllib.request.urlopen(url).read()
'''





