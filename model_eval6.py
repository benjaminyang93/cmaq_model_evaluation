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
from matplotlib.patches import Polygon
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
start = date(2019,6,1) # June 1, 2019 (2019,6,1)
end = date(2019,6,30) # June 28, 2019 (2019,6,28)
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

# Obtain longitude/latitude data for cross points
grdcro2d = Dataset(base_dir + 'Data/grdcro2d.nc','r')
lon_cro = grdcro2d.variables['LON'][:]
lat_cro = grdcro2d.variables['LAT'][:]
grdcro2d.close()

##################################################################
##########     AIRNOW PM2.5 OBSERVATIONS (DAILY AVG)    ##########
##################################################################

# Change directory
d_str = base_dir+'Data/Airnow/201906_T05_T04'
os.chdir(d_str) 

# Iterate over files in directory 
airnow_obs = [i for i in os.listdir(d_str) if i.endswith('.csv')] 

# Load data from CSV files
airnow = []
colnames=['LAT', 'LON', 'DATETIME', 'PARAM', 'PM2.5', 'UNITS']
usecol = ['LAT','LON','PM2.5'] 
for a in airnow_obs:
    data = pd.read_csv(a,names=colnames,usecols=usecol,header=None)
    data[data['PM2.5']<0]=np.nan # ignore negative PM2.5 values
    if daily=='max':
        dat = data.groupby(['LAT','LON'],sort=False,as_index=False).max()
    if daily=='ave':
        dat = data.groupby(['LAT','LON'],sort=False,as_index=False).mean()
    airnow.append(dat)

############################################################
##########     DAILY AVG PREDICTED PM2.5 MAPS    ###########
############################################################

for mod in models: 
    for d in alldates:
        # Open and read file using netCDF function (Dataset)
        nc = Dataset(base_dir+'Data/%sCMAQ/aqm.%s/aqm.t12z.ave_24hr_pm25.148.nc'%(mod,d),'r')
        pm = nc.variables['PMTF_1sigmalevel'][0,:,:] # particulate matter 2.5 (ug/m^3)
        # Close the netCDF file
        nc.close()
    
        # projection, lat/lon extents and resolution of polygons to draw
        # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
        m = Basemap(llcrnrlon=-121,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51,
                projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
        
        # projection, lat/lon extents and resolution of polygons to draw
        # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
        #m=Basemap(projection='mill',llcrnrlon=lon_cro.min(),urcrnrlon=lon_cro.max(),
        #          llcrnrlat=lat_cro.min(),urcrnrlat=lat_cro.max(),resolution='c')
        
        # Convert the lat/lon values to x/y projections.
        x, y = m(lon_cro,lat_cro)
    
        # Create figure
        fig1 = plt.figure(figsize=(10,8))
        ax1 = plt.subplot(1,1,1)
        
        # Add states, counties, coasts, national borders,& land-sea colored mask:
        m.drawcoastlines()
        m.drawstates() 
        m.drawcountries()
        #m.drawlsmask(land_color='white', ocean_color='#CCFFFF') # land_color = linen   
        
        cmap = plt.get_cmap('coolwarm')
        cmap.set_over('purple')
        cmap.set_under('white')
        clevs = [2,4,6,8,10,15,20,25,35,55]
        norm = colors.BoundaryNorm(clevs, ncolors=cmap.N)             
                     
        # Plot colorfill and dashed contours of PM2.5 in ppmv
        cf1 = ax1.contourf(x[0,0,:,:],y[0,0,:,:],pm,levels=clevs,vmin=2,vmax=55,cmap=cmap,norm=norm,extend='both')
        cb1 = plt.colorbar(cf1,orientation='horizontal',pad=0.08,aspect=50).set_label(label='PM2.5 ($\u03bcg/m^3$)',size=14)
        
        # Add some titles
        ax1.set_title('12z %s-CMAQ | %s' %(mod,re.sub('1906','19-06-',d)),fontsize=18)
        
        #plt.show()
        fig1.tight_layout()
        
        plt.savefig(base_dir+'Output/Case_Study/%s_PM/%s.png'%(mod,d), dpi=300, bbox_inches='tight') # save figure
        
#######################################################################
##########     DAILY AVG PREDICTED PM2.5 DIFFERENCE MAPS    ###########
#######################################################################

for d in alldates:
    # Open and read file using netCDF function (Dataset) - NAM-CMAQ
    nc = Dataset(base_dir+'Data/NAMCMAQ/aqm.%s/aqm.t12z.ave_24hr_pm25.148.nc'%d,'r')
    pm_nam = nc.variables['PMTF_1sigmalevel'][0,:,:] # particulate matter 2.5 (ug/m^3)
    # Close the netCDF file
    nc.close()
    # Open and read file using netCDF function (Dataset) - FV3-CMAQ
    nc = Dataset(base_dir+'Data/FV3CMAQ/aqm.%s/aqm.t12z.ave_24hr_pm25.148.nc'%d,'r')
    pm_fv3 = nc.variables['PMTF_1sigmalevel'][0,:,:] # particulate matter 2.5 (ug/m^3)
    # Close the netCDF file
    nc.close()

    # Compute difference between FV3-CMAQ and NAM-CMAQ PM2.5
    pm_diff = pm_fv3 - pm_nam
    
    # projection, lat/lon extents and resolution of polygons to draw
    # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
    m = Basemap(llcrnrlon=-121,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51,
            projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
    
    # projection, lat/lon extents and resolution of polygons to draw
    # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
    #m=Basemap(projection='mill',llcrnrlon=lon_cro.min(),urcrnrlon=lon_cro.max(),
    #          llcrnrlat=lat_cro.min(),urcrnrlat=lat_cro.max(),resolution='c')
    
    # Convert the lat/lon values to x/y projections.
    x, y = m(lon_cro,lat_cro)

    # Create figure
    fig1 = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)
    
    # Add states, counties, coasts, national borders,& land-sea colored mask:
    m.drawcoastlines()
    m.drawstates() 
    m.drawcountries()
    #m.drawlsmask(land_color='white', ocean_color='#CCFFFF') # land_color = linen   
    
    # Plot southeast U.S. study region bounding box
    x1,y1 = m(-92.053287,28.864955)
    x2,y2 = m(-92.053287,36.539066)
    x3,y3 = m(-83.751954,36.539066)
    x4,y4 = m(-83.751954,28.864955)
    
    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='k',linewidth=2.5)
    ax1.add_patch(poly)
    
    cmap = plt.get_cmap('bwr')
    cmap.set_over('maroon')
    cmap.set_under('navy')
    clevs = [-15,-10,-5,-4,-3,-2,-1,0,1,2,3,4,5,10,15]
    norm = colors.BoundaryNorm(clevs, ncolors=cmap.N)             
                 
    # Plot colorfill and dashed contours of PM2.5 in ppmv
    cf1 = ax1.contourf(x[0,0,:,:],y[0,0,:,:],pm_diff,levels=clevs,vmin=-20,vmax=20,cmap=cmap,norm=norm,extend='both')
    cb1 = plt.colorbar(cf1,orientation='horizontal',pad=0.08,aspect=50).set_label(label='PM2.5 Difference ($\u03bcg/m^3$)',size=14)
    
    # Add some titles
    ax1.set_title('12z Model Difference (FV3-NAM) | %s' %re.sub('1906','19-06-',d),fontsize=18)
    
    #plt.show()
    fig1.tight_layout()
    
    plt.savefig(base_dir+'Output/Case_Study/DIFF_PM/%s.png'%d, dpi=300, bbox_inches='tight') # save figure
                   
################################################################
##########     DAILY TIME SERIES - POINT LOCATION     ##########
################################################################

# Lat/lon coordinates - Birmingham, AL
lat_point = 33.5186 
lon_point = -86.8104 

pm_dict = {} 
for mod in models: 
    for idx,d in enumerate(alldates):
        # Extract AirNow info for corresponding day
        lat_obs = airnow[idx]['LAT']
        lon_obs = airnow[idx]['LON']
        lat_pt = min(lat_obs, key=lambda x:abs(x-lat_point))
        lon_pt = min(lon_obs, key=lambda x:abs(x-lon_point))
        pm_obs = float(airnow[idx][lon_obs==lon_pt]['PM2.5'])
        
        # Open and read file using netCDF function (Dataset)
        nc = Dataset(base_dir+'Data/%sCMAQ/aqm.%s/aqm.t12z.ave_24hr_pm25.148.nc'%(mod,d),'r')
        pm = nc.variables['PMTF_1sigmalevel'][0,:,:] # particulate matter 2.5 (ug/m^3)
        # Close the netCDF file
        nc.close()
        
        # Sample lat/lon grid cell information - model
        iym,ixm = naive_fast(lat_cro[0,0,:,:],lon_cro[0,0,:,:],lat_pt,lon_pt)
        pm_mod = pm[iym,ixm]
        
        # Calculate model/observation difference 
        pm_diff = pm_mod-pm_obs
               
        # Add observed/predicted values & difference to dataframe & list
        pm_dict.setdefault('MOD_%s'%mod,[]).append(pm_mod)
        pm_dict.setdefault('OBS_%s'%mod,[]).append(pm_obs)
        pm_dict.setdefault('DIFF_%s'%mod,[]).append(pm_diff)

# Insert hyphens into date strings for entire period
date_list = []
for d in alldates:
    date_list.append(dts.datetime.strptime(d,'%Y%m%d').strftime('%m-%d'))
   
# Create a time series plot of PM2.5 averaged over CONUS domain
fig3,ax3 = plt.subplots(figsize=(6,4))

# Find correlation coefficients
cc_nam = round(np.corrcoef(pm_dict['OBS_NAM'],pm_dict['MOD_NAM'])[0,1],2) 
cc_fv3 = round(np.corrcoef(pm_dict['OBS_FV3'],pm_dict['MOD_FV3'])[0,1],2) 

# Plot data
ax3.plot(date_list,pm_dict['OBS_NAM'],c='k',label='Observations')      
ax3.plot(date_list,pm_dict['MOD_NAM'],c='b',label='NAM-CMAQ: r = %s'%cc_nam)
ax3.plot(date_list,pm_dict['MOD_FV3'],c='r',label='FV3GFS-CMAQ: r = %s'%cc_fv3)

ax3.set_title('Time Series: June 2019 (Birmingham, AL)',fontsize='14')
ax3.set(xlabel='Date',ylabel='PM2.5 ($\u03bcg/m^3$)',ylim=(0,45))
plt.xticks(rotation=60)
ax3.legend(loc='best',fancybox=True, shadow=True)

#plt.show()
fig3.tight_layout()

fig3.savefig(base_dir+'Output/Case_Study/PM2.5_daily_avg_ts_point.png',dpi=300,bbox_inches='tight')

###############################################################################

# Create PM2.5 bias time series plot
fig4,ax4 = plt.subplots(figsize=(6,4))

ax4.plot(date_list,pm_dict['DIFF_NAM'],c='b',label='NAM-CMAQ')
ax4.plot(date_list,pm_dict['DIFF_FV3'],c='r',label='FV3GFS-CMAQ')
plt.axhline(y=0, color='k', linestyle='-')

ax4.set_title('Time Series: June 2019 (Birmingham, AL)',fontsize='14')
ax4.set(xlabel='Date',ylabel='PM2.5 Bias ($\u03bcg/m^3$)',ylim=(-30,30))
plt.xticks(rotation=60)
ax4.legend(loc='best',fancybox=True, shadow=True)

#plt.show()
fig4.tight_layout()

fig4.savefig(base_dir+'Output/Case_Study/PM2.5_bias_daily_avg_ts_point.png',dpi=300,bbox_inches='tight')

################################################################
##########     AIRNOW PM2.5 OBSERVATIONS (HOURLY)     ##########
################################################################

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

########################################################################
##########     HOURLY MODEL VERIFICATION  - POINT LOCATION    ##########
########################################################################   

# Lat/lon coordinates - Birmingham, AL
lat_point = 33.5186 
lon_point = -86.8104 
     
# Specify model run of interest
d = '20190610'
    
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
var_dict = {}
for mod in models:
    # Change directory
    os.chdir(base_dir+'Data/%sCMAQ/aqm.%s'%(mod,d)) 
    
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
        
        # Loop through each forecast hour
        # Choose station lat, lon, and met variable observations
        bias_dt = []
        stat_dt = []
        for idt,dt in enumerate(alldatetimes):    
            if v=='pm25':
                lat_obs = airnow.loc[airnow['DATETIME']==dt]['LAT']
                lon_obs = airnow.loc[airnow['DATETIME']==dt]['LON']
                lat_pt = min(lat_obs, key=lambda x:abs(x-lat_point))
                lon_pt = min(lon_obs, key=lambda x:abs(x-lon_point))
                var_obs = float(airnow.loc[(airnow['DATETIME']==dt) & (airnow['LON']==lon_pt),'PM2.5'])
            elif v=='pblh':
                lat_obs = pblh_df.loc[pblh_df['DATETIME']==dt]['LAT'].astype(float)
                lon_obs = pblh_df.loc[pblh_df['DATETIME']==dt]['LON'].astype(float)
                try:
                    lat_pt = min(lat_obs, key=lambda x:abs(x-lat_point))
                    lon_pt = min(lon_obs, key=lambda x:abs(x-lon_point))
                    var_obs = float(np.mean(pblh_df.loc[(pblh_df['DATETIME']==dt) & (pblh_df['LON'].astype(float)==lon_pt),'PBLH']))
                except:
                    var_obs = np.nan
            else:
                lat_obs = obs_meso_all.loc[dt,'LAT'].astype(float)
                lon_obs = obs_meso_all.loc[dt,'LON'].astype(float)
                lat_pt = min(lat_obs, key=lambda x:abs(x-lat_point))
                lon_pt = min(lon_obs, key=lambda x:abs(x-lon_point))
                var_obs = float(obs_meso_all.loc[(obs_meso_all.index==dt) & (obs_meso_all['LON'].astype(float)==lon_pt),v])
        
            # Sample lat/lon grid cell information
            iy,ix = naive_fast(lat_cro[0,0,:,:],lon_cro[0,0,:,:],lat_pt,lon_pt)
            var_mod = var[idt,iy,ix]
    
            # Calculate model/observation difference (bias)
            if v=='wind_direction':
                var_diff = abs(var_mod-var_obs) # get absolute error instead
            else:
                var_diff = var_mod-var_obs    

            # Add observed/predicted values & difference to dataframe & list
            var_dict.setdefault('MOD_%s_%s'%(mod,v),[]).append(var_mod)
            var_dict.setdefault('OBS_%s_%s'%(mod,v),[]).append(var_obs)
            var_dict.setdefault('DIFF_%s_%s'%(mod,v),[]).append(var_diff)

########################################################################
##########     FORECAST HOUR TIME SERIES - POINT LOCATION     ##########
########################################################################   

# Create list of forecast hours
fh = np.arange(1,len(alldatetimes)+1)

# Create time series plots of biases averaged by forecast hour
for v in var_all:
    fig3,ax3 = plt.subplots(figsize=(6,4))
    
    ax3.plot(fh,var_dict['DIFF_NAM_'+v],c='b',label='NAM-CMAQ')
    ax3.plot(fh,var_dict['DIFF_FV3_'+v],c='r',label='FV3-CMAQ')
    plt.axhline(y=0, color='k', linestyle='-')
    ax3.set_title('Time Series (Birmingham, AL) | Initial Time: 2019-06-10 12z',fontsize='12')
    plt.xlabel('Forecast Hour',labelpad=10)
    if v=='pm25':
        ax3.set(ylabel='PM2.5 Bias ($\u03bcg/m^3$)',ylim=(-20,50))
    if v=='air_temp':
        ax3.set(ylabel='Temperature Bias (C°)',ylim=(-5,20))
    if v=='relative_humidity':
        ax3.set(ylabel='Relative Humidity Bias (%)',ylim=(-80,80))
    if v=='wind_speed':
        ax3.set(ylabel='Wind Speed Bias (m/s)',ylim=(-5,10))
    if v=='wind_direction': 
        ax3.set(ylabel='Cosine of Wind Direction Abs Error',ylim=(0,2))
    if v=='pblh':
        ax3.set(ylabel='PBLH Bias (m)',ylim=(-1000,2000))
    ax3.set_xticks(np.arange(2,len(fh)+2,2))
    ax3.legend(loc='best',fancybox=True, shadow=True)
    
    #plt.show()
    fig3.tight_layout()
    fig3.savefig(base_dir+'Output/Case_Study/%s_bias_ts_point.png'%v,dpi=300,bbox_inches='tight')
            
#########################################################################
##########     BIAS MAPS FOR DAY (HR 25) AND NIGHT (HR 36)     ##########
#########################################################################

# Specify model run of interest
d = '20190610'

# Set initialization & start/end forecast times for given model run
init = dts.datetime.combine(dts.datetime.strptime(d,'%Y%m%d').date(),dts.time(hour=int(run)))
start_t = init + timedelta(hours=hr_start)
end_t = init + timedelta(hours=hr_end)

# Calculate and save all-inclusive datetime objects in a list
alldatetimes=[]
for result in perdelta(start_t, end_t, timedelta(hours=1)):
    alldatetimes.append(result)
    
for mod in models:
    # Change directory
    os.chdir(base_dir+'Data/%sCMAQ/aqm.%s'%(mod,d)) 
    
    # Loop through each model variable
    # Open, read, and close model output files (for given forecast hours)
    # Convert/calculate variables and set variable labels, units, levels
    for v in var_all:
        if v=='pm25': # total particulate matter 2.5 (ug/m^3)
            nc = Dataset('aqm.t%sz.aconc_sfc.nc'%run,'r')
            pm = nc.variables['PM25_TOT'][hr_start-1:hr_end,0,:,:]
            var = pm
            var_label = 'PM2.5 Bias ($\u03bcg/m^3$)'
        if v=='air_temp': # air temperature at 2 m (C°)
            nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
            temp = nc.variables['TEMP2'][hr_start:hr_end+1,0,:,:] - 273.15 # K to C°
            var = temp
            var_label = 'Temperature Bias (C°)'
        if v=='relative_humidity': # relative humidity (%) 
            nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
            temp = nc.variables['TEMP2'][hr_start:hr_end+1,0,:,:] # air temperature at 2 m (K)
            psfc = nc.variables ['PRSFC'][hr_start:hr_end+1,0,:,:] # surface pressure (Pa)
            q2 = nc.variables['Q2'][hr_start:hr_end+1,0,:,:] # water vapor mixing ratio at 2 m (kg/kg)
            # Calculate 2-m relative humidity using Q2, T2, PSFC, & constants
            # http://mailman.ucar.edu/pipermail/wrf-users/2012/002546.html
            rh = q2/((379.90516/psfc)*np.exp(17.2693882*(temp-273.16)/(temp-35.86)))*100  
            var = rh
            var_label = 'Relative Humidity Bias (%)'
        if v=='wind_speed': # wind speed at 10 m (m/s)
            nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
            ws10 = nc.variables['WSPD10'][hr_start:hr_end+1,0,:,:] # wind speed at 10 m (m/s)
            var = ws10
            var_label = 'Wind Speed Bias (m/s)'
        if v=='wind_direction': # cosine of wind direction at 10 m
            nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
            wd10 = nc.variables['WDIR10'][hr_start:hr_end+1,0,:,:]
            var = np.cos(wd10*np.pi/180) # degrees to cosine of wind direction
            var_label = 'Cosine of Wind Direction Absolute Error'
        if v=='pblh': # planetary boundary layer height (m)
            nc = Dataset('aqm.t%sz.metcro2d.nc'%run,'r')
            pblh = nc.variables['PBL2'][hr_start:hr_end+1,0,:,:] 
            var = pblh
            var_label = 'PBL Height Bias (m)'
        
        # Close the netCDF file
        nc.close()
        
        # Loop through each forecast hour
        # Choose station lat, lon, and met variable observations
        for idt,dt in enumerate([alldatetimes[25-1],alldatetimes[36-1]]):    
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
            if v=='wind_direction':
                var_diff = abs(var_mod-var_obs) # get absolute error instead
            else:
                var_diff = var_mod-var_obs    

            # projection, lat/lon extents and resolution of polygons to draw
            # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
            m1 = Basemap(llcrnrlon=-121,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51,
                    projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
            
            # Convert the lat/lon values to x/y projections.
            x, y = m1(list(lon_obs.astype(float)),list(lat_obs.astype(float)))
            
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
            cmap.set_over('maroon')
            cmap.set_under('navy')
            
            # Draw the contours and filled contours
            #clevs = [-3,-2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,2,3]
            #sc = m1.scatter(x,y,s=(abs(var_diff)*1.3)+100,c=var_diff,cmap=cmap,vmin=-3,vmax=3,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
            if v=='pm25':
                clevs = [-20,-10,-5,-4,-3,-2,-1,0,1,2,3,4,5,10,20]
                sc = m1.scatter(x,y,s=(abs(var_diff)*1.2)+30,c=var_diff,cmap=cmap,vmin=-20,vmax=20,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
            if v=='air_temp':
                clevs = [-5,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,5]
                sc = m1.scatter(x,y,s=(abs(var_diff)*1.5)+20,c=var_diff,cmap=cmap,vmin=-5,vmax=5,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
            if v=='relative_humidity':
                clevs = [-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50]
                sc = m1.scatter(x,y,s=(abs(var_diff)*1.1)+5,c=var_diff,cmap=cmap,vmin=-50,vmax=50,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
            if v=='wind_speed':
                clevs = [-5,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,5]
                sc = m1.scatter(x,y,s=(abs(var_diff)*1.5)+20,c=var_diff,cmap=cmap,vmin=-5,vmax=5,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
            if v=='wind_direction':
                cmap = plt.get_cmap('YlOrRd')
                cmap.set_over('purple')
                cmap.set_under('linen')
                clevs = [0.05,0.1,0.2,0.3,0.5,0.8,1.0,1.5]
                sc = m1.scatter(x,y,s=(abs(var_diff)*1.5)+30,c=var_diff,cmap=cmap,vmin=0.01,vmax=1.5,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
            if v=='pblh':
                clevs = [-500,-300,-100,-50,-30,-10,0,10,30,50,100,300,500]
                sc = m1.scatter(x,y,s=(abs(var_diff)*0.02)+80,c=var_diff,cmap=cmap,vmin=-500,vmax=500,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
                
            cb = plt.colorbar(sc,orientation='horizontal',pad=0.05,aspect=50,extend='both').set_label(var_label,size=14)

            if idt==0:
                ax1.set_title('12z %s-CMAQ | F25 | Valid: 2019-06-11 13z'%mod,fontsize=16)
                fig1.tight_layout()
                plt.savefig(base_dir+'Output/Case_Study/%s_CMAQ_%s_bmap_f25.png'%(mod,v), dpi=300, bbox_inches='tight') # save figure
            if idt==1:
                ax1.set_title('12z %s-CMAQ | F36 | Valid: 2019-06-12 00z'%mod,fontsize=16)
                fig1.tight_layout()
                plt.savefig(base_dir+'Output/Case_Study/%s_CMAQ_%s_bmap_f36.png'%(mod,v), dpi=300, bbox_inches='tight') # save figure

#########################################################
##########     HOVMOLLER DIAGRAMS OF PM2.5     ##########
#########################################################

# Sample lat/lon grid cell information - Birmingham, AL
lat_point = 33.5186 
lon_point = -86.8104 
iy,ix = naive_fast(lat_cro[0,0,:,:],lon_cro[0,0,:,:],lat_point,lon_point)

for mod in ['NAM','FV3']:
    # open and read file using netCDF function (Dataset) - June 9, 2019
    nc = Dataset(base_dir+'Data/%sCMAQ_3D/aqm.20190610/aqm.t12z.conc.nc'%mod,'r')
    pm25 = nc.variables['PM25_TOT'][hr_start-1:hr_end,0:20,iy,ix]    
    nc.close()
    # open and read file using netCDF function (Dataset) - June 9, 2019
    nc = Dataset(base_dir+'Data/%sCMAQ_3D/aqm.20190610/aqm.t12z.metcro3d.nc'%mod,'r')
    zf = nc.variables['ZF'][hr_start:hr_end+1,0:20,iy,ix]    
    nc.close()
    
    # Create list of forecast hours
    x = np.arange(1,48+1,1)
    # Create list of heights in km (average for each level)
    y = zf[0:8,:].mean(axis=0)/1000 
    # Create list of hybrid sigma levels
    #y = np.arange(1,20+1,1)
    
    # Get coordinate matrices from coordinate vectors
    xx,yy = np.meshgrid(x,y)
    # Transpose PM2.5 array
    zz = np.transpose(pm25)
     
    fig2 = plt.figure(figsize=(8,6))
    ax2 = plt.subplot(1,1,1)
    
    cmap = plt.get_cmap('jet')
    cmap.set_over('purple')
    #cmap.set_under('linen')
    #clevs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40]
    clevs = [0,0.1,0.5,1,2,3,4,6,8,10,12,15,20]
    norm = colors.BoundaryNorm(clevs, ncolors=cmap.N)
    
    # Draw the contours and filled contours
    #ax2.contour(X,Y,Z,levels=clevs,linewidths=0.5, colors='k')
    cf2 = ax2.contourf(xx,yy,zz,levels=clevs,vmin=0,vmax=20,cmap=cmap,norm=norm,extend='max')
    
    #cf2 = ax2.contourf(X, Y, pm25, levels=14, cmap='jet',vmin=-20,vmax=20,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N)))
    
    ax2.set_title('%s-CMAQ | Initial Time: 2019-06-10 12z | Birmingham, AL'%mod,fontsize=14)
    ax2.set_xlabel('Forecast Hour',fontsize=12)
    ax2.set_ylabel('Height (km)',fontsize=12)
    ax2.set_xticks(np.arange(2,48+2,2))
    ax2.set_yticks([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])

    plt.colorbar(cf2,orientation='horizontal',pad=0.12,aspect=50).set_label(label='PM2.5 ($\u03bcg/m^3$)',size=12)
    
    #plt.show()
    fig2.tight_layout()
    
    plt.savefig(base_dir+'Output/Case_Study/Hovmoller_%s.png'%mod, dpi=300, bbox_inches='tight') # save figure

###############################################################################





