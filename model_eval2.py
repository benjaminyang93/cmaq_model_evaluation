### Model Evaluation: NAM-CMAQ vs. FV3GFS-CMAQ
### NCEP Internship - Summer 2019
### Author: Benjamin Yang

# Import modules
import pandas as pd
import numpy as np
import datetime as dts
from datetime import date, time, timedelta
from netCDF4 import Dataset
import pytz
import os
import met_functions as met
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from MesoPy import Meso
from matplotlib import dates
import re
import urllib
import scipy as sp
plt.close('all')

# Modify default figure properties to make them more legible
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

# Set a directory containing python scripts
base_dir = '/Users/Ben/Documents/NCEP/'

# set a directory to save output files
fig_dir = base_dir + 'Output/'

# all the functions are saved in Met_functions_for_Ben.py
#exec(open(base_dir + 'met_functions.py').read())

# Date format
dtfmt = '%Y%m%d'

# Specify start/end dates, model run time, & number of forecast hours
start = date(2019,1,1) # Jan 1, 2019
end = date(2019,1,19) # Jan 19, 2019
run = '12' # model run time (00z, 06z, 12z, or 18z)
daily = 'max' # PM2.5 daily 1-hr maximum ('max') or 24-hr average ('avg') 
hr_out = 4 # number of forecast hours (4)

# Calculate and save all-inclusive dates in a list
alldates=[]
for result in met.perdelta(start, end, timedelta(days=1)):
    alldates.append(re.sub('-','',str(result)))

# List of forecast hours for a given run
hours = np.arange(1,hr_out+1,1)

#################################################################
##########     MesoWest Meteorological Observations    ##########  
#################################################################

# Enter Mesonet API token (generate using API key)
m = Meso(token = '23fd1c019ccb4ec3a6946eb3a13c99ad')

# Convert the start and end time to the string format requried by the API
start_t = dts.datetime.combine(dts.datetime.strptime(alldates[0],'%Y%m%d').date(),dts.time(hour=int(run)))
end_t = dts.datetime.combine(dts.datetime.strptime(alldates[-1],'%Y%m%d').date(),dts.time(hour=int(run)+hr_out))
start_t = start_t.strftime("%Y%m%d%H%M")
end_t = end_t.strftime("%Y%m%d%H%M")

# String of some MesoWest variables available from this list:
# https://developers.synopticdata.com/mesonet/v2/api-variables/
variables = 'relative_humidity,wind_speed,wind_direction'

var_list = variables.split(",")

# Get station data based on start/end dates, selecting UTC, all variables of 
# interest, and stations within the AirNow network & CMAQ domain
mesonet = m.timeseries(start=start_t,end=end_t,obtimezone='utc',vars=var_list,varoperator='and',
                       network=136,bbox=[-131.77422,21.081833,-58.66989,53.096016])

# Find latitudes, longitudes, and observations for all stations 
lat_meso_all = [i['LATITUDE'] for i in mesonet['STATION']]
lon_meso_all = [i['LONGITUDE'] for i in mesonet['STATION']]
obs = [i['OBSERVATIONS'] for i in mesonet['STATION']]
ws_meso = []
lat_meso = []
lon_meso = []

# Iterate over stations and variables
for idx, ob in enumerate(obs):
    if 'wind_speed_set_1' in ob:
        lat_meso.append(lat_meso_all[idx])
        lon_meso.append(lon_meso_all[idx])
        ws_meso.append(ob['wind_speed_set_1'])
        
    
'''
# Obtain longitude/latitude data for dot points
grddot2d = Dataset('/Users/Ben/Documents/NCEP/Data/grddot2d.nc','r')
lon_dot = grddot2d.variables['LON'][:]
lat_dot = grddot2d.variables['LAT'][:]
grddot2d.close()

# Obtain longitude/latitude data for cross points
grdcro2d = Dataset('/Users/Ben/Documents/NCEP/Data/grdcro2d.nc','r')
lon_cro = grdcro2d.variables['LON'][:]
lat_cro = grdcro2d.variables['LAT'][:]
grdcro2d.close()

for mod in ['NAM','FV3GFS']: 
    for d in alldates:
        # Change directory
        os.chdir(base_dir+'Data/'+'%s_CMAQ'%mod+'/'+d) 
    
        # Iterate over files in directory 
        modeloutput = [i for i in os.listdir(base_dir+'Data/'+'%s_CMAQ/'%mod+d) if i.endswith('.nc')] 
        
        for f in modeloutput:
            # open and read file using netCDF function (Dataset)
            nc = Dataset(f,'r')
            #print(nc)
            
            # obtain relevant variables
            if f=='aqm.t%sz.O3_pm25.nc'%run:
                pm = nc.variables['pm25'][:] # particulate matter 2.5 (ug/m^3)
                var = pm
                var_label = '$PM_{2.5}$'+' $(\u03bcg/m^3)$'
                var_units = '$\u03bcg/m^3$'
                clevs = np.arange(0, 40+5, 5)
                f_label = 'PM2.5' 
            
            if f=='sfc_met_n_PBL.t%sz.nc'%run:
                pblh = nc.variables['PBL2'][:] # planetary boundary layer height (m)
                var = pblh
                var_label = 'PBL Height (m)'
                var_units = 'm'
                clevs = np.arange(0, 2000+200, 200)
                f_label = 'PBLH'
            if f=='Spec_humid.t%sz.nc'%run:
                w = nc.variables['QV'][:] # water vapor mixing ratio (kg/kg)
                q = w/(1+w)*1000 # specific humidity (g/kg)
                var = q
                var_label = 'Specific Humidity (g/kg)'
                var_units = 'g/kg'
                clevs = np.arange(0, 20+2, 2)
                f_label = 'Humid'
            if f=='UV_wind.t%sz.nc'%run:
                u = nc.variables['UWIND'][:] # u component of true wind at dot point (m/s)
                v = nc.variables['VWIND'][:] # v component of true wind at dot point (m/s)
                U = np.sqrt((u**2)+(v**2)) # wind speed (m/s)
                var = U
                var_label = 'Wind Speed (m/s)'
                var_units = 'm/s'
                clevs = np.arange(0, 20+2, 2)
                f_label = 'Wind'
            
            # Close the netCDF file
            nc.close()
            
            if f=='UV_wind.t%sz.nc'%run:
                lon = lon_dot
                lat = lat_dot
            else:
                lon = lon_cro
                lat = lat_cro
            
            # projection, lat/lon extents and resolution of polygons to draw
            # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
            m=Basemap(projection='mill',llcrnrlon=lon.min(),urcrnrlon=lon.max(),
                      llcrnrlat=lat.min(),urcrnrlat=lat.max(),resolution='c')
            
            # Convert the lat/lon values to x/y projections.
            x, y = m(lon,lat)
            
            # Starting datetime
            dt = dts.datetime.combine(dts.datetime.strptime(d, '%Y%m%d').date(),dts.time(hour=int(run)+1))
            
            ###############################################################
            ##########     Hourly CONUS Maps of Model Ouptut     ##########
            ###############################################################
            
            for hr in hours:
                # Adjust datetime
                dt_str = re.sub(':00:00','z',str(dt))
                dt += timedelta(hours=1)
                
                # Create figure
                fig1 = plt.figure(figsize=(8,6))
                ax1 = plt.subplot(1,1,1)
                
                # Add states, counties, coasts, national borders,& land-sea colored mask:
                m.drawcoastlines()
                m.drawstates() 
                m.drawcountries()
                m.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
                #m.drawcounties() # you can even add counties (and other shapefiles!)
                
                parallels = np.arange(25,50+5,5) # make latitude lines ever 5 degrees from 30N-50N
                meridians = np.arange(-125,-65+10,10) # make longitude lines every 5 degrees from 95W to 70W
                m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
                m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
                
                # Plot colorfill and dashed contours of PM2.5 in ppmv
                if f=='aqm.t%sz.O3_pm25.nc'%run: 
                    cf = ax1.contourf(x[0,0,:,:],y[0,0,:,:],var[hr,0,:,:],clevs,cmap=plt.cm.coolwarm)
                else:
                    cf = ax1.contourf(x[0,0,:,:],y[0,0,:,:],var[hr+1,0,:,:],clevs,cmap=plt.cm.coolwarm)
                cb = plt.colorbar(cf, orientation='horizontal', pad=0.05, aspect=50)
                #cb.set_label(var_label)
                #csf = ax1.contour(x[0,0,:,:],y[0,0,:,:],var[hr,0,:,:],clevs,colors='grey',linestyles='solid')
                #ax1.clabel(csf, fmt='%d')
                
                # Plot wind barbs (m/s), regrid to reduce number of barbs
                if f=='UV_wind.t%sz.nc'%run:
                    wind_slice = (slice(None, None, 20), slice(None, None, 20))
                    ax1.barbs(x[0,0,:,:][wind_slice],y[0,0,:,:][wind_slice],
                              u[hr+1,0,:,:][wind_slice],v[hr+1,0,:,:][wind_slice],
                              pivot='middle',color='black')
                
                # Add some titles
                ax1.set_title(var_label, loc='left')
                ax1.set_title('%sZ %s-CMAQ | F%i | Valid: %s' %(run,mod,hr,dt_str), loc='right')
                
                #plt.show()
                fig1.tight_layout()
                
                dt_str = re.sub('-','',dt_str)
                dt_str = re.sub(' ','_',dt_str)
                plt.savefig(fig_dir+'%s_CMAQ/'%mod + f_label + '/' + dt_str + 
                            '.png', dpi=300, bbox_inches='tight') # save figure
'''   
#######################################################
##########     AirNow PM2.5 Observations     ##########
#######################################################

# Change directory
os.chdir(base_dir+'Data/Airnow') 

# Iterate over files in directory 
airnow_obs = [i for i in os.listdir(base_dir+'Data/Airnow') if i.endswith('.csv')] 

# Load data from CSV files
airnow = []
colnames=['LAT', 'LON', 'DATETIME', 'PARAM', 'PM2.5', 'UNITS']
usecol = ['LAT','LON','PM2.5'] 
for a in airnow_obs:
    data = pd.read_csv(a,names=colnames,usecols=usecol,header=None)
    data[data['PM2.5']<0]=np.nan # ignore negative PM2.5 values
    dat = data.groupby(['LAT','LON'],sort=False,as_index=False).max()
    airnow.append(dat)

#########################################################
##########     FV3GFS-CMAQ PM2.5 Daily Max     ##########
#########################################################

cnt = 0 # initializing index for AirNow PM2.5 obs
df_pm = [] # dataframe for time series
stat_pm = [] # dataframe for time series & statistics 
for mod in ['FV3GFS']: # add 'NAM' later
    for d in alldates:
        # Extract AirNow info for corresponding day
        lat_obs = airnow[cnt]['LAT']
        lon_obs = airnow[cnt]['LON']
        pm_obs = airnow[cnt]['PM2.5']
        
        # Change directory
        os.chdir(base_dir+'Data/'+'%s_CMAQ_dailyPM'%mod+'/'+'aqm.'+d) 
        
        if daily == 'max':
            f = 'aqm.t%sz.%s_1hr_pm25.148.nc' %(run,daily) # FV3GFS-CMAQ daily max PM2.5 (1/1/19)
        if daily == 'ave':
            f = 'aqm.t%sz.%s_24hr_pm25.148.nc' %(run,daily) # FV3GFS-CMAQ daily max PM2.5 (1/1/19)
        
        # open and read file using netCDF function (Dataset)
        nc = Dataset(f,'r')
        #print(nc)
            
        # obtain relevant variables
        pm_fv3 = nc.variables['PDMAX1_1sigmalevel'][1,:,:]
        lat_fv3 = nc.variables['latitude'][:]
        lon_fv3 = nc.variables['longitude'][:]-360 # convert from degrees east to degrees
        
        # Close the netCDF file
        nc.close()
            
        pm_mod = []
        # Sample lat/lon grid cell information
        for k in np.arange(0,len(lat_obs)):
            iy,ix = met.naive_fast(lat_fv3, lon_fv3, lat_obs[k], lon_obs[k])
            pm_mod.append(pm_fv3[iy,ix])
        
        # Calculate model/observation difference 
        pm_diff = pm_mod-pm_obs
        
        # Compute statistics 
        df = pd.DataFrame({'%s_PM'%mod:pm_mod,'OBS_PM':pm_obs,'DIFF':pm_diff})
        mean_pm = np.mean(df)
        df_pm.append(mean_pm)
        stats = met.stats(df,'%s_PM'%mod,'OBS_PM','$\u03bcg/m^3$')
        stat_pm.append(stats)
        
        # projection, lat/lon extents and resolution of polygons to draw
        # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
        m2 = Basemap(llcrnrlon=-121,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51,
                projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
        
        # Convert the lat/lon values to x/y projections.
        x, y = m2(list(lon_obs),list(lat_obs))
        
        # Insert hyphens into date string
        dt_str = dts.datetime.strptime(d,'%Y%m%d').strftime('%Y-%m-%d')
        
        
        #############################################
        ##########     Daily Bias Maps     ##########
        #############################################
        
        fig2 = plt.figure(figsize=(8,6.5))
        ax2 = plt.subplot(1,1,1)
        
        # Add states, counties, coasts, national borders,& land-sea colored mask:
        m2.drawcoastlines()
        m2.drawstates() 
        m2.drawcountries()
        m2.drawlsmask(land_color='white', ocean_color='#CCFFFF') # land_color = linen               
        
        #cmap = plt.get_cmap('seismic',12) # 12 discrete colors
        cmap = plt.get_cmap('bwr')
        cmap.set_over('maroon',30)
        cmap.set_under('navy',30)
        
        # Draw the contours and filled contours
        #sc = m2.scatter(x,y,s=pm_diff,c=pm_diff,edgecolors='k',cmap='RdYlBu',vmin=-200,vmax=200)
        clevs = [-30,-20,-15,-10,-8,-6,-4,-2,0,2,4,6,8,10,15,20,30]
        sc = m2.scatter(x,y,s=(abs(pm_diff)*1.3)+50,c=pm_diff,cmap=cmap,vmin=-30,vmax=30,norm=colors.BoundaryNorm(clevs,ncolors=cmap.N))
        
        plt.title('%sZ %s-CMAQ Daily Max PM2.5 Bias ($\u03bcg/m^3$): %s'%(run,mod,dt_str),fontsize=16)
        
        #clevs = np.arange(-30, 30+5, 5)
        #cb = plt.colorbar(sc,orientation='horizontal',ticks=clevs,pad=0.05, aspect=50)
        cb = plt.colorbar(sc,orientation='horizontal',pad=0.05,aspect=50,extend='both')
          
        #plt.show()
        fig2.tight_layout()
        
        plt.savefig(fig_dir+'%s_CMAQ_dailyPM/'%mod + d + '_%sz_bias.png'%run, dpi=300, bbox_inches='tight') # save figure
        
        
        cnt+= 1
        '''
        
###############################################
##########     Time Series Plots     ##########
###############################################

# Insert hyphens into date strings for entire period
date_list = []
for d in alldates:
    date_list.append(dts.datetime.strptime(d,'%Y%m%d').strftime('%m-%d'))
   
# Create a time series plot of PM2.5 averaged over CONUS domain
fig3,ax3 = plt.subplots(figsize=(6,4))

# AirNow observations
y_obs = [item['OBS_PM'] for item in df_pm]
ax3.plot(date_list,y_obs,c='k',label='Observations')

# FV3GFS-CMAQ model output        
y_fv3 = [item['FV3GFS_PM'] for item in df_pm]
ax3.plot(date_list,y_fv3,c='b',label='FV3GFS-CMAQ')

ax3.set_title('Time Series: Jan. 2019 (CONUS)',fontsize='14')
ax3.set(xlabel='Date',ylabel='PM2.5 ($\u03bcg/m^3$)',ylim=(0,50))
plt.xticks(rotation=45)
#fig3.autofmt_xdate(rotation=30)

# Compute correlation coefficient
y = pd.DataFrame({'OBS_PM':y_obs,'FV3_PM':y_fv3})
coefdet = met.r2(y,'FV3_PM','OBS_PM')

# Display r^2 on plot
ax3.text(0.9,0.92,'$r^2$ = %s'%coefdet,fontsize=12,ha='center',va='center',transform=ax3.transAxes)
ax3.legend(loc='upper left',fancybox=True, shadow=True)

#plt.show()
fig3.tight_layout()

fig3.savefig(fig_dir+'FV3GFS_CMAQ_timeseries.png',dpi=300,bbox_inches='tight')

###############################################################################

# Create PM2.5 NMB time series plot
fig4,ax4 = plt.subplots(figsize=(6,4))

# Normalized Mean Bias (NMB) - FV3GFS-CMAQ
nmb = [item.at['NMB [%]','FV3GFS_PM'] for item in stat_pm]
ax4.plot(date_list,nmb,c='b',label='FV3GFS-CMAQ')

ax4.set_title('Time Series: Jan. 2019 (CONUS)',fontsize='14')
ax4.set(xlabel='Date',ylabel='PM2.5 NMB (%)',ylim=(0,250))
plt.xticks(rotation=45)
ax4.legend(loc='lower right',fancybox=True, shadow=True)
#plt.show()
fig4.tight_layout()
fig4.savefig(fig_dir+'FV3GFS_CMAQ_timeseries_nmb.png',dpi=300,bbox_inches='tight')

###############################################################################

# Create PM2.5 NME time series plot
fig5,ax5 = plt.subplots(figsize=(6,4))

# Normalized Mean Error (NME) - FV3GFS-CMAQ
nme = [item.at['NME [%]','FV3GFS_PM'] for item in stat_pm]
ax5.plot(date_list,nme,c='b',label='FV3GFS-CMAQ')

ax5.set_title('Time Series: Jan. 2019 (CONUS)',fontsize='14')
ax5.set(xlabel='Date',ylabel='PM2.5 NME (%)',ylim=(0,250))
plt.xticks(rotation=45)
ax5.legend(loc='lower right',fancybox=True, shadow=True)
#plt.show()
fig5.tight_layout()
fig5.savefig(fig_dir+'FV3GFS_CMAQ_timeseries_nme.png',dpi=300,bbox_inches='tight')

###############################################################################

# Create PM2.5 RMSE time series plot
fig6,ax6 = plt.subplots(figsize=(6,4))

# Root Mean Square Error (RMSE) - FV3GFS-CMAQ
rmse = [item.at['RMSE [$\u03bcg/m^3$]','FV3GFS_PM'] for item in stat_pm]
ax6.plot(date_list,rmse,c='b',label='FV3GFS-CMAQ')

ax6.set_title('Time Series: Jan. 2019 (CONUS)',fontsize='14')
ax6.set(xlabel='Date',ylabel='PM2.5 RMSE ($\u03bcg/m^3$)',ylim=(0,50))
plt.xticks(rotation=45)
ax6.legend(loc='lower right',fancybox=True, shadow=True)
#plt.show()
fig6.tight_layout()
fig6.savefig(fig_dir+'FV3GFS_CMAQ_timeseries_rmse.png',dpi=300,bbox_inches='tight')

###############################################################################

# Create PM2.5 R^2 time series plot
fig7,ax7 = plt.subplots(figsize=(6,4))

# Coefficient of Determination (R^2) - FV3GFS-CMAQ
r2 = [item.at['R^2 [-]','FV3GFS_PM'] for item in stat_pm]
ax7.plot(date_list,r2,c='b',label='FV3GFS-CMAQ')

ax7.set_title('Time Series: Jan. 2019 (CONUS)',fontsize='14')
ax7.set(xlabel='Date',ylabel='PM2.5 $R^2$',ylim=(0,0.1))
plt.xticks(rotation=45)
ax7.legend(loc='upper right',fancybox=True, shadow=True)
#plt.show()
fig7.tight_layout()
fig7.savefig(fig_dir+'FV3GFS_CMAQ_timeseries_r2.png',dpi=300,bbox_inches='tight')

'''                
# Calling Linux commands to convert png files to a gif file (saved in new directory)
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





