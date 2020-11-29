#"when you're talking about things with big magnitudes, you need precise data, becuase those make big differences" - charlie baker

#importing libraries
import math
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

#importing data
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

confirmed_table = confirmed_df.melt(id_vars=["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State", "Country_Region","Lat","Long_","Combined_Key"], var_name="Date", value_name="Confirmed").fillna('').drop(["UID", "iso2", "iso3", "code3", "FIPS","Country_Region","Lat","Long_"], axis=1)
death_table = death_df.melt(id_vars=["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State", "Country_Region","Lat","Long_","Combined_Key"], var_name="Date", value_name="Deaths").fillna('').drop(["UID", "iso2", "iso3", "code3", "FIPS","Country_Region","Lat","Long_"], axis=1)

full_table = confirmed_table.merge(death_table)

full_table['Date'] = pd.to_datetime(full_table['Date'])

def get_state(state):
    # summing up the counties in each state
    if full_table[full_table['Province_State'] == state]['Admin2'].nunique() > 1:
        state_table = full_table[full_table['Province_State'] == state]
        summed = pd.pivot_table(state_table, values = ['Confirmed', 'Deaths'],index='Date', aggfunc=sum)
        state_df = pd.DataFrame(summed.to_records())
        return state_df.set_index('Date')[['Confirmed', 'Deaths']]
    df = full_table[(full_table['Province_State'] == state) 
                & (full_table['Admin2'].isin(['', state]))]
    return df.set_index('Date')

def model(N, a, alpha, t):
    return N * (1 - math.e ** (-a * t)) ** alpha

def model_loss(params):
    N, a, alpha = params
    r = 0
    for t in range(len(df)):
        r += (model(N, a, alpha, t) - df.iloc[t, model_index]) ** 2
    return math.sqrt(r)

#an optimization function I found: python has a built-in function to do this, but for the spirit of the project I tried to code it myself
def Nelder_Mead(N, a, alpha):
    x0 = [N,a,alpha]
    x1 = [N*1.05,a,alpha]
    x2 = [N,a*1.05,alpha]
    x3 = [N,a,alpha*1.05]
    var = [x0,x1,x2,x3]
    for j in range(0,500):
        loss = list(map(model_loss,var))
        index = np.argmax(loss)
        xh = var[index]
        best3 = var.copy()
        best3.pop(index)
        best_loss = list(map(model_loss,best3))
        centroid = [np.mean([best3[0][0],best3[1][0],best3[2][0]]),np.mean([best3[0][1],best3[1][1],best3[2][1]]),np.mean([best3[0][2],best3[1][2],best3[2][2]])]
        xr = [centroid[0]+(centroid[0]-xh[0]),centroid[1]+(centroid[1]-xh[1]),centroid[2]+(centroid[2]-xh[2])]
        #reflection
        if(model_loss(xr)<max(best_loss) and model_loss(xr)>min(best_loss)):
            best3.append(xr)
            var = best3.copy()
        #expansion
        elif(model_loss(xr)<min(loss)):
            xr2 = [centroid[0]+2*(centroid[0]-xh[0]),centroid[1]+2*(centroid[1]-xh[1]),centroid[2]+2*(centroid[2]-xh[2])]
            if(model_loss(xr2)<model_loss(xr)):
               best3.append(xr2)
               var = best3.copy()
            else:
                best3.append(xr)
                var = best3.copy()
        #contraction
        else:
            xc = [np.mean([xh[0],centroid[0]]),np.mean([xh[1],centroid[1]]),np.mean([xh[2],centroid[2]])]
            if(model_loss(xc)<model_loss(xh)):
                best3.append(xc)
                var = best3.copy()
            #super contraction
            else:
                best_index = np.argmin(loss)
                xl = var[best_index]
                for i in range(3):
                    point = var[i]
                    var[i] = [np.mean([xl[0],point[0]]),np.mean([xl[1],point[1]]),np.mean([xl[2],point[2]])]
    return var[0]

#confirmed cases
model_index = 0
my_confirmed = Nelder_Mead(200000, 0.05, 15)
print("Confirmed Parameters from my function: ")
print(my_confirmed)

opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
print("Confirmed Parameters from python function:")
print(opt_confirmed)

#deaths
model_index = 1
my_deaths = Nelder_Mead(200000, 0.05, 15)
print("Death Parameters from my function: ")
print(my_deaths)

opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
print("Death Parameters from python function: ")
print(opt_deaths)

model_x = []
for t in range(len(df)):
    model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

plot_color = ['#0000FF55', '#FF000055', '#0000FF99', '#FF000099']

pd.concat([model_sim, df], axis=1).plot(color = plot_color)

start_date = df.index[0]
n_days = len(df) + 30
extended_model_x = []

for t in range(n_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t)])

extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

plot_color = ['#0000FF55', '#FF000055', '#0000FF99', '#FF000099']

pd.concat([extended_model_sim, df], axis=1).plot(color = plot_color)
print('NJ COVID-19 Pjrediction')
print(extended_model_sim.tail())
plt.show()



#compare model success between states
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
table = []

for state in states:
    df = get_state(state)
    model_index = 0
    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_index = 1
    opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
    model_state = []
    error_confirmed = 0
    error_deaths = 0
    for t in range(len(df)):
        t_confirmed = model(*opt_confirmed, t)
        t_deaths = model(*opt_deaths, t)
        error_confirmed += (t_confirmed - df.iloc[t,0])**2
        error_deaths += (t_deaths - df.iloc[t,1])**2
    error_confirmed = math.sqrt(error_confirmed/df.iloc[len(df)-1,0])
    error_deaths = math.sqrt(error_deaths/df.iloc[len(df)-1,1])
    print(state)
    print(error_confirmed)
    print(error_deaths)
    table.append([state, error_confirmed, error_deaths])

state_errors = pd.DataFrame(table)
state_errors.set_index(0, inplace=True)
state_errors.columns = ['Confirmed Error', 'Death Error']
state_errors.to_csv('doc.csv')


#toying around with NJ

def get_state_stop(state, date):
    # summing up the counties in each state
    if full_table[full_table['Province_State'] == state]['Admin2'].nunique() > 1:
        state_table = full_table[(full_table['Province_State'] == state) & (full_table['Date'].dt.date <= date)]
        summed = pd.pivot_table(state_table, values = ['Confirmed', 'Deaths'],index='Date', aggfunc=sum)
        state_df = pd.DataFrame(summed.to_records())
        return state_df.set_index('Date')[['Confirmed', 'Deaths']]
    cd = full_table[(full_table['Province_State'] == state) 
                & (full_table['Admin2'].isin(['', state]))]
    cd2 = df.set_index('Date')
    cd3 =  cd2[(cd2['Date'] > date)]
    return cd3

#end on May 4
df = get_state_stop('New Jersey', datetime.date(2020,5,4))

model_index = 0

opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

model_index = 1

opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

model_x = []
for t in range(len(df)):
    model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

plot_color = ['#0000FF55', '#FF000055', '#0000FF99', '#FF000099']

pd.concat([model_sim, df], axis=1).plot(color = plot_color)

start_date = df.index[0]
n_days = len(df) + 31
extended_model_x = []

for t in range(n_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t)])

extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

print(extended_model_sim.tail(10))

#end on May 5
df = get_state_stop('New Jersey',datetime.date(2020,5,5))

model_index = 0

opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

model_index = 1

opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

model_x = []
for t in range(len(df)):
    model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t)])
model_sim = pd.DataFrame(model_x, dtype=int)
model_sim.set_index(0, inplace=True)
model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

plot_color = ['#0000FF55', '#FF000055', '#0000FF99', '#FF000099']

pd.concat([model_sim, df], axis=1).plot(color = plot_color)

start_date = df.index[0]
n_days = len(df) + 30
extended_model_x = []

for t in range(n_days):
    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t)])

extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)
extended_model_sim.set_index(0, inplace=True)

extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

print(extended_model_sim.tail(10))

