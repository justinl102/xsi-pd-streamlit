# app.py
import streamlit as st
import pandas as pd 
import numpy as np 
import pickle
from scipy.optimize import curve_fit, least_squares
import plotly.express as px
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
def main():
   pass

if __name__ == '__main__':
    main()

@st.cache_data
def load_dataset(path):
    return pd.read_pickle("growth_df.pkl")

df = load_dataset("growth_df.pkl")
df['monitoring_date'] = pd.to_datetime(df['monitoring_date'])
df['harvest_date'] = pd.to_datetime(df['harvest_date'])
df['stockedAt'] = pd.to_datetime(df['stockedAt'])
df['cycle_days'] = (df['monitoring_date'] - df['stockedAt']).dt.days
df['mlResultWeightCv'] = round(df['mlResultWeightCv'],2)
df_2023 = df[df['monitoring_date'] >= '2023-01-01']

sidebar_farm = st.sidebar.selectbox("Finca",
    #list(labels_reverse_dict.keys()),
    ['Pesquera e Industrial bravito','Velomar Cia Ltda', 'Oro del Rio', 'Grupo Litoral - Cumar','Lanconor', 'White Panther Produktion'], 
    placeholder="Finca",
    )

farm_name = sidebar_farm

farm_df = df_2023[df_2023['farmName'] == farm_name]
farm_df = farm_df[farm_df['cycle_days'] <90]




def log_function(x, a, b):
    return a * np.log(x) + b 

def exponential_function(x, a, b,c):
    return a * np.exp(b * x) + c

def linear_function(x, a, b):
    return a * x + b
def exponential_function_2d(x, a, b,c):
    return a*x**2 +b*x + c

def exponential_fit_3d(x, a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

def growth_model(time, Linf, K, position, w0):
    return Linf * np.exp(-np.exp(position - K * time)) + w0

def get_active_cycles(df, date_cutoff=datetime(2023,10,15)):
   #print(df.dtypes)
   df1 = df.loc[df.groupby('pondName')['monitoring_date'].idxmax()]
   return df1[df1['monitoring_date'] >= date_cutoff]

def get_current_cycle(df, population_id):
   return df[df['populationId'] == population_id]


param_dict = {
    'mlResultAverageWeight':{'lower_factor':0.25,'upper_factor':1.5, 'min_value': 0, 'max_value': 60},
    'mlResultWeightCv':{'lower_factor':0.25,'upper_factor': 4, 'min_value': 0, 'max_value': 50},
    '2week_growth_rate':{'lower_factor':0.1,'upper_factor': 3, 'min_value': -0.5, 'max_value': 5},
    'growth_since_stocking':{'lower_factor':0.1,'upper_factor': 3, 'min_value': 0, 'max_value': 250},
    'growth_rate_since_stocking':{'lower_factor':0.1,'upper_factor': 4, 'min_value': 0, 'max_value': 1},
    'weekly_growth_rate_two_weeks':{'lower_factor':0.1,'upper_factor': 3, 'min_value': 0, 'max_value': 1},
}

labels_dict = {
    'mlResultAverageWeight':'Peso Promedio',
    'mlResultWeightCv':'CV',
  #  '2week_growth_rate':'xx',
   # 'growth_since_stocking':'Peso Promedio',
   # 'growth_rate_since_stocking':'Peso Promedio',
    'weekly_growth_rate_two_weeks':'2 Week Avg Growth Rate',
    'growth_rate_since_stocking' :'Cumulative Growth Rate'
}
labels_dict_diff = {
   'mlResultAverageWeight_pct_diff':'Peso Promedio',
    'mlResultWeightCv_pct_diff':'CV',
  #  '2week_growth_rate':'xx',
   # 'growth_since_stocking':'Peso Promedio',
   # 'growth_rate_since_stocking':'Peso Promedio',
    'weekly_growth_rate_two_weeks_pct_diff':'2 Week Average Growth Rate',
    'growth_rate_since_stocking_pct_diff' :'Cumulative Growth Rate'
   
}
reverse_labels = dict((v,k) for k,v in labels_dict.items())


model_dict = {
    #'mlResultAverageWeight': {"model": exponential_fit_3d,"p0":None, "bounds":([-100, -100,-100, -100], [100, 100,100, 100])},
    'mlResultAverageWeight': {"model": linear_function,"p0":None, "bounds":([-1000, -1000,], [1000, 1000,])},
    'mlResultWeightCv': {"model": exponential_function,"p0":None,'bounds':([-1000, -1000,-1000], [1000, 1000,1000])},
   # '2week_growth_rate': {"model": logarithmic_function,"p0":None,"bounds":None},
    'growth_since_stocking': {"model": exponential_fit_3d,"p0":None,"bounds":None},
    'growth_rate_since_stocking': {"model": linear_function,"p0":None,"bounds":([-1000, -1000], [1000, 1000])},
    'weekly_growth_rate_two_weeks': {"model": linear_function,"p0":None,"bounds":([-1000, -1000], [1000, 1000])},

   # 'weekly_growth_rate_two_weeks': {"model": logarithmic_function,"p0":[0.21020183188716965, 1.475460221686761], 'bounds':([1.1052524301498764e-17, 0.90000000000130473], [1.1052524301498764e-17, 1.1000000000130473])}

}


def remove_outliers(df, x_column, y_column,  upper_factor=2, lower_factor = 0.25, num_intervals=5):
    # Sort the DataFrame by the time_column
    df = df.sort_values(by=x_column)
    # Generate time intervals using qcut
    df['time_intervals'] = pd.qcut(df[x_column], num_intervals, labels=False)
    df_list = []
    # Iterate through each time interval
    for interval_num in range(num_intervals):
        interval_df = df[df['time_intervals'] == interval_num]
        interval_median = interval_df[y_column].median()
        
        upper_threshold = upper_factor * interval_median
        lower_threshold = lower_factor * interval_median
        interval_df = interval_df[(interval_df[y_column]< upper_threshold) & (interval_df[y_column]> lower_threshold) ]
        df_list.append(interval_df)
    training_df = pd.concat(df_list)
    return training_df

def get_curve_params(df, y, x = 'cycle_days'):
  x_train = df[x]
 
  y_train = df[y]
  
  model = model_dict[y]['model']
  p0 = model_dict[y]['p0']
  bounds = model_dict[y]['bounds']
  curve_params, covariance = curve_fit(model,
                                     x_train,
                                     y_train,
                                    # p0 = p0,
                                     maxfev = 10000000,
                                   #  bounds=([0.1, 1.1], [0.3, 1.3])
                                     bounds =bounds,
                                  #   method='dogbox'
                                          )
  print(curve_params)
  return curve_params 




def plot_benchmark(curve_params_y1,curve_params_y2, model_1, model_2, x_min, x_max, increment, x_label, y_label_1, y_label_2):
  
  x_plot = np.arange(x_min, x_max, increment)
  print("curve_params_below again")
  print(curve_params_y1)
  print(model_1)
  print(model_1(10,*curve_params_y1))
  y_plot_1 = np.round(model_1(x_plot, *curve_params_y1),3)
  y_plot_2 = np.round(model_2(x_plot, *curve_params_y2),3)

  return pd.DataFrame({x_label:x_plot,
                       y_label_1:y_plot_1,
                       y_label_2:y_plot_2
                       })

def get_difference_from_benchmark(curve_params, x1,y1, y_variable):
    model = model_dict[y_variable]['model']
    #x_actual['cycle_days'] = 80
    result = round(model(x1, *curve_params),3)
    diff = round(y1 - result,3)
    pct_diff = round(diff / result,3)
    return(y1, result, diff, pct_diff)
   

active_df = get_active_cycles(farm_df)
sidebar_var1 = st.sidebar.selectbox(
    "Variable #1",
    #list(labels_reverse_dict.keys()),
    list(reverse_labels.keys()),
    placeholder="Variable #1",
    )

sidebar_var2 = st.sidebar.selectbox(
    "Variable #2",
    #list(labels_reverse_dict.keys()),
    list(reverse_labels.keys()),
    placeholder="Variable #2",
    )

sidebar_pond = st.sidebar.selectbox(
    "Piscina",
    #list(labels_reverse_dict.keys()),
    active_df['pondName'], 
    placeholder="Select Pond",
    )





color_1 = "#83c9ff"
color_2 = "#0068c9"

y_variable1 = reverse_labels[sidebar_var1]
y_variable2 = reverse_labels[sidebar_var2]

x_variable = 'cycle_days'

model_1 = model_dict[y_variable1]['model']
model_2 = model_dict[y_variable2]['model']
model1_bounds = model_dict[y_variable1]['bounds']
model2_bounds = model_dict[y_variable2]['bounds']


active_df = get_active_cycles(farm_df)



farm_df_clean1 = remove_outliers(farm_df, x_column = x_variable, y_column = y_variable1)
farm_df_clean2 = remove_outliers(farm_df, x_column = x_variable, y_column = y_variable2)

population_id = active_df.loc[active_df['pondName'] == sidebar_pond, 'populationId'].iloc[0]

current_cycle_df = get_current_cycle(farm_df_clean1, population_id)
current_cycle_df.sort_values('monitoring_date', inplace = True)
max_cycle_days = current_cycle_df['cycle_days'].max() + 10

curve_params1 = get_curve_params(farm_df_clean1, x = x_variable, y = y_variable1)
curve_params2 = get_curve_params(farm_df_clean2, x = x_variable, y = y_variable2)

plot_df = plot_benchmark(curve_params1, curve_params2, model_1,model_2, 1,max_cycle_days, 1, x_variable, y_variable1, y_variable2)


last_monitoring = current_cycle_df.loc[current_cycle_df['monitoring_date'].idxmax()]

def generate_trace_benchmarks(fig, plot_df, x_variable, y_variable1, y_variable2):
    
    fig.add_trace(
        go.Scatter( x=plot_df[x_variable], 
                    y=plot_df[y_variable1], 
                    line=dict(color=color_1),
                    name = sidebar_var1,
                        ),
                
                secondary_y=False,
                
            )
    fig.add_trace(
        go.Scatter(x=plot_df[x_variable], 
                   y=plot_df[y_variable2], 
                   line=dict(color=color_2),
                   name = sidebar_var2,
                   
                        ),
                
                secondary_y=True,
             
            )
    fig.update_layout(
    yaxis2=dict(
        side="right",
        tickmode="sync",
    )
    )
    return fig 

def generate_trace_cyles(current_cycle_df,fig,  x_variable, y_variable1, y_variable2):
    plot_df_1 = current_cycle_df.dropna(subset = [y_variable1])
    plot_df_2 = current_cycle_df.dropna(subset = [y_variable2])
    fig.add_trace(
            go.Scatter(x=plot_df_1[x_variable], 
                    y=plot_df_1[y_variable1], 
                    name= "Ciclo " + labels_dict[y_variable1],
                    line=dict(color=color_1)
                    ),
            secondary_y=False,
            
        )
    fig.add_trace(
            go.Scatter(x=plot_df_2[x_variable], 
                    y=plot_df_2[y_variable2], 
                    name= "Ciclo " + labels_dict[y_variable2],
                    line=dict(color=color_2)
                    ),
            secondary_y=True,
        )
    return fig
y_variable1_label = labels_dict[y_variable1]
y_variable2_label = labels_dict[y_variable2]
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = generate_trace_benchmarks(fig, plot_df, x_variable, y_variable1, y_variable2)
fig = generate_trace_cyles(current_cycle_df,fig,  x_variable, y_variable1, y_variable2)
fig.update_xaxes(title_text="Días del ciclo")
fig.update_yaxes(title_text=y_variable1_label, secondary_y=False)
fig.update_yaxes(title_text=y_variable2_label, secondary_y=True)
st.plotly_chart(fig, use_container_width=False)

######
#######
curve_param_dict = {}
for variable in list(labels_dict.keys()):
   clean_df = remove_outliers(farm_df, x_column = x_variable, y_column = variable)
   curve_params = get_curve_params(clean_df, x = x_variable, y = variable)
   curve_param_dict[variable] = curve_params

for variable in list(labels_dict.keys()):
   xx = current_cycle_df.apply(lambda x: get_difference_from_benchmark(curve_param_dict[variable], x.loc[x_variable],x.loc[variable], variable)[3], axis = 1)
   col_str = variable + "_pct_diff"
   current_cycle_df[col_str] = xx

y_variable_test = list(labels_dict.keys())[0] + "_pct_diff"
fig = make_subplots(specs=[[{"secondary_y": False}]])
colors = ["red",'blue','orange','green']
for j,variable in enumerate(['mlResultWeightCv_pct_diff', 'mlResultAverageWeight_pct_diff','growth_rate_since_stocking_pct_diff','weekly_growth_rate_two_weeks_pct_diff']):
    plot_df = current_cycle_df.dropna(subset = [variable])
    fig.add_trace(
                    go.Scatter(
                        x=plot_df[x_variable],
                        y=plot_df[variable],
                        line=dict(color=colors[j]),
                        name = labels_dict_diff[variable],
                        # Set the mode to 'markers'
                        ),
    
                )
fig.update_layout(
   yaxis= dict(
    tickformat=',.0%',
  )
)
fig.update_xaxes(title_text="Días del ciclo")
fig.update_yaxes(title_text="Diferencia porcentual referencia", secondary_y=False)
fig.add_hline(y=0, line_width=3,  line_color="black")
st.plotly_chart(fig, use_container_width=True)

#####
#####

current_distribution = current_cycle_df['weightDistribution'].iloc[-1]
print(type(current_distribution))
if current_distribution:
    hist_df = pd.DataFrame({
            'weight_distribution':current_distribution
        })
    bin_width = 1
    data_range = np.ceil(np.max(current_distribution)) - np.floor(np.min(current_distribution))
    num_bins = int(data_range / bin_width)
    fig3 =  px.histogram(hist_df,x = 'weight_distribution', nbins = num_bins)
        
    fig3.update_layout(bargap=0.1,margin=dict(l=2, r=2, t=2, b=2),)
    fig3.update_xaxes(title_text="Peso")
    st.plotly_chart(fig3, use_container_width=True)
else: 
   'Histograma no disponible'
   


fig = make_subplots(specs=[[{"secondary_y": False}]])

plot_df = plot_benchmark(curve_params1, curve_params2, model_1,model_2, 5,90, 1, x_variable, y_variable1, y_variable2)
    
fig.add_trace(
                go.Scatter(
                    x=farm_df_clean1[x_variable],
                    y=farm_df_clean1[y_variable1],
                    name= sidebar_var1,
                    mode='markers',  # Set the mode to 'markers'
        marker=dict(
            size=7,      # Adjust the marker size
            color='blue'  # Set the marker color
        ) 
                    ),
            )
fig.add_trace(
        go.Scatter( x=plot_df[x_variable], 
                    y=plot_df[y_variable1], 
                    line=dict(color='red'),
                    name= "Punto de referencia"
                        ),
            
                
            )
fig.update_xaxes(title_text="Días del ciclo")
fig.update_yaxes(title_text=y_variable1_label, secondary_y=False)
    
st.plotly_chart(fig, use_container_width=True)




#clean_df[] = current_cycle_df.apply(lambda x: get_difference_from_benchmark(curve_params, x.loc[x_variable],x.loc[variable], variable)[3], axis = 1)
#   cycle_diff_dict[variable] = {clean_df['cycle_days']: pd.Series(cycle_pct_diff)}


