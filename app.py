# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:33:42 2021

@author: Jiwoo Ahn
"""

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output 
from datetime import timedelta
from datetime import date as dtdate

# Import relevant libraries
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initiate the app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'AusCovidDash'

colors = {
    'background': '#000000',
    'text': '#5d76a9',
    'label': '#f5b112'
}

# Pull data from John Hopkins University and organise into dataframe 
df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

# Curve fitting Global COVID-19 Cases

def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))

def exponential(t, a, b, c):
    return a * np.exp(b * t) + c
    
def doubling(t):
    return (1+np.log(2)/2)**t

def plotCases(dataframe, column, state, start_date, curvefit, forecast):

    #fig = go.Figure()
    fig = make_subplots(rows=1,cols=2,subplot_titles=('Total Cases','New Cases'))
    fig.update_layout(template='ggplot2')
    fig.update_layout(autosize=True,margin={'t':30})
    #fig.update_layout(title='Australia Covid-19 Dashboard',title_font_size=30,title_x=0.5)     
    fig.update_layout(legend={'title':'Legend','bordercolor':'black','borderwidth':1})
    fig.update_layout(legend_title_font=dict(family="Verdana",size=16,color=colors['text']))
    
    fig.update_layout(
        font=dict(
            family="Verdana",
            size=12,
            color=colors['text']
        ))

    #clean_chart_format(fig)
    
    # PSM_ColorMap = [(0,0,0),
    #             (27/256,38/256,100/256),
    #             (245/256,130/256,100/256),
    #             (134/256,200/256,230/256),
    #             (210/256,210/256,185/256),
    #             (74/256,93/256,206/256),
    #             (249/256,180/256,161/256),
    #             (16/256,23/256,60/256),
    #             (194/256,50/256,13/256),
    #             (37/256,136/256,181/256),
    #             (144/256,144/256,93/256)]
        
    co = dataframe[dataframe[column] == state].iloc[:,4:].T.sum(axis = 1)
    co = pd.DataFrame(co)
    co.columns = ['Cases']
    co['date'] = co.index
    co['date'] = pd.to_datetime(co['date'])  
    mask = (co['date'] >= start_date)
    co = co.loc[mask]
    co['Cases'] = co['Cases'] - co['Cases'][0]
    
    y = np.array(co['Cases'])
    x = np.arange(y.size)
    date = co['date']
    
    x2 = np.arange(y.size+forecast)
    x3 = np.arange(y.size+forecast+100)
    
    date2 = pd.date_range(date[0],freq='1d',periods=len(date)+forecast)
    
    fig.add_trace(go.Scatter(x=date,y=y,mode='markers',name='Total Cases',marker_color='rgba(27,38,100,.8)'),row=1,col=1)

    # Logistic regression -----------------------------------------------------------------------
    lpopt, lpcov = curve_fit(logistic, x, y, maxfev=10000)
    #lerror = np.sqrt(np.diag(lpcov))
    # for logistic curve at half maximum, slope = growth rate/2. so doubling time = ln(2) / (growth rate/2)
    ldoubletime = np.log(2)/(lpopt[1]/2)
    ## standard error
    #ldoubletimeerror = 1.96 * ldoubletime * np.abs(lerror[1]/lpopt[1])
    
    # calculate R^2
    residuals = y - logistic(x, *lpopt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    logisticr2 = 1 - (ss_res / ss_tot)  
    
    if logisticr2 >= 0 and logisticr2 <= 1:
        fig.add_trace(go.Scatter(x=date2,y=logistic(x2, *lpopt), mode='lines', name="Logistic (r2={0}) Td={1}d".format(round(logisticr2,2),round(ldoubletime,1)),line_color='rgba(245,130,100,.8)',line_shape='spline',line_dash='dash'),row=1,col=1)
    # -----------------------------------------------------------------------
    
    
    # Exponential regression--------------------------------------------------------------------
    epopt, epcov = curve_fit(exponential, x, y, bounds=([0.99,0,-0.001],[1.01,0.9,0.001]), maxfev=10000)
    #eerror = np.sqrt(np.diag(epcov))
    
    # for exponential curve, slope = growth rate. so doubling time = ln(2) / growth rate
    edoubletime = np.log(2)/epopt[1]
    ## standard error
    #edoubletimeerror = 1.96 * edoubletime * np.abs(eerror[1]/epopt[1])
    
    # calculate R^2
    residuals = y - exponential(x, *epopt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    expr2 = 1 - (ss_res / ss_tot)
    
    if expr2 >= 0 and expr2 <= 1:
        fig.add_trace(go.Scatter(x=date2,y=exponential(x2, *epopt), mode='lines', name="Exponential (r2={0}) Td={1}d".format(round(expr2,2),round(edoubletime,1)),line_color='rgba(134,200,230,.8)',line_shape='spline',line_dash='dash'),row=1,col=1)
    # --------------------------------------------------------------------

    # Calculations for new cases
    delta = np.diff(co['Cases'])
    fig.add_trace(go.Scatter(x=y[1:],y=delta,mode='lines',name='New Daily Cases',line_color='rgba(210,210,185,.8)'),row=1,col=2)
    
    dbl_cases = 2**(x3/2)
    dbl_delta = 0.5*np.log(2)*np.exp((np.log(2)*x3)/2)
    fig.add_trace(go.Scatter(x=dbl_cases,y=dbl_delta,mode='lines',name='2 Day Doubling Time',line = {'color':'black','dash':'dash'}),row=1,col=2)
    5
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    fig.update_xaxes(title_text='Date',row=1,col=1)
    fig.update_yaxes(title_text='Total confirmed cases since {0}'.format(start_date),row=1,col=1)
    
    fig.update_xaxes(title_text='Total confirmed cases since {0}'.format(start_date),range=[0,np.log10(max(y)+100)],type="log",row=1,col=2)
    fig.update_yaxes(title_text='New daily cases',type="log",range=[0,np.log10(max(delta)+100)],row=1,col=2)
        
    return fig
    
aus_states = ['Queensland','New South Wales','Victoria','Western Australia','South Australia', 'Tasmania', 'Australian Capital Territory']



app.layout = html.Div([
    html.H1(children='Australia Covid-19 Dashboard',
            style={'textAlign': 'center','font-family':'Verdana','color': colors['text'],'padding-top': 20}),
    html.P(children='''Dashboard configurations''',
           style={'textAlign': 'center','font-size':24,'font-family':'Verdana','color': colors['text'],'padding-bottom': 10}),
    html.Div([html.Label(["State",dcc.Dropdown(id='state-select', options=[{'label': i, 'value': i} for i in aus_states],
                       value='Queensland', style={'width': '250px', 'display':'inline-block', 'margin-left':'10px','vertical-align':'middle'})])],style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    html.Div([
        html.Label(["Start Date",dcc.DatePickerSingle(id='my-date-picker-single',
        min_date_allowed=dtdate(2020, 1, 22),
        max_date_allowed=(dtdate.today() - timedelta(days=7)),
        initial_visible_month=dtdate(2021, 1, 1),
        date=dtdate(2021, 1, 1),style={'display':'inline-block', 'margin-left':'10px'})])],style={'vertical-align':'middle','margin-top':'10px','font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']}),
    dcc.Graph('dashboard', figure={"layout" : {"height":600}},config={'displayModeBar': False}),
    html.Div(dcc.Markdown('''
        The total cases chart presents all COVID-19 cases in each state since the specified start date, as a function of time. Logistic and exponential regression indicates potential trajectories of growth.
                          
        The new cases chart presents daily increase in COVID-19 cases vs. the total confirmed cases to date. When plotted in this way, exponential growth is represented as a straight line that slopes upwards. 
                          
                          
        _Created by : Jiwoo Ahn_
        
        Data provided by Johns Hopkins University (updated daily around 00:00 UTC / 20:00 ET)
        
        [Github Repo](https://github.com/j-ahn/AusCovidDash)
        '''), style = {'font-size':10,'font-family':'Verdana','textAlign':'center','color':colors['text']})
    ])

@app.callback(
    Output('dashboard', 'figure'),
    [Input('state-select', 'value')],
    [Input('my-date-picker-single','date')]
)
def update_graph(value,date_value):
    return plotCases(df, 'Province/State', value, date_value, True, 3)

if __name__ == '__main__':
    app.run_server(debug=False)