import pandas as pd
import joblib
import pm4py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st 
from PIL import Image

import os 

dirname = os.path.dirname(__file__)


st.session_state.data = joblib.load("../../data/processed/data_clean.pkl")
st.session_state.data["key"] = st.session_state.data["id"].astype(str) + "-" + st.session_state.data["sessionId"].astype(str)

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

with st.sidebar:
    st.title("Customer Journey Dashboard")
    st.write("Gebruik deze tool om inzicht te krijgen in de web customer journey.")
    st.write("Filter op: ")
    st.write("* De omvang (in aantal ondernomen acties) en duur (in seconden) van de sessie .")
    st.write("* Het kanaal, apparaat en platform van de gebruikers.")
    st.write("* De evenementen die wel en niet van interesse zijn.")
    st.write("En klik vervolgens op 'visualiseer'.")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    image = Image.open(os.path.join(dirname, '../../reports/figures/Algorhythm_Positive.png'))
    st.image(image)

tab1, tab2, tab3 = st.tabs(["Parameters", "Session Insights", "Customer Journey"])

with tab1: 
    col1, col2, col3 = st.columns((0.75, 0.75, 0.75))

    with col1: 
        st.number_input("Minimum events", min_value=1, value=1, key="n_cases_min")
        st.number_input("Maximum events", min_value=1, value=100, key="n_cases_max")
        st.write("")
        st.number_input("Minimum duration (s)", min_value=1, value=30, key="n_seconds_min")
        st.number_input("Maximum duration (s)", min_value=1, value=3000, key="n_seconds_max")
    
    with col2: 
       st.multiselect("Channel", options=np.sort(st.session_state.data["channelGrouping"].unique()), key="channel")
       st.multiselect("Device", options=np.sort(st.session_state.data["deviceCategory"].unique()), key="device") 
       st.multiselect("Platform", options=np.sort(st.session_state.data["platform"].unique()), key="platform")  
        #constant per case 

    with col3:
       st.multiselect("Events INcluded", options=np.sort(st.session_state.data["event"].unique()), key="events_needed")
       st.checkbox("Only sessions that have all events specified", value=True, key="filter")
       #filter all cases that include "either" (="or") or "all" (="and")
       st.multiselect("Events EXcluded", options=np.sort(st.session_state.data["event"].unique()), key="events_excluded")
       #exclude all cases that have ANY of these events
       st.number_input("Minimum number of activity occurences", min_value=1, value=3, key="n_act_min")
       st.number_input("Minimum number of transition occurences", min_value=1, value=3, key="n_arc_min")
       st.write("")
       st.write("")
       st.write("")

       if st.button("Visualiseer"):
           
           with tab2: 
               col1, col2 = st.columns((1, 1))


               st.session_state.filtered_data = pm4py.filter_case_size(st.session_state.data, st.session_state.n_cases_min, st.session_state.n_cases_max, case_id_key="key")

               st.session_state.filtered_data = pm4py.filter_case_performance(st.session_state.filtered_data, st.session_state.n_seconds_min, st.session_state.n_seconds_max, 
                                                    case_id_key="key", timestamp_key="activityTime")
               
               if len(st.session_state.channel) != 0:
                    st.session_state.filtered_data = pm4py.filter_event_attribute_values(st.session_state.filtered_data, attribute_key="channelGrouping", 
                                                                        values=st.session_state.channel, level="case", retain=True, 
                                                                        case_id_key="key")
               if len(st.session_state.device) != 0:
                    st.session_state.filtered_data = pm4py.filter_event_attribute_values(st.session_state.filtered_data, attribute_key="deviceCategory", 
                                                                        values=st.session_state.device, level="case", retain=True, 
                                                                        case_id_key="key")
                
               if len(st.session_state.platform) != 0:
                    st.session_state.filtered_data = pm4py.filter_event_attribute_values(st.session_state.filtered_data, attribute_key="platform", 
                                                                        values=st.session_state.platform, level="case", retain=True, 
                                                                        case_id_key="key")

               if st.session_state.filter:
                   for event in st.session_state.events_needed: 
                       st.session_state.filtered_data = pm4py.filter_event_attribute_values(st.session_state.filtered_data, attribute_key="event", 
                                                                           values=[event], level="case", retain=True, 
                                                                           case_id_key="key")
               elif len(st.session_state.events_needed) != 0:
                   st.session_state.filtered_data = pm4py.filter_event_attribute_values(st.session_state.filtered_data, attribute_key="event", 
                                                                           values=st.session_state.events_needed, level="case", retain=True, 
                                                                           case_id_key="key")
                    
               st.session_state.filtered_data = pm4py.filter_event_attribute_values(st.session_state.filtered_data, attribute_key="event", 
                                                                   values=st.session_state.events_excluded, level="case", retain=False, 
                                                                   case_id_key="key")
               
               with col1: 
                   st.header("Events over time")
                   st.session_state.filtered_data["Time"] = st.session_state.filtered_data["activityTime"].dt.round('1H').dt.time
                   st.session_state.filtered_data["Event"] = np.where(st.session_state.filtered_data["event"].isin(st.session_state.events_needed), st.session_state.filtered_data["event"], "OTHER")

                   st.session_state.df_plot = st.session_state.filtered_data.groupby(['Event', 'Time']).size().reset_index().pivot(columns='Event', index='Time', values=0)

                   sns.set_theme(style="darkgrid", palette="bright")
                   fig, ax = plt.subplots()
                   st.session_state.df_plot.plot(kind='bar', stacked=True, ax=ax)
                   st.pyplot(fig)
                
               with col2: 
                   st.header("Distribution of session duration")
                   st.session_state.df_plot = st.session_state.filtered_data.groupby(["id", "sessionId"]).apply(lambda x: (x["activityTime"].max() - x["activityTime"].min()).total_seconds()).reset_index()
                   st.session_state.df_plot.columns = ["id", "sessionId", "n_seconds"]
                   st.session_state.df_plot["Time"] = (st.session_state.df_plot["n_seconds"])

                   sns.set_theme(style="darkgrid", palette="bright")
                   fig2, ax2 = plt.subplots()
                   sns.histplot(st.session_state.df_plot, x="Time", ax=ax2)
                   fig2.savefig(os.path.join(dirname, 'figures/test.png'))
                   st.pyplot(fig2)
                   

           with tab3: 
               map = pm4py.discover_heuristics_net(st.session_state.filtered_data, activity_key="event", timestamp_key="activityTime", case_id_key="key", 
                                                   min_act_count=st.session_state.n_act_min, min_dfg_occurrences=st.session_state.n_arc_min)
               pm4py.save_vis_heuristics_net(map, os.path.join(dirname, 'figures/map.png'))
               image = Image.open(os.path.join(dirname, 'figures/map.png'))
               st.image(image)
            

      

