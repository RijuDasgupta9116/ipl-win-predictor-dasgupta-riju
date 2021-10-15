import streamlit as st
import pandas as pd
import pickle

teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore','Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings','Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad', 'Pune', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata','Delhi', 'Rajkot', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town','Port Elizabeth', 'Durban', 'Centurion', 'East London','Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad','Cuttack', 'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam','Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Kanpur','Mohali', 'Bengaluru']


pipe = pickle.load(open('pipe.pkl','rb'))

st.title('IPL Win Predictor')

col1,col2 = st.beta_columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.beta_columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs_completed = st.number_input('Overs Completed')

with col5:
    wickets_out = st.number_input('Wickets out')

if st.button('Predict Probality'):
    runs_left = target - score
    balls_left = 120 - (overs_completed*6)
    wickets_out = 10 - wickets_out
    crr = score/overs_completed
    rrr = (runs_left *6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets_out],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    # st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team +" - "+str(round(win*100)) + "%")
    st.header(bowling_team +" - "+str(round(loss*100)) + "%")