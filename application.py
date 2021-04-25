import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import pickle
import xgboost as xgb
import time
from custom_preprocessing import *
from streamlit import caching




st.title('All state claim severity prediction application')

st.markdown("***")

st.image(image = 'allstate_logo.png')




st.subheader('For how many records do you want to predict the claim loss?')
option = st.radio('',('Single claim record', 'Multiple claim records'))
st.write('You selected:', option)



st.subheader('Input a csv file with claim records')
uploaded_file = st.file_uploader(' ')
print(uploaded_file)

predict_button = st.button('Predict the loss ')

if predict_button:
	if option == 'Single claim record':
		df = pd.read_csv(uploaded_file.name)

		with st.spinner('Processing claim'):
			loss = final_fun_1(df)
		loss_df = pd.DataFrame(loss)
		st.success('The predicted loss is ')
		loss_df
		#st.success('The predicted loss is ' + str(k))

	if option == 'Multiple claim records':
		df = pd.read_csv(uploaded_file.name)
		df
		with st.spinner('Processing claim'):
			loss = final_fun_1(df)

		loss_df = pd.DataFrame(loss)
		st.success('The predicted loss is ')
		loss_df


st.markdown("***")

#st.write(' Try again with different inputs')

result = st.button(' Try again')
if result:
	
	uploaded_file = st.empty()
	predict_button = st.empty()
	caching.clear_cache()









