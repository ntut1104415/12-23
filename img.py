import streamlit as st
Image = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'])
   if Image is not None:
        Image = Image.read()
       
