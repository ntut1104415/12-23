import streamlit as st


Image = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'])
 st.image(Image)
