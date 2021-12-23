import streamlit as st
Image = st.file_uploader('Upload your portrait here',type=['jpg','jpeg','png'])
   if Image is not None:
        Image = Image.read()
            Image = tf.image.decode_image(Image, channels=3).numpy()                  
            Image = adjust_gamma(Image, gamma=gamma)
            with col1:
                st.image(Image)
