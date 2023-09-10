import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size = (64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #converting single image to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction) #return index of max element




#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",['Home','About','Prediction'])

#Home Page
if app_mode == 'Home':
    st.header('Welcome to Fruits and Vegetable Recognition System..!')
    image_path = 'homepage.jpeg'
    st.image(image_path,width=379,use_column_width='always')
    st.divider()
    st.markdown('<div style="text-align: justify;">Our Vegetable and Fruit Recognition System is an innovative computer vision solution designed to automate and streamline various aspects of the produce industry. This system leverages state-of-the-art machine learning and image processing techniques to recognize, classify, and manage different types of vegetables and fruits.</div>', unsafe_allow_html=True)

#About Page
if app_mode == 'About':
    st.header('Welcome to About Page..!')
    st.subheader('About Dataset')
    st.markdown('This dataset contains images of the following food items: ')
    st.code('1. fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.')
    st.code('2. vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.')
    st.subheader('Content')
    st.markdown('This dataset contains three folders:')
    st.text('1. train (100 images each) \n2. test (10 images each) \n3. validation (10 images each)')

#Prediction Page
if app_mode == 'Prediction':
    st.header('Welcome to Model Prediction Page..!')
    test_image = st.file_uploader('Upload Image')
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width='always')
    #Predcit Button
    if(st.button('Predict')):
        st.snow()
        st.write('Prediction is :')
        result_index = model_prediction(test_image)
        #Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting , it's a {}".format(label[result_index]))

        