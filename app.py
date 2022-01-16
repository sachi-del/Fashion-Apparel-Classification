import numpy as np
from itertools import chain
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import os
import tensorflow as tf
import cv2
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model

model = load_model('fashion.h5',compile=False)
lab={0:'Goggles',1:'Hat',2:'Jacket',3:'Shirt',4:'Shoes',5:'Shorts',6:'T-Shirt',7:'Trouser',
     8:'Wallet',9:'Watch'
 }
class_names = ['Goggles','Hat','Jacket','Shirt','Shoes','Shorts','T-Shirt','Trouser','Wallet','Watch']
def processed_img(img_path):
    img = cv2.imread(img_path)
    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(preds)
    preds_unlist = list(chain(*preds))
    print(preds_unlist)
    preds_int = [int((round(i, 2))) for i in preds_unlist]
    print(preds_int)
    # self.final_pred_unused = dict(zip(self.class_names,self.preds_int))
    final_pred = dict(zip(class_names, preds_int))
    # finale = final_pred[1]
    print(100 * '-')
    print(final_pred)
    ans=None
    t=0
    for key in final_pred.keys():
        if final_pred[key]>t:
            t=final_pred[key]
            ans=key
    print(ans)
    return ans


def run():
    img1 = Image.open('./meta/logo.jpg')
    img1 = img1.resize((350,350))
    st.image(img1,use_column_width=False)
    st.title("Fashion Apparel Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Dataset is taken from Kaggle"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './uploads/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Apparel is: "+result)
run()
