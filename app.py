import streamlit as st
import av
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import matplotlib.pyplot as plt
import pandas as pd
import os

st.set_page_config(layout="wide", page_title="Emotion Detection | ML Model", page_icon="😃")

selection = st.selectbox("Navigation", ["Home", "Code", "Dataset", "About"])

if selection == "Home":
    st.html("<center><h1>Facial Emotion Recognition</h1></center><hr>")
    model = load_model("model.keras")

    # Define emotions (update based on your training data if necessary)
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    op = {i: emotion for i, emotion in enumerate(emotions)}

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.html("<br><h2>Prediction</h2><hr>")
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Convert to RGB since OpenCV loads images in BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the uploaded image
        st.image(image_rgb, caption='Uploaded Image', width=300)

        # Preprocess the image for prediction
        img_resized = cv2.resize(image_rgb, (224, 224))  # Resize to match model input size
        img_array = img_to_array(img_resized) / 255.0  # Normalize pixel values
        input_arr = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Debug: Print image shape and pixel values to compare with notebook
        st.write("Image shape:", img_array.shape)

        # Make emotion prediction
        pred = np.argmax(model.predict(input_arr))
    
        predicted_emotion = op[pred]

        # Display the predicted emotion
        st.write(f"The predicted emotion is: **{predicted_emotion}**")



elif selection == "Code":
    st.html("<center><h1>Code Snippets</h1></center><hr>")

    st.write("Importing the libraries")
    st.code("""
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            from tensorflow.keras.applications import MobileNet
            from tensorflow.keras.losses import categorical_crossentropy
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Flatten, Dense
            from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
            """
    )

    st.write("We start off by using a base model from MobeileNet to build our model on top of it")
    st.code("""
            # Using a pre-trained model
            base_model = MobileNet( input_shape=(224,224,3), include_top= False )

            for layer in base_model.layers:
            layer.trainable = False

            x = Flatten()(base_model.output)
            x = Dense(units=7 , activation='softmax' )(x)

            # creating our model.
            model = Model(base_model.input, x)
            """
    )

    st.write("Compiling the model")
    st.code("""
            model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )
            """
    )

    st.write("We use ImageDataGenerator to prepare our data")
    st.code(
        """
            train_datagen = ImageDataGenerator(
                zoom_range = 0.2, 
                shear_range = 0.2, 
                horizontal_flip=True, 
                rescale = 1./255
            )
    train_data = train_datagen.flow_from_directory(
                directory= "C:/Users/hamid/OneDrive/Desktop/emotion_dataset/train", 
                target_size=(224,224), 
                batch_size=32,
    )

    train_data.class_indices
            """
    )

    with st.expander("Output:"):
        st.code("""
                Found 28709 images belonging to 7 classes.
                {
                    'angry': 0,
                    'disgusted': 1,
                    'fearful': 2,
                    'happy': 3,
                    'neutral': 4,
                    'sad': 5,
                    'surprised': 6
                }
                """, language="bash"
        )

    st.code("""
            val_datagen = ImageDataGenerator(rescale = 1./255 )

            val_data = val_datagen.flow_from_directory(
                    directory= "C:/Users/hamid/OneDrive/Desktop/emotion_dataset/test", 
                    target_size=(224,224), 
                    batch_size=32,
            )
            """
    )

    with st.expander("Output:"):
        st.code("""
                Found 7178 images belonging to 7 classes.
                """, language="bash"
        )

    st.write("Data which is fed to train datagen is visualized below")
    st.code("""
            t_img , label = next(train_data)

            def plotImages(img_arr, label):
                count = 0
                for im, l in zip(img_arr,label) :
                    plt.imshow(im)
                    plt.title(im.shape)
                    plt.axis("off")
                    plt.show()
                    
                    count += 1
                    if count == 10:
                        break
            
            plotImages(t_img, label)
            """
    )

    width = 150
    with st.expander("Output:"):
        st.image("C:/Users/hamid/Downloads/emotion_dataset/train/angry/im0.png", width=width)
        st.image("c:/Users/hamid/Downloads/emotion_dataset/train/disgusted/im0.png", width=width)
        st.image("c:/Users/hamid/Downloads/emotion_dataset/train/fearful/im0.png", width=width)
        st.image("c:/Users/hamid/Downloads/emotion_dataset/train/happy/im0.png", width=width)
        st.image("c:/Users/hamid/Downloads/emotion_dataset/train/neutral/im6.png", width=width)
        st.image("c:/Users/hamid/Downloads/emotion_dataset/train/sad/im1.png", width=width)
        st.image("c:/Users/hamid/Downloads/emotion_dataset/train/surprised/im2.png", width=width)

    st.write("We are now adding early stopping criteria and a model checkpoint")
    st.code("""
            from keras.callbacks import ModelCheckpoint, EarlyStopping

            # early stopping
            es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 5, verbose= 1, mode='auto')

            # model check point
            mc = ModelCheckpoint(filepath="model.keras", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')

            # puting call back in a list 
            call_back = [es, mc]
            """
    )
    st.write("We are saving the trained model at model.keras")

    st.write("Training our data")
    st.code("""
            hist = model.fit(
                train_data, 
                steps_per_epoch= 10, 
                epochs= 30, 
                validation_data= val_data, 
                validation_steps= 8, 
                callbacks=[es,mc]
            )
            """
    )

    with st.expander("Output:"):
        st.code(
            r"""
                Epoch 1/30
                C:\Users\hamid\AppData\Roaming\Python\Python312\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your PyDataset class should call super().__init__(**kwargs) in its constructor. **kwargs can include workers, use_multiprocessing, max_queue_size. Do not pass these arguments to fit(), as they will be ignored.
                self._warn_if_super_not_called()
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.1467 - loss: 21.2150
                Epoch 1: val_accuracy improved from -inf to 0.19922, saving model to model.keras
                10/10 ━━━━━━━━━━━━━━━━━━━━ 32s 2s/step - accuracy: 0.1473 - loss: 21.4324 - val_accuracy: 0.1992 - val_loss: 18.5053
                Epoch 2/30
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.2239 - loss: 17.0664
                Epoch 2: val_accuracy did not improve from 0.19922
                10/10 ━━━━━━━━━━━━━━━━━━━━ 17s 2s/step - accuracy: 0.2274 - loss: 16.8072 - val_accuracy: 0.1836 - val_loss: 13.1848
                Epoch 3/30
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 888ms/step - accuracy: 0.2852 - loss: 11.6581
                Epoch 3: val_accuracy improved from 0.19922 to 0.39062, saving model to model.keras
                10/10 ━━━━━━━━━━━━━━━━━━━━ 14s 1s/step - accuracy: 0.2874 - loss: 11.4990 - val_accuracy: 0.3906 - val_loss: 8.9427
                Epoch 4/30
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 886ms/step - accuracy: 0.3972 - loss: 7.5431
                Epoch 4: val_accuracy did not improve from 0.39062
                10/10 ━━━━━━━━━━━━━━━━━━━━ 14s 1s/step - accuracy: 0.3932 - loss: 7.4701 - val_accuracy: 0.3867 - val_loss: 6.1464
                Epoch 5/30
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 878ms/step - accuracy: 0.4038 - loss: 5.6808
                Epoch 5: val_accuracy improved from 0.39062 to 0.44141, saving model to model.keras
                10/10 ━━━━━━━━━━━━━━━━━━━━ 14s 1s/step - accuracy: 0.4063 - loss: 5.6646 - val_accuracy: 0.4414 - val_loss: 5.0247
                Epoch 6/30
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 947ms/step - accuracy: 0.3769 - loss: 5.0343
                Epoch 6: val_accuracy did not improve from 0.44141
                10/10 ━━━━━━━━━━━━━━━━━━━━ 14s 1s/step - accuracy: 0.3779 - loss: 5.0229 - val_accuracy: 0.4023 - val_loss: 5.1102
                Epoch 7/30
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 910ms/step - accuracy: 0.4281 - loss: 4.6859
                ...
                10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 777ms/step - accuracy: 0.4182 - loss: 5.4653
                Epoch 10: val_accuracy did not improve from 0.44141
                10/10 ━━━━━━━━━━━━━━━━━━━━ 12s 1s/step - accuracy: 0.4185 - loss: 5.4945 - val_accuracy: 0.4141 - val_loss: 6.5665
                Epoch 10: early stopping
                """, language="bash"
        )

    st.write("Now we load the model (model.keras)")
    st.code("""
            from keras.models import load_model

            model = load_model("C:/Users/hamid/Downloads/model.keras")
            """
    )

    st.code("""
            h =  hist.history
            h.keys()
            """
    )

    with st.expander("Output:"):
        st.code("""
                dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss'])
                """
    )

    st.write("Plotting accuracy vs validation-accuracy graph")
    st.code("""
            plt.plot(h['accuracy'])
            plt.plot(h['val_accuracy'] , c = "red")
            plt.title("acc vs v-acc")
            plt.show()
            """
    )

    with st.expander("Output:"):
        st.image("c:/Users/hamid/OneDrive/Desktop/acc-vs-vacc.jpg")

    st.write("Plotting loss vs validation-loss graph")
    st.code("""
            plt.plot(h['loss'])
            plt.plot(h['val_loss'] , c = "red")
            plt.title("loss vs v-loss")
            plt.show()
            """
    )

    with st.expander("Output:"):
        st.image("c:/Users/hamid/OneDrive/Desktop/loss-vs-vloss.jpg")

    st.write("Mapping Output values")
    st.code("""
            op = dict(zip( train_data.class_indices.values(), train_data.class_indices.keys()))
            """
    )

    st.write("Testing our model to see if it predicts accurately")
    st.code("""
            path = "C:/Users/hamid/OneDrive/Desktop/emotion_dataset/test/angry/im4.png"
            img = load_img(path, target_size=(224,224) )

            i = img_to_array(img)/255
            input_arr = np.array([i])
            input_arr.shape

            pred = np.argmax(model.predict(input_arr))

            print(f" the image is of class: {op[pred]}")

            # to display the image  
            plt.imshow(input_arr[0])
            plt.title("input image")
            plt.show()
            """
    )

    with st.expander("Output:"):
        st.code("""
                1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step
                the image is of class: angry
                """, language="bash"
        )
        st.image("c:/Users/hamid/OneDrive/Desktop/testingimg.jpg")

    st.write("Testing Accuracy")
    st.code("""
            # Evaluate the model on the validation/test data
            evaluation = model.evaluate(val_data)

            # Print and accuracy
            print(f"Test Accuracy: {evaluation[1]}")
            """
    )

    with st.expander("Output:"):
        st.code("""
                225/225 ━━━━━━━━━━━━━━━━━━━━ 173s 762ms/step - accuracy: 0.8339 - loss: 5.0338
                Test Accuracy: 0.8298824317455292
                """, language="bash"
    )


elif selection == "Dataset":
    st.html("<center><h1>Sample Images</h1></center><hr>")
    # Individual Images 
elif selection == "About":
    st.title("About this App")
    st.image("newplot.jpg", width=750)
