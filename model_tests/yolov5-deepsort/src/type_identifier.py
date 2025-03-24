import cv2
import numpy as np

def identify_vehicle_type(subimage, identification_model, identification_dictionary):
    #The model has an input size of 224,224

    resized_image = cv2.resize(subimage, (224,224)) #getting it in the right size for the input tensor
    
    resized_image = np.expand_dims(resized_image, axis=0)

    normalised_image = np.array(resized_image, dtype=np.float32)/255

    predictions = identification_model.predict(normalised_image)

    final_prediction = np.argmax(predictions, axis = 1)

    return(identification_dictionary[final_prediction[0]])
