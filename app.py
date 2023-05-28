import os
import sys
import uuid
import flask
import urllib
from PIL import Image
# from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
# from tensorflow.keras.preprocessing.image import load_img , img_to_array
from flask import jsonify
from flask_cors import CORS, cross_origin


# Integrating Actual Model
import cv2
# import pandas as pd
import numpy as np

import skimage
import warnings
import joblib
warnings.filterwarnings("ignore")


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = 0

# Allowing CORS
CORS(app, support_credentials=True)



ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif','JPG','JPEG'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['airplane' ,'automobile', 'bird' , 'cat' , 'deer' ,'dog' ,'frog', 'horse' ,'ship' ,'truck']

loaded_rf = joblib.load("./mymodel.joblib")


def segment(filepath):
    result={}
    X = cv2.imread(filepath)

    # cv2_imshow(X)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

    # Perform adaptive thresholding
    x,thresholded_img = cv2.threshold(gray_img, 140, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_BINARY)


    # Dilate the mask with a square structuring element
    width = 9
    kernel = np.ones((width, width), np.uint8)
    dilated_img = cv2.dilate(thresholded_img, kernel)

    # Erode the mask with a square structuring element
    eroded_img = cv2.erode(dilated_img, kernel)


    # Specify the desired display width and height
    display_width = 4000
    display_height = 3000

    # Calculate the resize ratio
    resize_ratio = min(display_width / eroded_img.shape[1], display_height / eroded_img.shape[0])

    # Resize the image for display
    resized_img = cv2.resize(eroded_img, (0, 0), fx=resize_ratio, fy=resize_ratio)

    # Display the resized image
    # cv2_imshow(resized_img)


    # Find contours in the binary image
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate bounding boxes for each contour
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    properties2 = np.array(bounding_boxes, dtype=np.int)

    # print(properties2)




    # Define the path to save the cropped images within Colab
    # !s2
    path = './segment/'
    

    img = cv2.imread(filepath)
    # Create the directory to store the cropped images if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Iterate over each row in 'properties2'
    for row in range(properties2.shape[0]):
        x = int(properties2[row, 0]) - 15
        y = int(properties2[row, 1]) - 15
        w = int(properties2[row, 2]) + 30
        h = int(properties2[row, 3]) + 30

        # Crop the region of interest
        # print(img.shape)
        temp = img[y:y+h, x:x+w]
        # print(temp)

        # Generate the filename for the cropped image
        imgnumber = row + 100
        filename = str(imgnumber) + '.png'
        s = os.path.join(path, filename)
        # print(s)
        # print(type(temp))

        if(temp.size==0):
          break
        cv2.imwrite(s, temp)
        curr=actualpredict(s)
        # print(curr)
        try:
            result[curr] += 1
        except KeyError:
            result[curr] = 1
        # result[curr]=result.get('curr',0)+1
        # Save the cropped image in Colab's local file system
        
    return result


def actualpredict(filepath):
    feature_data = []
    image = cv2.imread(filepath)

    # Extract color features
    color_mean = np.mean(image, axis=(0, 1))
    color_std = np.std(image, axis=(0, 1))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract texture features
    glcm = skimage.feature.graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = skimage.feature.graycoprops(glcm, 'contrast')[0, 0]
    energy = skimage.feature.graycoprops(glcm, 'energy')[0, 0]

    # Extract shape features
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Extract edge features
    edges = cv2.Canny(gray_image, 100, 200)
    num_edge_pixels = np.count_nonzero(edges) // 255

    # Append the features, image name, and class name to the dataset
    feature_data.append([
        *color_mean, *color_std,
        contrast, energy,
        area, perimeter,
        num_edge_pixels,

    ])

    feature_vector=np.array(feature_data)
    # Make predictions on the feature vector
    pred=loaded_rf.predict(feature_vector)
    # print(pred[-1])
    return pred[-1]




def predict(filename , model):
    # img = load_img(filename , target_size = (32 , 32))
    # img = img_to_array(img)
    # img = img.reshape(1 , 32 ,32 ,3)

    # img = img.astype('float32')
    # img = img/255.0
    # result = model.predict(img)
    # print(result, file=sys.stderr)
    # dict_result = {}
    # for i in range(10):
    #     dict_result[result[0][i]] = classes[i]

    # res = result[0]
    # res.sort()
    # res = res[::-1]
    # prob = res[:3]
    
    prob_result = []
    class_result = []
    # for i in range(3):
    #     prob_result.append((prob[i]*100).round(2))
    #     class_result.append(dict_result[prob[i]])

    return class_result , prob_result




@app.route('/home',methods = ['GET'])
@cross_origin(supports_credentials=True)
def home():
        temp={
            'message':'hello'
        }
        print("Recieved fake request")
        temp=jsonify(temp)
        return temp

@app.route('/success' , methods = ['GET' , 'POST'])
@cross_origin(supports_credentials=True)
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        # temp=jsonify(message="Success")
        # return temp
        if(request.form):
            # temp=jsonify(message="Success")
            # return temp
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            # temp=jsonify(message="Success123")
            # return temp
            file = request.files['file']
            if file and allowed_file(file.filename):
                # temp=jsonify(message="Success123")
                # return temp
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                # class_result , prob_result = predict(img_path , model)
                final=segment(img_path)
                final = dict(sorted(final.items(), key=lambda x: x[1], reverse=True))
                print(final)
                # result=actualpredict(img_path)
                temp=jsonify(final)
                # temp=jsonify(message="Hello")
                return temp


                # return prob_result
                predictions = {
                    #   "class1":class_result[0],
                    #     "class2":class_result[1],
                    #     "class3":class_result[2],
                    #     "prob1": prob_result[0],
                    #     "prob2": prob_result[1],
                    #     "prob3": prob_result[2],
                        class_result[0]:prob_result[0],
                        class_result[1]:prob_result[1],
                        class_result[2]:prob_result[2]
                }

            else:
                error = "Please upload images of jpg, jpeg and png extension only"
            temp=jsonify(predictions)
            print(temp)
            return temp
            return render_template('index.html' , error = "Hellllllo")
            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                temp=jsonify(message="Failed1")
                return temp

        else:
            temp=jsonify(message="Failed111")
            return temp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)


