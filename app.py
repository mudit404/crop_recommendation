from flask import Flask, request, render_template
import numpy as np
import joblib
import os

# Load model and scalers
model = joblib.load('model.pkl')
sc = joblib.load('standscaler.pkl')
ms = joblib.load('minmaxscaler.pkl')

# Path to the folder containing crop images
IMAGE_FOLDER = 'static/crop_images'

# creating flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# Define routes

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{}".format(crop)
        print(result)
        # Search for image file with any extension
        image_filename = find_crop_image(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        image_filename = 'default.jpg'

    return render_template('index.html', result=result, image_filename=image_filename)

def find_crop_image(crop_name):
    """
    Search for the image file corresponding to the crop name in the IMAGE_FOLDER.
    """
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().startswith(crop_name.lower()):
            return filename
    return 'default.jpg'  # Default image if not found

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
