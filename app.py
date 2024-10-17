import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
# Replace 'vgg16_brain_tumor_model.h5' with your actual model file
model_path = 'C:/Users/admin/Brain tumor using vgg16/vgg16_brain_tumor_model.h5'
model = load_model(model_path)

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file to the 'uploads' folder
        filename = secure_filename(file.filename)
        img_path = os.path.join('static', 'uploads', filename)
        file.save(img_path)
        
        # Perform prediction using your model on the uploaded image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
        predictions = model.predict(img_array)
        
        # Map the predicted class index to the actual class label
        class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        predicted_label_index = np.argmax(predictions)
        prediction_label = class_labels[predicted_label_index]
        
        prediction_accuracy = np.max(predictions) * 100
        
        return render_template('result.html', 
                               image_path=img_path, 
                               prediction_label=prediction_label, 
                               prediction_accuracy=prediction_accuracy,
                               uploaded_image=filename)  # Pass the uploaded image filename
    else:
        return render_template('index.html', error='Invalid file format. Please upload a valid image.')

if __name__ == '__main__':
    app.run(debug=True)
