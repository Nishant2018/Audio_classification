from flask import Flask, render_template, request
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__,static_folder='static')
labelencoder = LabelEncoder()

# List of classes
possible_labels = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "enginge_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

labelencoder.fit(possible_labels)

# MongoDB connection
connection_string = 'mongodb+srv://Nishant_17j03:mansinr2211@nishant17j03.focohu4.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(connection_string)
db = client.get_database('Audio_data')
collection_name = "Audio_predictions" 
collection = db[collection_name]  # Create a new collection

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        audio_file = request.files['file']
        audio, sample_rate = librosa.load(audio_file, sr=None)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        model = load_model('last_ann_model.h5', compile=False)
        x_predict = model.predict(mfccs_scaled_features)
        predicted_label = np.argmax(x_predict, axis=1)
        prediction_class = labelencoder.inverse_transform(predicted_label)[0]

        current_date_time = datetime.now()

        # Generate a new collection name based on the date and time


        collection.insert_one({
            'filename': audio_file.filename,
            'predicted_class': prediction_class,
            'date_time': current_date_time
        })

        return render_template('index.html', prediction_text='Predicted class: {}'.format(prediction_class))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
    #app.run(debug=True)
