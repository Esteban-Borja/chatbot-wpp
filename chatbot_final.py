from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Cargar intenciones
with open("intenciones.json", "r", encoding="utf-8") as f:
    datos = json.load(f)

train_data = [(frase, intent) for intent, frases in datos.items() for frase in frases]

X_train = [x[0] for x in train_data]
y_train = [x[1] for x in train_data]

# Preprocesamiento
def clean_text(text):
    return ' '.join([word for word in text.lower().split() if word not in stop_words])

X_train = [clean_text(text) for text in X_train]

# Vectorización y entrenamiento
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_vectors, y_train)

# Cargar respuestas
with open("respuestas.json", "r", encoding="utf-8") as f:
    respuestas = json.load(f)

# Predicción y respuesta
def predict_intent(text):
    text_clean = clean_text(text)
    text_vector = vectorizer.transform([text_clean])
    prediction = clf.predict(text_vector)[0]
    return prediction

def obtener_respuesta(mensaje):
    intent = predict_intent(mensaje)
    if intent in respuestas:
        return random.choice(respuestas[intent])
    else:
        return "No entendí tu mensaje o aún no tengo información."

# Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return "¡Chatbot activo!"

@app.route("/chat", methods=["POST"])
def chat():
    mensaje = request.values.get("Body", "")
    respuesta = obtener_respuesta(mensaje)

    # Crear respuesta Twilio
    twilio_resp = MessagingResponse()
    twilio_resp.message(respuesta)
    return str(twilio_resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
