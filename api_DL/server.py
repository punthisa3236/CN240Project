from fastapi import FastAPI, UploadFile, Form, File
import tensorflow as tf
import cv2
import numpy as np

def predict_model(image_set):
    class_names = ["normal", "glaucoma", "other"]
    
    model = tf.keras.models.load_model("C:/Users/User/DL/model/CNN_model4/")
    predictions = model.predict(image_set)
    
    ans = np.argmax(predictions[0])
    return class_names[ans], predictions[0][ans]

    
app = FastAPI()

@app.get("/")
async def helloworld():
    return {"greeting": "Hello World"}


@app.post("/api/fundus")
async def upload_image(nonce: str=Form(None, title="Query Text"), 
                       image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    image = cv2.resize(img,(100,100))
    image_set=[]
    image_set.append(image)
    image_set = np.asarray(image_set)
    image_set = image_set / 255
    
    
    class_out, class_conf = predict_model(image_set)
    
    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
        }
    }
