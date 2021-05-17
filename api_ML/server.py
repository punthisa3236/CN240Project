from fastapi import FastAPI, UploadFile, Form, File
from skimage.feature import hog
import cv2
import numpy as np

import pickle

def findhog(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd

def fix(i1,i2,max):
    if(i1 < 10):
        return 0, i2-i1
    if(i2>max):
        return i1-i2+max, max
    return i1,i2

def crop(image):
    width,height,n = image.shape
    loc = cv2.minMaxLoc(image[:,:,1])
    x = loc[3][0]
    y = loc[3][1]
    size = width/3

    y1, y2, x1, x2 = y-int(size/2), y+int(size/2), x-int(size/2), x+int(size/2)


    y1,y2 = fix(y1,y2,width)
    x1,x2 = fix(x1,x2,height)
    
    cropPic = image[y1:y2,x1:x2,]

    return cropPic
    
def resize2(image,width,height):
    dim = (width,height)
    res = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return res

def muteluh_model(img):
    gg = img[:,:,0]
    x,y = img[:,:,1].shape

    if np.count_nonzero(gg < 20)/(x*y) > 0.1:
        img = crop(img)

    img = resize2(img,240,240)
    feature = findhog(img).reshape(1,-1)
    

    with open("svmModel.model",'rb') as file:
        modelList = pickle.load(file)

    g = modelList[0].predict_proba(feature)[0][0]
    n = modelList[2].predict_proba(feature)[0][0]
    o = modelList[2].predict_proba(feature)[0][0]
    
    pred = ""
    conf = 0
    
    if g > 0.83 and g > n and g > o:
        pred = "glaucoma"
        conf = g
    elif n > o:
        pred = "normal"
        conf = n
    else:
        pred = "other"
        conf = o
    
    return pred,conf
    
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
    
    class_out, class_conf = muteluh_model(img)
    
    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
        }
    }
