import numpy as np
import requests
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import base64
from mimetypes import guess_type
from datetime import datetime
import boto3
import uuid
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)  # can adjust max_workers as needed

# New! Authenticate to MinIO object store
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['MINIO_URL'],  # e.g. 'http://minio:9000'
    aws_access_key_id=os.environ['MINIO_USER'],
    aws_secret_access_key=os.environ['MINIO_PASSWORD'],
    region_name='us-east-1'  # required for the boto client but not used by MinIO
)

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']  # FastAPI server URL

# New! for uploading production images to MinIO bucket
def upload_production_bucket(img_path, preds, confidence, prediction_id):
    classes = np.array(["apple black rot", "apple leaf", "apple mosaic virus", "apple rust", "apple scab",
    "banana leaf", "banana panama disease", "basil downy mildew", "basil leaf", "bean halo blight",
    "bean leaf", "bean mosaic virus", "bean rust", "bell pepper leaf", "bell pepper leaf spot",
    "blueberry leaf", "blueberry rust", "broccoli downy mildew", "broccoli leaf",
    "cabbage alternaria leaf spot", "cabbage leaf", "carrot cavity spot", "cauliflower alternaria leaf spot",
    "cauliflower leaf", "celery anthracnose", "celery early blight", "celery leaf", "cherry leaf",
    "cherry leaf spot", "cherry powdery mildew", "citrus canker", "citrus greening disease", "coffee leaf",
    "coffee leaf rust", "corn gray leaf spot", "corn leaf", "corn northern leaf blight", "corn rust",
    "corn smut", "cucumber angular leaf spot", "cucumber bacterial wilt", "cucumber leaf",
    "cucumber powdery mildew", "eggplant cercospora leaf spot", "eggplant leaf", "garlic leaf",
    "garlic leaf blight", "garlic rust", "ginger leaf", "ginger leaf spot", "ginger sheath blight",
    "grape black rot", "grape downy mildew", "grape leaf", "grape leaf spot", "grapevine leafroll disease",
    "lettuce downy mildew", "lettuce leaf", "lettuce mosaic virus", "maple leaf", "maple tar spot",
    "peach leaf", "peach leaf curl", "plum leaf", "plum pocket disease", "potato early blight",
    "potato late blight", "potato leaf", "raspberry leaf", "rice blast", "rice leaf", "rice sheath blight",
    "soybean leaf", "squash leaf", "squash powdery mildew", "strawberry anthracnose", "strawberry leaf",
    "strawberry leaf scorch", "tobacco leaf", "tobacco mosaic virus", "tomato bacterial leaf spot",
    "tomato early blight", "tomato late blight", "tomato leaf", "tomato leaf mold", "tomato mosaic virus",
    "tomato septoria leaf spot", "tomato yellow leaf curl virus", "zucchini yellow mosaic virus"])
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    pred_index = np.where(classes == preds)[0][0]
    class_dir = f"class_{pred_index:02d}"

    bucket_name = "production"
    root, ext = os.path.splitext(img_path)
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = f"{class_dir}/{prediction_id}{ext}"
    
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(f, 
            bucket_name, 
            s3_key, 
            ExtraArgs={'ContentType': content_type}
            )

    # tag the object with predicted class and confidence
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=s3_key,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_class', 'Value': preds},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )

# For making requests to FastAPI
def request_fastapi(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}
        
        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", json=payload)
        response.raise_for_status()
        
        result = response.json()
        predicted_class = result.get("prediction")
        probability = result.get("probability")
        
        return predicted_class, probability

    except Exception as e:
        print(f"Error during inference: {e}")  
        return None, None  

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        img_path = os.path.join(app.instance_path, 'uploads', secure_filename(f.filename))

        # create a unique filename for the image
        prediction_id = str(uuid.uuid4())
        
        preds, probs = request_fastapi(img_path)
        if preds:
            executor.submit(upload_production_bucket, img_path, preds, probs, prediction_id) # New! upload production image to MinIO bucket
            return f'<button type="button" class="btn btn-info btn-sm">{preds}</button>'

    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    preds, probs = request_fastapi(img_path)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)  # Security PORT setting!
