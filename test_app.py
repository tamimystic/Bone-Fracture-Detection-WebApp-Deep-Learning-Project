import io
from flask import Flask
from werkzeug.datastructures import FileStorage
from app import app
import json

client = app.test_client()

# Create a dummy image
from PIL import Image
import numpy as np

img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

# Simulate upload
data = {
    'image': (img_byte_arr, 'test.png')
}
response = client.post('/predict', data=data, content_type='multipart/form-data')

print("Status Code:", response.status_code)
print("Response Data:", response.data.decode('utf-8'))
