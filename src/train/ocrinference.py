import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()


def decode_prediction(prediction):
    # This is a simple greedy decoding. You might want to implement
    # beam search or other more advanced decoding methods.
    chars = "".join([chr(i) for i in range(32, 127)])  # ASCII 32-126
    decoded = []
    for i in range(prediction.shape[1]):
        index = np.argmax(prediction[:, i])
        if index > 0:  # Ignore blank label
            decoded.append(chars[index - 1])
    return "".join(decoded)

"""
Important considerations:
1.This basic example uses greedy decoding, which may not be optimal. 
More advanced decoding methods like beam search could improve results.
2.The model expects single-line text images. 
For multi-line text, you'd need to implement line detection and process each line separately.
3.The model's performance depends on how well it was trained and how similar the input images are to the training data.
4. For production use, you might want to add error handling,
 logging, and possibly integrate with a spell-checker for improved accuracy.
5. If processing multiple images, consider batching them for improved efficiency.
Remember, the effectiveness of this OCR model depends on various factors including the quality of training data, the 
complexity of the OCR task, and how well the input images match the characteristics of the training data. 
You may need to fine-tune the model or use more advanced techniques for optimal performance in specific use cases.
"""

# Load the ONNX model
session = onnxruntime.InferenceSession("ocr_model.onnx")

# Prepare input image
image_path = "./test/graph.jpg"
input_data = preprocess_image(image_path)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
prediction = session.run([output_name], {input_name: input_data})[0]

# Decode the prediction
text = decode_prediction(prediction[0])
print(f"Recognized text: {text}")
