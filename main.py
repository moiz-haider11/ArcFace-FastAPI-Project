from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
from utils import read_image, detect_face, preprocess_face

app = FastAPI()

# Load ONNX models
arcface = ort.InferenceSession("models/arcface.onnx")


@app.post("/encode")
async def encode(file: UploadFile = File(...)):
    img = read_image(file.file)

    # Detect face (simple full image)
    box = detect_face(img)
    x1, y1, x2, y2 = box
    face = img[y1:y2, x1:x2]

    # Preprocess face
    face_input = preprocess_face(face)

    # Generate embedding
    embedding = arcface.run(None, {"input": face_input})[0]
    embedding = embedding[0].tolist()

    return {
        "status": "success",
        "embedding": embedding
    }
