import torch
import torchvision
import cv2
import numpy as np
import time

# --- Model and Device Setup ---
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained DeepLabV3 model
# This model is trained on the COCO dataset, which includes a 'person' class.
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.to(device)
model.eval() # Set the model to evaluation mode

# COCO class index for 'person' is 15
PERSON_CLASS_INDEX = 15

# --- Pre-processing Transformations ---
# The model expects input images to be normalized in a specific way.
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Video Capture and Background ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Capturing background in 3 seconds... Stay out of the frame.")
time.sleep(3)

# Capture a stable background
ret, background = cap.read()
if not ret:
    print("Error: Could not read background frame.")
    cap.release()
    exit()

background_resized = cv2.resize(background, (640, 480))
print("Background captured successfully.")


# --- Main Loop for Real-time Processing ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))

    # --- ML Inference ---
    # 1. Preprocess the frame
    # Convert frame from BGR (OpenCV) to RGB (PIL/PyTorch)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb_frame)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and send to device

    # 2. Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # 3. Get segmentation predictions
    output_predictions = output.argmax(0)

    # --- Mask Creation ---
    # Create a boolean mask where True corresponds to the 'person' class
    mask = (output_predictions == PERSON_CLASS_INDEX).byte().cpu().numpy()

    # The mask is 2D, but we need a 3D mask to work with 3-channel (BGR) images.
    # We stack the 2D mask 3 times along a new axis.
    mask_3d = np.stack([mask]*3, axis=-1)

    # --- Background Replacement ---
    # Create an inverse mask
    inv_mask_3d = 1 - mask_3d

    # Use the masks to combine the background and the current frame
    # Where the mask is 1 (person), use the background.
    # Where the inverse mask is 1 (not a person), use the current frame.
    foreground = frame * inv_mask_3d
    removed_person = background_resized * mask_3d

    # Combine the two parts
    final_frame = cv2.add(foreground, removed_person.astype(np.uint8))

    cv2.imshow('Invisibility Cloak (ML)', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
