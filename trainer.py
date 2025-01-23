import ultralytics

# Load the model
model = ultralytics.YOLOv8('yolov8n.yaml')

# Load the data configuration
data = 'data.yaml'

# Train the model
model.train(data=data, epochs=100, imgsz=640, batch_size=16, workers=4)

# Save the trained model
model.save('trained_model')

# Evaluate the model
# Evaluate the Model
results = model.val(data=data)
print(results)

# Plot The Results
results.plot(show=True, save=True, path='results')