import pandas as pd
import tensorflow as tf
import numpy as np

IMG_HEIGHT, IMG_WIDTH = 160, 160

classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy',
           'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 
           'sunflower', 'tulip']

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def decode_img(img, img_height, img_width):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def get_images_labels(df, img_height, img_width):
    test_images = []
    test_labels = []
    paths = []
    for _, row in df.iterrows():
        img_path = row['image_path']
        img = tf.io.read_file(img_path)
        img = decode_img(img, img_height, img_width) / 255.0  # Normalize
        test_images.append(img)
        test_labels.append(classes.index(row['label']))  # Convert label to class index
        paths.append(img_path)  # Keep track of image paths for reporting
    return np.array(test_images), np.array(test_labels), paths

# Define paths directly
model_path = 'my_model_mobile_net.keras'
test_csv = 'flowers_test.csv'

# Load test data
test_df = pd.read_csv(test_csv)
test_df.columns = test_df.columns.str.strip()  # Clean up any whitespace in column names
test_df['label'] = test_df['label'].str.strip()
test_images, test_labels, test_paths = get_images_labels(test_df, IMG_HEIGHT, IMG_WIDTH)

# Load model
my_model = load_model(model_path)

# Track misclassifications
misclassified_images = []

# Predict and evaluate for each image
predictions = my_model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Loop over each prediction to compare with true labels
for i, (true_label, predicted_label) in enumerate(zip(test_labels, predicted_labels)):
    if true_label != predicted_label:  # Check if the prediction is wrong
        misclassified_images.append({
            'image_path': test_paths[i],
            'true_label': classes[true_label],
            'predicted_label': classes[predicted_label]
        })

# Display results
if misclassified_images:
    print("Misclassified images:")
    for item in misclassified_images:
        print(f"Image: {item['image_path']}, True Label: {item['true_label']}, Predicted Label: {item['predicted_label']}")
else:
    print("All images classified correctly!")

# Overall accuracy
loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
print(f'Test model, accuracy: {acc * 100:.2f}%')