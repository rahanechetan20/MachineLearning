import pandas as pd
import argparse
import tensorflow as tf
import os
# import cv2  # Uncomment if you need OpenCV for additional image processing

def load_model_weights(model_path, weights=None):
    # Load model and weights if necessary
    my_model = tf.keras.models.load_model(model_path)
    my_model.summary()  # Show model architecture
    return my_model

def decode_img(img, img_height, img_width):
    # Decode the image and resize it to the required size
    img = tf.io.decode_jpeg(img, channels=3)  # Decode JPEG image
    img = tf.image.resize(img, [img_height, img_width])  # Resize image to the target dimensions
    img = img / 255.0  # Normalize the image (if the model was trained on normalized images)
    return img

def get_images_labels(df, classes, img_height=224, img_width=224):
    test_images = []
    test_labels = []

    for index, row in df.iterrows():
        label = row['label']  # Assuming label is the column with true class label
        image_path = row['image_path']  # Assuming 'image_path' is the column with image file paths

        # Ensure the label is in the format expected by the model (e.g., class index)
        label_index = classes.index(label)  # Convert the string label to an index
        
        # Read and preprocess the image
        img = tf.io.read_file(image_path)
        img = decode_img(img, img_height, img_width)
        
        test_images.append(img)
        test_labels.append(label_index)

    # Convert to numpy arrays for evaluation
    test_images = tf.stack(test_images)  # Stack the images into a batch
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)  # Convert labels to tensor
    return test_images, test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Test")
    parser.add_argument('--model', type=str, default='my_model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='Weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model_path = args.model
    weights = args.weights
    test_csv = args.test_csv

    # Read the test data CSV
    test_df = pd.read_csv(test_csv)

    # Define the class names    
    classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 
               'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 
               'sunflower', 'tulip']
    # Clean up column names
    test_df.columns = test_df.columns.str.strip()
    test_df['label'] = test_df['label'].str.strip()

    # Get images and labels from the CSV
    test_images, test_labels = get_images_labels(test_df, classes)

    # Load the pre-trained model
    my_model = load_model_weights(model_path)

    # Evaluate the model on the test set
    loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print('Test model accuracy: {:5.2f}%'.format(100 * acc))
