import pandas as pd
import argparse
import tensorflow as tf
import os
# import cv2

# Note that you can save models in different formats. Some format needs to save/load model and weight separately.
# Some saves the whole thing together. So, for your set up you might need to save and load differently.

def load_model_weights(model, weights = None):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model

def get_images_labels(df, classes, img_height, img_width):
    test_images = []
    test_labels = []
    
    class_to_index = {class_name: i for i, class_name in enumerate(classes)}
    
    for index, row in df.iterrows():
        label = row['label']
        image_path = row['image_path']
        
        img = tf.io.read_file(image_path)
        img = decode_img(img, img_height, img_width)
        
        # Append the processed image and label (as an integer)
        test_images.append(img)
        test_labels.append(class_to_index[label])
    
    # Convert lists to TensorFlow Tensors
    test_images = tf.convert_to_tensor(test_images)
    test_labels = tf.convert_to_tensor(test_labels)
    
    return test_images, test_labels


def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width]) / 255.0  # Normalize the image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Test")
    parser.add_argument('--model', type=str, default='my_model_mobile_net.keras', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='Weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')
    parser.add_argument('--img_height', type=int, default=160, help='Image height for resizing')
    parser.add_argument('--img_width', type=int, default=160, help='Image width for resizing')

    args = parser.parse_args()
    model_path = args.model
    weights_path = args.weights
    test_csv = args.test_csv
    img_height = args.img_height
    img_width = args.img_width

    # Load test data
    test_df = pd.read_csv(test_csv)
    classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy',
               'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip']
    
    # Clean up column names
    test_df.columns = test_df.columns.str.strip()
    test_df['label'] = test_df['label'].str.strip()
    # Load test images and labels
    test_images, test_labels = get_images_labels(test_df, classes, img_height, img_width)
    
    # Load the model
    my_model = load_model_weights(model_path, weights_path)
    
    # Evaluate the model on the test set
    loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test model, accuracy: {acc * 100:.2f}%')


    