import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

# --- CONFIGURACIÓN ---
TRAIN_DIR = 'VehiclesDatasetDataAugmentation/Train'
VAL_DIR = 'VehiclesDatasetDataAugmentation/Validation'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 20
EPOCHS = 50
MODEL_PATH = 'models/my_model.h5'


# --- PREPROCESAMIENTO DE IMÁGENES ---
def resize_images(directories, size=IMAGE_SIZE):
    for directory in directories:
        image_files = glob.glob(os.path.join(directory, '*.jpg'))
        for image_file in image_files:
            with Image.open(image_file) as img:
                img_resized = img.resize(size)
                img_resized.save(image_file)
        print(f"[INFO] {len(image_files)} imágenes redimensionadas en {directory}")


# --- RENOMBRAR IMÁGENES ---
def rename_images(directories_with_prefixes):
    for dir_path, prefix in directories_with_prefixes.items():
        if not os.path.exists(dir_path):
            continue
        image_files = sorted(os.listdir(dir_path))
        for idx, image in enumerate(image_files, 1):
            old_path = os.path.join(dir_path, image)
            new_name = f"{prefix}{idx}.jpg"
            new_path = os.path.join(dir_path, new_name)
            os.rename(old_path, new_path)
    print("[INFO] Imágenes renombradas correctamente")


# --- ENTRENAMIENTO DEL MODELO ---
def train_model():
    train_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    )

    num_classes = len(train_data.class_indices)

    # Modelo CNN
    i = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = Conv2D(256, (2, 2), strides=2, activation='relu')(i)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (2, 2), strides=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (2, 2), strides=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(i, x)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    image_files = glob.glob(os.path.join(TRAIN_DIR, '*/*.jpg'))
    val_files = glob.glob(os.path.join(VAL_DIR, '*/*.jpg'))

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        steps_per_epoch=int(np.ceil(len(image_files) / BATCH_SIZE)),
        validation_steps=int(np.ceil(len(val_files) / BATCH_SIZE)),
    )

    model.save(MODEL_PATH)
    print(f"[INFO] Modelo guardado en: {MODEL_PATH}")
    return history


# --- GRAFICAR MÉTRICAS ---
def plot_history(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Accuracy')
    plt.show()


# --- PREDICCIÓN DE UNA IMAGEN ---
def predict_image(img_path, model_path=MODEL_PATH):
    model = load_model(model_path)
    img = Image.open(img_path).resize(IMAGE_SIZE)
    img_array = np.expand_dims(preprocess_input(np.array(img)), axis=0)
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    print(f"[INFO] Predicción: Clase {class_index}")
    return class_index


if __name__ == "__main__":
    # Opcional: Preprocesamiento inicial
    # resize_images([...])
    # rename_images({...})
    history = train_model()
    plot_history(history)
    # predict_image('VehiclesDatasetDataAugmentation/Train/Bus/Bus15.jpg')
