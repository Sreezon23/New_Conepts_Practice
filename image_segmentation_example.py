import numpy as np
import tensorflow as tf


IMG_SIZE = 64
BATCH_SIZE = 8
OUTPUT_CLASSES = 3


def create_synthetic_dataset(num_samples):
    images = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    masks = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.int32)

    for i in range(num_samples):
        image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.int32)

        if i % 2 == 0:
            y1, x1 = 16, 16
            y2, x2 = 48, 48
            image[y1:y2, x1:x2, :] = [0.2, 0.6, 0.9]
            mask[y1:y2, x1:x2, 0] = 1
        else:
            center = (IMG_SIZE // 2, IMG_SIZE // 2)
            radius = 16
            for y in range(IMG_SIZE):
                for x in range(IMG_SIZE):
                    if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2:
                        image[y, x, :] = [0.9, 0.4, 0.2]
                        mask[y, x, 0] = 2

        # Add a soft background pattern
        image += np.linspace(0.0, 0.2, IMG_SIZE).reshape(IMG_SIZE, 1, 1)
        image = np.clip(image, 0.0, 1.0)

        images[i] = image
        masks[i] = mask

    return images, masks


def get_datasets():
    images, masks = create_synthetic_dataset(40)
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=40, seed=42)

    train_size = 32
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    train_batches = (
        train_dataset
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_batches = (
        test_dataset
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_batches, test_batches


def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation="relu")(x)
    return x


def build_simple_unet():
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64)

    u2 = tf.keras.layers.UpSampling2D((2, 2))(c3)
    concat2 = tf.keras.layers.Concatenate()([u2, c2])
    c4 = conv_block(concat2, 32)

    u3 = tf.keras.layers.UpSampling2D((2, 2))(c4)
    concat3 = tf.keras.layers.Concatenate()([u3, c1])
    c5 = conv_block(concat3, 16)

    outputs = tf.keras.layers.Conv2D(OUTPUT_CLASSES, 1, padding="same")(c5)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def main():
    train_batches, test_batches = get_datasets()

    model = build_simple_unet()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print(model.summary())

    model.fit(
        train_batches,
        epochs=5,
        validation_data=test_batches,
    )

    for images, masks in test_batches.take(1):
        predictions = model.predict(images)
        sample_mask = create_mask(predictions[0])
        print("Sample prediction mask shape:", sample_mask.shape)
        break

    model.save("segmentation_model.h5")
    print("Saved model to segmentation_model.h5")


if __name__ == "__main__":
    main()
