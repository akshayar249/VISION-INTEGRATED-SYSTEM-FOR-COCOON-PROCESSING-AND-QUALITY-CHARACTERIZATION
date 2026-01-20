import os
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
IMG_SIZE = 128
LATENT_DIM = 16
BATCH_SIZE = 16
EPOCHS = 50
N_GENERATE = 300
ZIP_FILE = "dataset2.0.zip"
DATA_DIR = "cocoon_images"
OUTPUT_DIR = "vae_cocoon_dataset"
FINAL_ZIP = "vae_cocoon_dataset.zip"

# ---------------- UNZIP ----------------
if not os.path.exists(DATA_DIR):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
print(f"Images extracted to {DATA_DIR}")

# ---------------- LOAD DATA ----------------
def load_images(folder):
    exts = {".jpg", ".jpeg", ".png"}
    files = [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    images = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img) / 255.0)
    return np.array(images, dtype="float32"), files

x_data, file_paths = load_images(DATA_DIR)
dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(len(x_data)).batch(BATCH_SIZE)

# ---------------- VAE ----------------
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
encoder_inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense((IMG_SIZE // 4) * (IMG_SIZE // 4) * 64, activation="relu")(latent_inputs)
x = layers.Reshape((IMG_SIZE // 4, IMG_SIZE // 4, 64))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE class
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "recon": recon_loss, "kl": kl_loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(dataset, epochs=EPOCHS)

# ---------------- OUTPUT FOLDERS ----------------
original_dir = os.path.join(OUTPUT_DIR, "original")
recon_dir = os.path.join(OUTPUT_DIR, "reconstructed")
gen_dir = os.path.join(OUTPUT_DIR, "generated")
os.makedirs(original_dir, exist_ok=True)
os.makedirs(recon_dir, exist_ok=True)
os.makedirs(gen_dir, exist_ok=True)

# save original images
for f in file_paths:
    img_name = Path(f).name
    Image.open(f).save(os.path.join(original_dir, img_name))

# save reconstructed images
z_mean, z_log_var, z = encoder.predict(x_data)
reconstructions = decoder.predict(z)
for i, img in enumerate(reconstructions):
    img_pil = Image.fromarray((img*255).astype("uint8"))
    img_pil.save(os.path.join(recon_dir, f"recon_{i+1:03d}.png"))

# generate new synthetic images
random_z = np.random.normal(size=(N_GENERATE, LATENT_DIM))
gen_images = decoder.predict(random_z)
for i, img in enumerate(gen_images):
    img_pil = Image.fromarray((img*255).astype("uint8"))
    img_pil.save(os.path.join(gen_dir, f"gen_{i+1:03d}.png"))

# ---------------- ZIP EVERYTHING ----------------
with zipfile.ZipFile(FINAL_ZIP, 'w') as zipf:
    for folder in [original_dir, recon_dir, gen_dir]:
        for file in Path(folder).rglob("*.*"):
            zipf.write(file, arcname=os.path.join(Path(folder).name, file.name))

print(f"All done! Final dataset saved as {FINAL_ZIP}")
