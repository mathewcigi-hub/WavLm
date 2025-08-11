import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Proposed_model import UNet_model


# Paths
#DATASET_PATH = r"H:\HTIC\PulseDB\Data_PPG_ABP_ECG\Samples_30k_8_Rpeaks\PPG_ABP_ECG_Rpeaks_JSON_1D_Normalized_ABP_20_200"
DATASET_PATH = r"H:\HTIC\PulseDB\Data_PPG_ABP_ECG\PPG_ABP_ECG_Rpeaks_heartpy_JSON_filtered_20_200_only_50k"
FINAL_MODEL_PATH = r"D:\MS_IITM\Research_work\PPG_to_ECG_ABP\U_net\Model3_U_net_dynamic_rpeaks_triangle\final_model\final_model.h5"
BEST_MODEL_PATH = r"D:\MS_IITM\Research_work\PPG_to_ECG_ABP\U_net\Model3_U_net_dynamic_rpeaks_triangle\model_sample\best_model.h5"
EPOCH_FILE = r"D:\MS_IITM\Research_work\PPG_to_ECG_ABP\U_net\Model3_U_net_dynamic_rpeaks_triangle\model_sample\current_epoch.txt"



def rpeak_weighted_mse(y_true, y_pred, rpeak_mask):
    return tf.reduce_mean(rpeak_mask * tf.square(y_true - y_pred))


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, file_list, batch_size, input_size, shuffle=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.file_list = file_list
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]
        ppg_batch, ecg_batch, abp_batch, rpeak_batch = [], [], [], []

        for file_name in batch_files:
            with open(os.path.join(self.dataset_path, file_name), 'r') as file:
                data = json.load(file)

                if len(data['PPG_derivatives_values']) == 1250:
                    ppg = np.array(data['PPG_derivatives_values'], dtype=np.float32)
                    ecg = np.array(data['ECG_f_values'], dtype=np.float32)
                    abp = np.array(data['abp_raw_values'], dtype=np.float32)
                    #rpeak_mask = np.ones(1250, dtype=np.float32)
                    #rpeak_mask = np.full_like(ecg, 2, dtype=np.float32) # Setting the base to 2 and gaussian with max as 10
                    rr_intervals = np.diff(data['ECG_Rpeak'])
                    avg_rr = int(np.mean(rr_intervals))
                    std = int(avg_rr / 15) 

                    triangle_signal = np.ones_like(ecg)

                    # Define segment widths (in samples)
                    
                    w1 = 4 * std     # wide base for 1→5
                    w2 = std         # narrow base for 5→15
                    total_width = w1 + w2  # half triangle
                    full_width = 2 * total_width + 1  # full triangle length

                    for r in data['ECG_Rpeak']:
                        start = r - total_width
                        end = r + total_width + 1

                        if start >= 0 and end <= len(triangle_signal):
                            # Construct triangle: 1 → 10 → 15 → 10 → 1
                            rise1 = np.linspace(1.0, 5.0, w1 + 1)
                            rise2 = np.linspace(5.0, 15.0, w2 + 1)[1:]  # skip duplicate 10
                            fall2 = np.linspace(15.0, 5.0, w2 + 1)[1:]
                            fall1 = np.linspace(5.0, 1.0, w1 + 1)[1:]
                            triangle = np.concatenate((rise1, rise2, fall2, fall1))  # final length = full_width

                            triangle_signal[start:end] = np.maximum(triangle_signal[start:end], triangle)



                    ppg_batch.append(ppg)
                    ecg_batch.append(ecg)
                    abp_batch.append(abp)
                    rpeak_batch.append(triangle_signal)

        ppg_batch = np.array(ppg_batch)[..., np.newaxis] if len(ppg_batch[0].shape) == 1 else np.array(ppg_batch)
        ecg_batch = np.array(ecg_batch)[..., np.newaxis]
        abp_batch = np.array(abp_batch)[..., np.newaxis]
        rpeak_batch = np.array(rpeak_batch)[..., np.newaxis]

        return [ppg_batch, rpeak_batch], [ecg_batch, abp_batch]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def save_epoch(epoch):
    with open(EPOCH_FILE, 'w') as f:
        f.write(str(epoch))

def load_epoch():
    if os.path.exists(EPOCH_FILE):
        with open(EPOCH_FILE, 'r') as f:
            return int(f.read())
    return 0

class EpochCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        save_epoch(epoch + 1)


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        (ppg, rpeak_mask), (y_ecg, y_abp) = data
        with tf.GradientTape() as tape:
            pred_ecg, pred_abp = self([ppg, rpeak_mask], training=True)
            ecg_loss = rpeak_weighted_mse(y_ecg, pred_ecg, rpeak_mask)
            #abp_loss = tf.reduce_mean(tf.abs(y_abp - pred_abp))
            abp_loss = tf.reduce_mean(tf.square(y_abp - pred_abp))
            total_loss = ecg_loss + abp_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_ecg, pred_ecg)

        return {"loss": total_loss, "ecg_loss": ecg_loss, "abp_loss": abp_loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        (ppg, rpeak_mask), (y_ecg, y_abp) = data
        pred_ecg, pred_abp = self([ppg, rpeak_mask], training=False)
        ecg_loss = rpeak_weighted_mse(y_ecg, pred_ecg, rpeak_mask)
        abp_loss = tf.reduce_mean(tf.abs(y_abp - pred_abp))
        total_loss = ecg_loss + abp_loss
        self.compiled_metrics.update_state(y_ecg, pred_ecg)

        return {"loss": total_loss, "ecg_loss": ecg_loss, "abp_loss": abp_loss, **{m.name: m.result() for m in self.metrics}}


all_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.json')]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_generator = DataGenerator(DATASET_PATH, train_files, batch_size=16, input_size=1250)
val_generator = DataGenerator(DATASET_PATH, val_files, batch_size=16, input_size=1250, shuffle=False)


print("Building model...")
ppg_input = tf.keras.Input(shape=(1250, 3), name="PPG_derivatives_values")
rpeak_input = tf.keras.Input(shape=(1250, 1), name="RPeak_Mask")
ec_output, ab_output = UNet_model()(ppg_input)
model = CustomModel(inputs=[ppg_input, rpeak_input], outputs=[ec_output, ab_output])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), metrics=["mse", "mae"])


callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor="val_loss", mode="min", save_weights_only=True),
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1),
    EpochCheckpoint()
]

os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(EPOCH_FILE), exist_ok=True)


initial_epoch = load_epoch()
print("Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=400,
    initial_epoch=initial_epoch,
    callbacks=callbacks, 
    verbose =2
)


os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
print("Saving final model...")
model.save(FINAL_MODEL_PATH)


print("Evaluating model...")
eval_result = model.evaluate(val_generator, return_dict=True)
print("\nEvaluation Results:")
for k, v in eval_result.items():
    print(f"{k}: {v:.4f}")

