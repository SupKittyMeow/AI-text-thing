import tensorflow as tf # The main thing handling all the ai stuff
import keras # keras does... something idk magic ig
import numpy as np # numpy is used for some array manipulations and conversions.
from os.path import exists # to see if a file exists i really dont need this comment
from datasets import load_dataset # to load the dataset from huggingface
import sentencepiece as spm # type: ignore | tokenizer 

if (not exists('booksum_sp.model')):
    spm.SentencePieceTrainer.train(
        input='vocab_set.txt',
        model_prefix='booksum_sp',
        vocab_size=16000,
        model_type='bpe',
        character_coverage=1.0
    )

sp = spm.SentencePieceProcessor(model_file='booksum_sp.model')

try:
    if tf.config.experimental.list_physical_devices("GPU"):
        tf.config.set_visible_devices(tf.config.experimental.list_physical_devices("GPU")[0], "GPU")
        print("✅ GPU device found and configured for the GPU. yay")
    else:
        print("❌ GPU device not found sad oof aaaaaaaaaa, training will use the CPU.")
except Exception as e:
    print(f"An error occurred during MPS configuration: {e}")

# keras.mixed_precision.set_global_policy('mixed_float16')

SEQ_LENGTH = 256
BATCH_SIZE = 140
BUFFER_SIZE = 10000
vocab_set = set()
ds = load_dataset("kmfoda/booksum", "default", split="train")

print("Dataset loaded!")

def build_model(vocab_size, logits, x, inputs):
    """Builds the AI model using the provided vocabulary list."""
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(vocab_size)(x)
    model = keras.models.Model(inputs, logits)
    return model

def generate_text(model, prompt, max_len=200, temperature=0.4, top_k=10):
    """Generates text using the AI model."""
    ids = [] for sample in ds:
    for _ in range(max_len):
        window = ids[-SEQ_LENGTH:]
        window = [sp.pad_id()] * (SEQ_LENGTH-len(window)) + window
        logit = model.predict(tf.constant([window]))[0, -1] / temperature
        top = np.argpartition(logit, -top_k)[-top_k:]
        probs = tf.nn.softmax(tf.where(np.isin(range(len(logit)), top), logit, -1e9))
        ids.append(np.random.choice(len(logit), p=probs.numpy()))
    return sp.decode(ids)

def main():
    """Main function to train the AI model"""

    full_text = (sp.encode(ds['text'], out_type=str))
    vocab_list = sp.encode() # todo

    if '\0' not in vocab_list:
        vocab_list.append('\0')

    print(len(full_text), "characters in the dataset.")

    vocab_size = sp.vocab_size()
    inputs = keras.layers.Input((SEQ_LENGTH,))
    x = keras.layers.Embedding(vocab_size, 512)(inputs)
    x = keras.layers.LSTM(1024, return_sequences=True)(x)

    encoded = np.array([sp.encode(text, out_type=int)[c] for c in full_text], dtype=np.int32)

    split_idx = round(0.9 * len(encoded))
    train_encoded = encoded[:split_idx]
    val_encoded = encoded[split_idx:]

    dataset = tf.data.Dataset.from_tensor_slices(train_encoded)
    windowed_dataset = dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
    train_ds = (
        windowed_dataset.map(lambda x: (x[:-1], x[1:]))
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(val_encoded)
    validation_windowed_dataset = validation_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
    val_ds = (
        validation_windowed_dataset.map(lambda x: (x[:-1], x[1:]))
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    logits = keras.layers.Dense(vocab)(x)

    model = build_model(vocab_size, logits, x, inputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy()]) # type: ignore
    while True:
        print("Generate or train? (g/t)")
        choice = input().strip().lower()
        if choice == 'g':
            try:
                model.load_weights("best_loss.weights.h5")  # Load weights if they exist
            except Exception as e:
                print("Could not load weights:", e)  # If the file doesn't exist, create
            print("What is your prompt? (note, it continues the prompt; it doesnt respond to the question)")
            input_prompt = input()
            allowed = set(sp.encode(text, out_type=int))
            print('Generating text...')
            print(generate_text(model, "".join(c for c in input_prompt if c in allowed))) # type: ignore
        elif choice == 't':
            try:
                model.load_weights("best_loss.weights.h5")
            except Exception as e:
                pass

            checkpoint = keras.callbacks.ModelCheckpoint(
                "best_loss.weights.h5",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True
            )

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=1
            )

            model.fit(train_ds.take(1000), validation_data=val_ds,
                callbacks=[
                    checkpoint,
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    reduce_lr
            ], epochs=1)
            
            model.fit(
                train_ds,
                validation_data=val_ds,
                callbacks=[
                    checkpoint,
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    reduce_lr
                ],
                epochs=500
            )

        
        else:
            print("Invalid choice. Please enter 'g' to generate text or 't' to train the model.")
    

if __name__ == "__main__":
    main()
