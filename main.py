import tensorflow as tf # The main thing handling all the ai stuff
import keras
import numpy as np # NumPy is used for some array manipulations and conversions.
from os.path import exists
from datasets import load_dataset

try:
    if tf.config.experimental.list_physical_devices("GPU"):
        tf.config.set_visible_devices(tf.config.experimental.list_physical_devices("GPU")[0], "GPU")
        print("✅ GPU device found and configured for the GPU. yay")
    else:
        print("❌ GPU device not found sad oof aaaaaaaaaa, training will use the CPU.")
except Exception as e:
    print(f"An error occurred during MPS configuration: {e}")

keras.mixed_precision.set_global_policy('mixed_float16')

SEQ_LENGTH = 100
BATCH_SIZE = 140
BUFFER_SIZE = 10000
vocab_set = set()
ds = load_dataset("kmfoda/booksum", "default", split="train")

print("Dataset loaded!")

def build_model(vocab_list):
    """Builds the AI model using the provided vocabulary list."""
    inputs = keras.layers.Input(shape=(SEQ_LENGTH,))
    x = keras.layers.Embedding(input_dim=len(vocab_list), output_dim=256)(inputs)
    x = keras.layers.LSTM(512, return_sequences=True)(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(len(vocab_list))(x)
    model = keras.models.Model(inputs, outputs)
    return model

def generate_text(model, prompt, char2idx, idx2char, temperature=0.4, top_k=10, n_chars=400):
    PAD_ID = char2idx['\0']
    vocab_size = len(idx2char)
    k = min(top_k, vocab_size)
    encoded_prompt = np.array([char2idx[c] for c in prompt], dtype=np.int32)
    tokens = list(encoded_prompt)

    for _ in range(n_chars):
        if len(tokens) > 100:
            context_tokens = tokens[-SEQ_LENGTH:]
        elif len(tokens) < 100: # left pad
            context_tokens = [PAD_ID] * (SEQ_LENGTH-len(tokens)) + tokens
        else: 
            context_tokens = tokens
        context = np.array(context_tokens, dtype=int)[None, :]

        logits = model.predict(context, verbose=0)[0, -1]
        logits /= temperature
        top_k_indicies = np.argpartition(logits, -k)[-k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_indicies] = logits[top_k_indicies]
        probs = tf.nn.softmax(mask).numpy() # type: ignore
        next_token = np.random.choice(vocab_size, p=probs)
        tokens.append(next_token)

    generated = ''.join(idx2char[t] for t in tokens[len(encoded_prompt):])
    return prompt + generated

def main():
    """Main function to train the AI model. running coming soon™. """

    full_text = ""
    vocab_list = []
    if exists("vocab_set.txt"):
        with open("vocab_set.txt", "r") as f:
            full_text = f.read()
            vocab_list = sorted(set(letter for letter in full_text))
            print("Vocabulary list already exists. Skipping vocabulary generation.")
    else:
        for ex in ds:
            text = ex.get("content") or ex.get("chapter") #type: ignore
            if text is None:
                continue
            vocab_set.update(letter for letter in text)
            full_text += text

        vocab_list = sorted(vocab_set)
        with open("vocab_set.txt", "w") as f:
            f.write(full_text)
            print("Vocabulary list generated and saved to vocab_set.txt.")

    if '\0' not in vocab_list:
        vocab_list.append('\0')
        
    char2idx = {c: i for i, c in enumerate(vocab_list)}
    idx2char = np.array(vocab_list)

    print(len(full_text), "characters in the dataset.")

    encoded = np.array([char2idx[c] for c in full_text], dtype=np.int32)

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

    validation_dataset = tf.data.Dataset.from_tensor_slices(val_encoded) # no error which is really suprizing but how do you get this?
    validation_windowed_dataset = validation_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
    val_ds = (
        validation_windowed_dataset.map(lambda x: (x[:-1], x[1:]))
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(vocab_list)
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy")
    while True:
        print("Generate or train? (g/t)")
        choice = input().strip().lower()
        if choice == 'g':
            try:
                model.load_weights("model_checkpoint.weights.h5")  # Load weights if they exist
            except Exception as e:
                print("Could not load weights:", e)  # If the file doesn't exist, create
            print("What is your prompt? (note, it continues the prompt; it doesnt respond to the question)")
            input_prompt = input()
            allowed = set(char2idx)
            print('Generating text...')
            print(generate_text(model, "".join(c for c in input_prompt if c in allowed), char2idx=char2idx, idx2char=idx2char))
        elif choice == 't':
            try:
                model.load_weights("model_checkpoint.weights.h5")
            except Exception as e:
                pass
            model.fit(
                train_ds,
                validation_data=val_ds,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        save_weights_only=True,
                        monitor="val_loss",
                        mode="min",
                        filepath="model_checkpoint.weights.h5"
                    ),
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.TensorBoard(log_dir="logs"),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=1, verbose=1),
                ],
                epochs=500
            )

        
        else:
            print("Invalid choice. Please enter 'g' to generate text or 't' to train the model.")
    

if __name__ == "__main__":
    main()
