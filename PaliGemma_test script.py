import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import sentencepiece
import jax
import jax.numpy as jnp
import ml_collections
import functools
import html
import io
import base64
import warnings
from IPython.core.display import display, HTML

# Suppress TensorFlow GPU/TPU usage
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

# Append big_vision to python path
if "big_vision_repo" not in sys.path:
    sys.path.append("big_vision_repo")

# Import big_vision modules
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
import big_vision.utils
import big_vision.sharding

# Print JAX environment
backend = jax.extend.backend.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")

# Define paths
MODEL_PATH = "/kaggle/working/PaliGemma_Fine_Tune_Flickr8k/paligemma_flickr8k.params.f16.npz"
TOKENIZER_PATH = "./paligemma_tokenizer.model"
DATA_DIR = "/kaggle/input/flickr8k"
IMAGES_DIR = os.path.join(DATA_DIR, "Images")

# Custom load_params implementation
def load_params(path):
    """Load JAX parameters from a file."""
    try:
        with np.load(path, allow_pickle=True) as data:
            # Check if it's saved in the "param_X" format
            if 'treedef' in data:
                import pickle
                treedef = pickle.loads(data['treedef'][0])
                params_flat = [data[f'param_{i}'] for i in range(len(data) - 1)]
                return jax.tree.unflatten(treedef, params_flat)
            else:
                # Fall back to traditional NPZ loading
                return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try fallback method
        print("Trying fallback loading method...")
        try:
            # Load tree definition
            with open(f"{path}.treedef", "wb") as f:
                import pickle
                treedef = pickle.load(f)
                
            # Load parameters
            params_flat = []
            i = 0
            while os.path.exists(f"{path}.{i}.npy"):
                params_flat.append(np.load(f"{path}.{i}.npy"))
                i += 1
                
            return jax.tree.unflatten(treedef, params_flat)
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            raise

# Define model
model_config = ml_collections.FrozenConfigDict({
    "llm": {"vocab_size": 257_152},
    "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
})
model = paligemma.Model(**model_config)
tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

# Load fine-tuned parameters
print(f"Loading fine-tuned model from {MODEL_PATH}")
try:
    # First try using the paligemma.load function
    params = paligemma.load(None, MODEL_PATH, model_config)
    print("Successfully loaded model with paligemma.load")
except Exception as e:
    print(f"Error loading model with paligemma.load: {e}")
    print("Falling back to custom load function")
    params = load_params(MODEL_PATH)
    print("Successfully loaded model with custom loader")

# Define decode function
decode_fn = predict_fns.get_all(model)['decode']
decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

# Create trainable params mask (for completeness)
def is_trainable_param(name, param):
    if name.startswith("llm/layers/attn/"): return True
    if name.startswith("llm/"): return False
    if name.startswith("img/"): return False
    raise ValueError(f"Unexpected param name {name}")
trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

# Shard parameters
mesh = jax.sharding.Mesh(jax.devices(), ("data"))
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
params_sharding = big_vision.sharding.infer_sharding(
    params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh
)

# Ignore unusable donated buffers warning
warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

@functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def maybe_cast_to_f32(params, trainable):
    return jax.tree.map(lambda p, m: p.astype(jnp.float32) if m else p, params, trainable)

# Load parameters with sharding
params, treedef = jax.tree.flatten(params)
sharding_leaves = jax.tree.leaves(params_sharding)
trainable_leaves = jax.tree.leaves(trainable_mask)
for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
    params[idx] = big_vision.utils.reshard(params[idx], sharding)
    params[idx] = maybe_cast_to_f32(params[idx], trainable)
    params[idx].block_until_ready()
params = jax.tree.unflatten(treedef, params)

print("Model parameters loaded and resharded successfully")

# Image preprocessing
def preprocess_image(image, size=224):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)
    image = image[..., :3]
    assert image.shape[-1] == 3
    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255] -> [-1, 1]

# Token preprocessing
def preprocess_tokens(prefix, suffix=None, seqlen=None):
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)
    mask_loss = [0] * len(tokens)
    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)
        tokens += suffix
        mask_ar += [1] * len(suffix)
        mask_loss += [1] * len(suffix)
    mask_input = [1] * len(tokens)
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding
    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

# Token postprocessing
def postprocess_tokens(tokens):
    tokens = tokens.tolist()
    try:
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)

SEQLEN = 128

# Test on new images
def test_image_captioning(image_paths, batch_size=1, sampler="greedy"):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[i:i+batch_size]
        batch_examples = []
        
        for image_path in batch_image_paths:
            try:
                image = Image.open(image_path)
                image = preprocess_image(image)
                prefix = "caption en"
                tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
                batch_examples.append({
                    "image": np.asarray(image),
                    "text": np.asarray(tokens),
                    "mask_ar": np.asarray(mask_ar),
                    "mask_input": np.asarray(mask_input),
                    "_mask": np.array(True)  # Valid example
                })
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        # Pad to batch size if needed
        while len(batch_examples) % batch_size:
            if len(batch_examples) > 0:
                batch_examples.append(dict(batch_examples[-1]))
                batch_examples[-1]["_mask"] = np.array(False)  # Mark as padding
            else:
                # Handle case where all images in batch failed
                break
        
        if not batch_examples:
            continue
            
        # Stack batch and move to device
        batch = jax.tree.map(lambda *x: np.stack(x), *batch_examples)
        batch = big_vision.utils.reshard(batch, data_sharding)
        
        # Generate captions
        tokens = decode({"params": params}, batch=batch, max_decode_len=SEQLEN, sampler=sampler)
        tokens, mask = jax.device_get((tokens, batch["_mask"]))
        tokens = tokens[mask]
        
        # Post-process captions
        for j, token_seq in enumerate(tokens):
            caption = postprocess_tokens(token_seq)
            caption = caption[len("caption en\n"):] if caption.startswith("caption en\n") else caption
            original_image = Image.open(batch_image_paths[j])
            results.append((original_image, batch_image_paths[j], caption))
    
    return results

# Render results 
def render_inline(image, resize=(128, 128)):
    image = image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format='jpeg')
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

def render_test_results(results):
    html_output = "<div style='display: flex; flex-wrap: wrap;'>"
    for image, image_path, caption in results:
        filename = os.path.basename(image_path)
        html_output += f"""
            <div style="width: 200px; margin: 10px; text-align: center;">
                <img style="width:180px; height:auto; max-height:180px; object-fit:contain;" 
                     src="{render_inline(image, resize=(180, 180))}" />
                <p style="font-size:12px; color:#666;">{filename}</p>
                <p style="font-size:14px;">{html.escape(caption)}</p>
            </div>
        """
    html_output += "</div>"
    display(HTML(html_output))

# Get list of test images
import random
all_images = [os.path.join(IMAGES_DIR, fname) for fname in os.listdir(IMAGES_DIR) if fname.endswith(('.jpg', '.jpeg', '.png'))]
test_images = random.sample(all_images, min(10, len(all_images)))
print(f"Testing on {len(test_images)} random images")

# Run the test
print("Generating captions for test images...")
test_results = test_image_captioning(test_images, batch_size=2)

# Display results
print(f"Results for {len(test_results)} images:")
render_test_results(test_results)
