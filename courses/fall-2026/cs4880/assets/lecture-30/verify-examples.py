"""Run the slide code and check mathematical/model interfaces.

Requires numpy and TensorFlow. Reads the exact verbatim blocks from the deck.
CNN: synthetic RGB minibatches test normalization, labels, architecture, and a
training update; does NOT reproduce a 20-epoch CIFAR-10 benchmark. Sequence:
executes every displayed sensor-training epoch. GAN: actual cached/downloaded
MNIST images, forward passes, and one update of each network; not 19 epochs.
Set KERAS_HOME to a writable dataset cache. Supplied plots remain historical.
"""
from pathlib import Path
from unittest.mock import patch
import ast
import contextlib
import io
import json
import re
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
source = next((HERE.parents[1] / 'lectures').glob('lecture-30*.tex')).read_text()
frames = re.findall(r'\\begin\{frame\}\[fragile\]\{([^\n]+)\}(.*?)\\end\{frame\}', source, re.S)
blocks = {title: '\n'.join(re.findall(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}', body, re.S))
          for title, body in frames}
for code in blocks.values():
    ast.parse(code)
tf.keras.utils.set_random_seed(4880)
results = {'tensorflow_version': tf.__version__, 'scope': __doc__, 'passed_code_blocks': []}
env = {}
def run(title, code=None):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(blocks[title] if code is None else code, '<lecture30>', 'exec'), env)
    results['passed_code_blocks'].append(title)
    print('Passed:', title, flush=True)

# Independently verify the actual supplied arithmetic and pooling figures.
patch_pixels = np.array([[1,1,0], [0,1,1], [1,0,1]])
kernel = np.array([[1,0,-1]] * 3)
assert (patch_pixels * kernel).sum() == 0
pool_input = np.array([[1,6,2,2], [3,9,4,1], [0,2,5,1], [8,3,1,7]])
pooled = pool_input.reshape(2,2,2,2).max(axis=(1,3))
assert np.array_equal(pooled, [[9,4], [8,7]])
assert (3*3*3+1)*32 == 896 and (3072+1)*32 == 98336
results['arithmetic'] = {'filter_response': 0, 'max_pool': pooled.tolist()}

# Execute data preparation exactly, substituting explicitly synthetic loader data.
rng = np.random.default_rng(4880)
train_images = rng.integers(0,256,(40,32,32,3),dtype=np.uint8)
test_images = rng.integers(0,256,(10,32,32,3),dtype=np.uint8)
labels = (np.arange(40)%10).reshape(-1,1)
with patch('tensorflow.keras.datasets.cifar10.load_data', return_value=(
        (train_images, labels), (test_images, labels[:10]))):
    run('Original CIFAR-10 Example: Load and Normalize')
assert env['x_train'].shape == (40,32,32,3)
assert 0 <= env['x_train'].min() <= env['x_train'].max() <= 1
assert np.allclose(env['y_train'].sum(axis=1), 1)
run('Original CIFAR-10 CNN: Convolutional Stages')
# Full 20-epoch training remains a lecture demonstration, not this smoke check.
classifier = blocks['Original CIFAR-10 CNN: Classifier and Training']
run('Original CIFAR-10 CNN: Classifier and Training', classifier.split('history =')[0])
cnn = env['model']
assert cnn.count_params() == 620362
assert [tuple(layer.output.shape[1:]) for layer in cnn.layers
        if isinstance(layer, tf.keras.layers.MaxPooling2D)] == [(16,16,32),(8,8,64),(4,4,128)]
before = [v.numpy().copy() for v in cnn.trainable_variables]
metrics = cnn.train_on_batch(env['x_train'][:8], env['y_train'][:8])
assert np.all(np.isfinite(metrics))
assert any(not np.array_equal(a,b.numpy()) for a,b in zip(before,cnn.trainable_variables))
assert np.allclose(cnn(env['x_test'],training=False).numpy().sum(axis=1),1,atol=1e-6)
results['cnn'] = {'parameters': cnn.count_params(), 'synthetic_batch_loss_accuracy': [float(v) for v in metrics],
                  'scope': 'One synthetic RGB minibatch update; no historical accuracy reproduced.'}

# Execute complete setup and training of all three displayed recurrent models.
for title in ['Build a Reproducible Sequence Before Fitting', 'Split Windows by the Time of Their Target',
              'Original Python Example: SimpleRNN for a Time Series',
              'Original LSTM Example: Replace the Recurrent Layer',
              'Original GRU Example: Another Memory Mechanism']:
    run(title)
assert [len(env[k]) for k in ['X_train','X_val','X_test']] == [340,120,120]
assert np.isclose(env['mean'],env['series'][:360].mean())
for idx, target in enumerate(env['target_time']):
    assert np.allclose(env['X'][idx,:,0],env['values'][target-20:target])
    assert env['y'][idx] == env['values'][target]
run('Train Equally; Choose on Validation; Test Once')
for model in env['models'].values():
    assert len(model.history.history['loss']) == 10
results['sequence'] = {'validation_mse': {k:float(v) for k,v in env['val_mse'].items()},
                       'selected':min(env['val_mse'],key=env['val_mse'].get),
                       'test_rmse_sensor_units':float(env['rmse']), 'last_value_rmse':float(env['naive']),
                       'parameters':{k:v.count_params() for k,v in env['models'].items()}}

for title in ['Original GAN Generator: Turn Noise into Images',
              'Original GAN Discriminator: Evaluate Real or Fake']:
    run(title)
G, D = env['build_generator'](), env['build_discriminator']()
(images, _), _ = tf.keras.datasets.mnist.load_data()
real = (images[:8,...,None].astype('float32') - 127.5) / 127.5
noise = tf.random.normal((8,100))
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_before = [v.numpy().copy() for v in G.trainable_variables]
d_before = [v.numpy().copy() for v in D.trainable_variables]
with tf.GradientTape() as gt, tf.GradientTape() as dt:
    fake = G(noise, training=True)
    real_logits, fake_logits = D(real, training=True), D(fake, training=True)
    g_loss = bce(tf.ones_like(fake_logits), fake_logits)
    d_loss = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)
g_grads, d_grads = gt.gradient(g_loss,G.trainable_variables), dt.gradient(d_loss,D.trainable_variables)
assert fake.shape == (8,28,28,1) and real_logits.shape == (8,1)
assert np.max(np.abs(fake.numpy())) <= 1 and np.max(np.abs(real)) <= 1
assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in g_grads+d_grads)
tf.keras.optimizers.Adam(1e-4).apply_gradients(zip(g_grads,G.trainable_variables))
tf.keras.optimizers.Adam(1e-4).apply_gradients(zip(d_grads,D.trainable_variables))
assert any(not np.array_equal(a,b.numpy()) for a,b in zip(g_before,G.trainable_variables))
assert any(not np.array_equal(a,b.numpy()) for a,b in zip(d_before,D.trainable_variables))
results['gan'] = {'generator_loss':float(g_loss),'discriminator_loss':float(d_loss),
                  'generator_parameters':G.count_params(), 'discriminator_parameters':D.count_params(),
                  'scope':'One update of each network on eight real MNIST examples; no epoch snapshots reproduced.'}
(HERE/'example-checks.json').write_text(json.dumps(results,indent=2)+'\n')
print('All math, model-interface, gradient, and sensor-training checks passed.',flush=True)
