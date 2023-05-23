
import tensorflow as tf
# Import common tensorflow layers and activations
from tensorflow.keras.layers import Dense, BatchNormalization, Layer
from tensorflow.keras.layers import Lambda, Rescaling
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras import Model


# Sparsemax activation function implemented with tensorflow ops
# https://arxiv.org/pdf/1602.02068.pdf
# Copied from https://github.com/tensorflow/addons/blob/b2dafcfa74c5de268b8a5c53813bc0b89cadf386/tensorflow_addons/activations/sparsemax.py#L96

def _compute_2d_sparsemax(logits, axis=-1):
    """Performs the sparsemax operation when axis=-1."""
    if axis != -1:
        raise ValueError("Only axis=-1 is supported.")

    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe

def sparsemax(logits, axis=-1):
    return Lambda(lambda x: _compute_2d_sparsemax(x))(logits)

class GLULayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GLULayer, self).__init__()
        self.forward = None
        self.units = units*2
        self.dense = Dense(self.units)

    def call(self, inputs):
        # Also difference here where we have some more parameters relative to the original paper
        x = self.dense(inputs)
        x1, x2 = tf.split(x, 2, axis=-1)
        return x1 * sigmoid(x2)

def glu(act):
    gate_signal, activation_signal = tf.split(act, 2, axis=-1)
    return gate_signal * sigmoid(activation_signal)

class SharedFeatureLayer(Layer):
    def __init__(self,
                units,
                depth=2,
                dense_activation="relu",
                virtual_batch_size=None,
                batch_momentum=0.95,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.dense_layers = [Dense(
            2*units, 
            activation=dense_activation,
            use_bias=True
            ) for _ in range(depth)]
        self.bn_layers = [
            BatchNormalization(
                virtual_batch_size=virtual_batch_size,
                momentum = batch_momentum
            ) for _ in range(depth)
        ]
        self.scaling = Rescaling(0.5**0.5)
    
    def call(self, inputs, training=False, *args, **kwargs):
        x = inputs
        for i in range(self.depth):
            y = self.dense_layers[i](x) # (B, units*2)
            y = self.bn_layers[i](y, training=training) # (B, units*2)
            y = glu(y) # (B, units)
            # Skip first residual connection as input is not guaranteed to be same shape as num units
            if i > 0:
                x = self.scaling(x + y) # (B, units)
            else:
                x = y # (B, units)
        return x

class FeatureTransformer(Layer):
    def __init__(
            self,
            units, 
            shared_layer, 
            dense_activation="relu", 
            depth=2, 
            virtual_batch_size=None,
            batch_momentum=0.95,
            *args, **kwargs):
        super().__init__( *args, **kwargs )
        self.shared_layer = shared_layer
        self.depth = depth
        self.dense_layers = [
            Dense(
            2*units, 
            activation=dense_activation,
            use_bias=True,
            ) for _ in range(depth)]
        self.bn_layers = [
            BatchNormalization(
                virtual_batch_size=virtual_batch_size,
                momentum = batch_momentum
            ) for _ in range(depth)
        ]
        # self.glu_layers = [GLULayer(units) for _ in range(depth)]
        self.scaling = Rescaling(0.5**0.5)

    def call(self, data, training=False, *args, **kwargs):
        x = self.shared_layer(data)
        for i in range(self.depth):
            y = self.dense_layers[i](x)
            y = self.bn_layers[i](y, training=training)
            y = glu(y)
            x = self.scaling(x + y)

        return x # Feature Transformer

class AttentiveTransformer(Layer):
    def __init__(
            self, 
            units, 
            dense_activation="relu", 
            virtual_batch_size=None,
            batch_momentum=0.95,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = Dense(units, activation=dense_activation)
        self.bn = BatchNormalization(virtual_batch_size=virtual_batch_size, momentum=batch_momentum)
    
    def call(self, data, training=False, *args, **kwargs):
        x = self.dense(data)
        x = self.bn(x, training=training)
        return x # Attentive
        

class TabNet(Model):
    def __init__(
            self, 
            dim_features,
            dim_attention, 
            dim_output, 
            output_activation, 
            sparsity=0.0001, 
            num_steps=5, 
            gamma=1.5, 
            feature_shared_layers=2,
            feature_transformer_layers=2,
            batch_momentum = 0.95,
            virtual_batch_size = None,
            preprocess_layers=None,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_step = num_steps
        self.dim_features = dim_features
        self.dim_attention = dim_attention # Currently keep it simple and just have equal attention and pre_output dimensions
        self.dim_pre_output = dim_attention
        self.dim_output = dim_output
        self.gamma = tf.constant(gamma)
        self.eps = tf.constant(1e-5)
        self.sparsity_coef = sparsity
        self.preprocess_layers = preprocess_layers
        self.shared_layer = SharedFeatureLayer(
            units=self.dim_attention+self.dim_pre_output, 
            depth=feature_shared_layers,
            virtual_batch_size=virtual_batch_size,
            batch_momentum=batch_momentum, 
            name="shared_feature_layer")
        self.feature_transformers = [
            FeatureTransformer(
                units=self.dim_attention+self.dim_pre_output,
                shared_layer=self.shared_layer,
                depth=feature_transformer_layers,
                virtual_batch_size=virtual_batch_size,
                batch_momentum=batch_momentum, 
                name=f"feat_{i}"
            ) 
            for i in range(num_steps + 1)]
        self.attention_layers = [
            AttentiveTransformer(
                units=self.dim_features, 
                virtual_batch_size=virtual_batch_size,
                batch_momentum=batch_momentum, 
                name=f"attn_{i+1}") 
            for i in range(num_steps)
        ]
        self.norm_in = BatchNormalization(
            name="norm_in",
            momentum=batch_momentum,
            )
        self.output_dense = Dense(dim_output, name="output", activation=output_activation)
        self.attn_activation = _compute_2d_sparsemax
    
    def call(self, data, training=False):
        if self.preprocess_layers is not None:
            data = self.preprocess_layers(data)
        normed_data = self.norm_in(data)

        d0, a_i = tf.split(
            self.feature_transformers[0](normed_data, training=training), 
            2, 
            axis=-1)
        decision = tf.zeros_like(d0)
        priors = []
        entropy = 0.
        for i in range(self.num_step):

            # Main difference to google implementation is usage of prior list?
            candidate_mask = self.attention_layers[i](a_i, training=training)
            for prior in priors:
                candidate_mask = candidate_mask*(self.gamma - prior)
            candidate_mask = self.attn_activation(candidate_mask)
            
            priors.append(candidate_mask)
            decision_entropy = tf.reduce_mean(
                tf.reduce_sum(
                    -candidate_mask * tf.math.log(candidate_mask + self.eps)/self.num_step,
                    axis=-1)
            )
            entropy += decision_entropy         
            
            normed_features = normed_data * candidate_mask
            x = self.feature_transformers[i+1](normed_features, training=training)
            d_i, a_i = tf.split(x, 2, axis=-1)
            decision += relu(d_i)
        
        return self.output_dense(decision), entropy
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, entropy = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if self.sparsity_coef > 0:
                loss += self.sparsity_coef * entropy
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        x, y = data
        y_pred, _ = self(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred, _ = self(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}