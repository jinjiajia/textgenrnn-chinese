from keras.engine import InputSpec, Layer
from keras import backend as K
from keras import initializers


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for
    a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        # print("self.input_spec",self.input_spec)
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        # print("self.W",self.W)
        self.trainable_weights = [self.W]
        # print("self.trainable_weights",self.trainable_weights)
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        # print("logits",logits)
        x_shape = K.shape(x)
        # print("x_shape",x_shape)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        # print("logits",logits)
        # print("logits - K.max(logits, axis=-1, keepdims=True)",logits - K.max(logits, axis=-1, keepdims=True))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
        # print("ai",ai)

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        # print("K.expand_dims(att_weights)",K.expand_dims(att_weights))
        weighted_input = x * K.expand_dims(att_weights)
        # print("weighted_input",weighted_input)
        result = K.sum(weighted_input, axis=1)
        # print("result",result)
        # print("self.return_attention",self.return_attention)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0],
                                                   input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
