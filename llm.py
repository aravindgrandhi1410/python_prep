!pip install tensorflow
import tensorflow as tf

class LLM(tf.keras.Model):

    def __init__(self, hidden_units=128, activation='relu', integration='gru'):
        super(LLM, self).__init__()

        self.hidden_units = hidden_units
        self.activation = activation
        self.integration = integration

        self.self_attention_layers = []
        for _ in range(hidden_units):
            self.self_attention_layers.append(
                tf.keras.layers.Attention(use_bias=False)
            )

    def call(self, inputs):
        outputs = inputs
        for layer in self.self_attention_layers:
            outputs = layer(outputs)

        return outputs

if __name__ == '__main__':
    model = LLM()
    print(model.summary())
