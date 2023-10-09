class HyperParameters:
    def __init__(self, source_language: str, target_language: str) -> None:
        self.source_language = source_language
        self.target_language = target_language
        self.source_vocab_size = 5000
        self.target_vocab_size = 5000
        self.batch_size = 256
        self.max_seq_length = 100
        self.learning_rate = 3e-4

        self.dropout = 0.2

        # Dimensions for the embedding vector
        self.d_embedding = 200

        # Dimension of the feedforward network model in ``nn.TransformerEncoder``
        self.feed_forward_size = 2048

        # The number of heads in nn.MultiheadAttention
        self.num_heads = 8

        # number of `nn.TransformerEncoderLayer` in `nn.TransformerEncoder`
        self.num_layers = 6
