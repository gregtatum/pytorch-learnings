class HyperParameters:
    def __init__(self) -> None:
        self.source_language = "en"
        self.target_language = "es"
        self.source_vocab_size = 5000
        self.target_vocab_size = 5000
        self.batch_size = 256
        self.d_model = 512  # d_k := 64
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.max_seq_length = 100
        self.dropout = 0.1
        self.learning_rate = 0.0001
        self.learning_betas = (0.9, 0.98)
        self.learning_epsilon = 1e-9
