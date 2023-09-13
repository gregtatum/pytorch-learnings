from typing import Any, Optional

class SentencePieceProcessor:
    def __init__(
        self,
        # The sentencepiece model file path.
        model_file: Optional[str] = None,
        # The sentencepiece model serialized proto.
        model_proto: Optional[Any] = None,
        out_type: Any = int,
        add_bos: bool = False,
        add_eos: bool = False,
        reverse: bool = False,
        emit_unk_piece: bool = False,
        enable_sampling: bool = False,
        nbest_size: int = -1,
        alpha: float = 0.1,
        num_threads: int = -1,
    ) -> None:
        """Initialize sentencepieceProcessor.

        Args:
            model_file: The sentencepiece model file path.
            model_proto: The sentencepiece model serialized proto.
            out_type: output type. int or str.
            add_bos: Add <s> to the result (Default = false)
            add_eos: Add </s> to the result (Default = false) <s>/</s> is added after
            reversing (if enabled).
            reverse: Reverses the tokenized sequence (Default = false)
            emit_unk_piece: Emits the unk literal string (Default = false)
            nbest_size: sampling parameters for unigram. Invalid in BPE-Dropout.
                        nbest_size = {0,1}: No sampling is performed.
                        nbest_size > 1: samples from the nbest_size results.
                        nbest_size < 0: assuming that nbest_size is infinite and samples
                        from the all hypothesis (lattice) using
                        forward-filtering-and-backward-sampling algorithm.
            alpha: Soothing parameter for unigram sampling, and dropout probability of
                merge operations for BPE-dropout.
            num_threads: number of threads in batch processing (Default = -1, auto-detected)
        """
        ...
    def id_to_piece(self, id: int) -> str: ...
    def piece_to_id(self, piece: str) -> int: ...
    def encode_as_ids(self, input: str) -> list[int]: ...
    def encode_as_pieces(self, input: str) -> list[str]: ...
    def decode(self, input: list[int]) -> str: ...
    def load(self, model_path: str) -> None: ...
    def get_score(self, id: int) -> float: ...
    def get_piece_size(self) -> int: ...
    def vocab_size(self) -> int: ...
    def unk_id(self) -> int: ...
    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def pad_id(self) -> int: ...

class SentencePieceTrainer:
    @staticmethod
    # https://github.com/google/sentencepiece/blob/8cbdf13794284c30877936f91c6f31e2c1d5aef7/doc/options.md
    def train(
        # comma separated list of input sentences
        input: str = "",
        # Input format. Supported format is `text` or `tsv`.
        input_format: str = "",
        # output model prefix
        model_prefix: str = "",
        # model algorithm: unigram, bpe, word or char
        model_type: str = "unigram",
        # vocabulary size
        vocab_size: int = 32,
        # comma-separated list of languages this model can accept
        accept_language: str = "",
        # the size of self test samples
        self_test_sample_size: int = 0,
        # character coverage to determine the minimum symbols
        character_coverage: float = 0.9995,
        # maximum size of sentences the trainer loads
        input_sentence_size: int = 0,
        # Randomly sample input sentences in advance. Valid when --input_sentence_size > 0
        shuffle_input_sentence: bool = True,
        # the size of seed sentencepieces
        seed_sentencepiece_size: int = 1000000,
        # Keeps top shrinking_factor pieces with respect to the loss
        shrinking_factor: float = 0.75,
        # number of threads for training
        num_threads: int = 16,
        # number of EM sub-iterations)
        num_sub_iterations: int = 2,
        # maximum length of sentence piece)
        max_sentencepiece_length: int = 16,
        # maximum length of sentence in byte)
        max_sentence_length: int = 4192,
        # use Unicode script to split sentence pieces)
        split_by_unicode_script: bool = True,
        # split tokens by numbers (0-9))
        split_by_number: bool = True,
        # use a white space to split sentence pieces)
        split_by_whitespace: bool = True,
        # split all digits (0-9) into separate pieces
        split_digits: bool = False,
        # treat whitespace marker as suffix instead of prefix.
        treat_whitespace_as_suffix: bool = False,
        # allow pieces that only contain (consecutive) whitespace tokens
        allow_whitespace_only_pieces: bool = False,
        # comma separated list of control symbols
        control_symbols: str = "",
        # load control_symbols from file.
        control_symbols_file: str = "",
        # comma separated list of user defined symbols
        user_defined_symbols: str = "",
        # load user_defined_symbols from file.
        user_defined_symbols_file: str = "",
        # UTF8 characters in this flag are always used in the character set regardless of --character_coverage
        required_chars: str = "",
        # load required_chars from file.
        required_chars_file: str = "",
        # decompose unknown pieces into UTF-8 byte pieces
        byte_fallback: bool = False,
        # Define score in vocab file)
        vocabulary_output_piece_score: bool = True,
        # Normalization rule name. Choose from nfkc or identity
        normalization_rule_name: str = "nmt_nfkc",
        # Normalization rule TSV file.
        normalization_rule_tsv: str = "",
        # Denormalization rule TSV file.
        denormalization_rule_tsv: str = "",
        # Add dummy whitespace at the beginning of text
        add_dummy_prefix: bool = True,
        # Removes leading, trailing, and duplicate internal whitespace)
        remove_extra_whitespaces: bool = True,
        # If set to false, --vocab_size is considered as a soft limit.)
        hard_vocab_limit: bool = True,
        # If set to true, use all tokens as vocab. Valid for word/char models.
        use_all_vocab: bool = False,
        # Override UNK (<unk>) id.)
        unk_id: int = 0,
        # Override BOS (<s>) id. Set -1 to disable BOS.)
        bos_id: int = 1,
        # Override EOS (</s>) id. Set -1 to disable EOS.)
        eos_id: int = 2,
        # Override PAD (<pad>) id. Set -1 to disable PAD.)
        pad_id: int = -1,
        # Override UNK (<unk>) piece.
        unk_piece: str = "<unk>",
        # Override BOS (<s>) piece.
        bos_piece: str = "<s>",
        # Override EOS (</s>) piece.
        eos_piece: str = "</s>",
        # Override PAD (<pad>) piece.
        pad_piece: str = "<pad>",
        # Dummy surface string for <unk>. In decoding <unk> is decoded to `unk_surface`.
        unk_surface: str = " ⁇ ",
        # Increase bit depth for unigram tokenization.
        train_extremely_large_corpus: bool = False,
        # Seed value for random generator.
        random_seed: int = 4294967295,
        # Whether to add DP while training. Currently supported only by UNIGRAM model.
        enable_differential_privacy: bool = False,
        # Amount of noise to add for DP
        differential_privacy_noise_level: float = 0,
        # Threshold for clipping the counts for DP
        differential_privacy_clipping_threshold: int = 0,
        # show help
        help: bool = False,
        # show version
        version: bool = False,
        # Messages logged at a lower level than this don't actually get logged anywhere
        minloglevel: int = 0,
    ) -> None: ...
