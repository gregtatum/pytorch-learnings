#!/usr/bin/env python3
from os import path
from transformer2.model import Transformer
from transformer2.parameters import HyperParameters
from transformer.utils import SentenceProcessor
from transformer2.translate import translate_sentence
from utils import get_device
from transformer.trainer_manager import ArtifactPathFn, TrainerManager
import torch
from torch import nn, optim
from utils.data_loader import load_data, load_tokenizers
from tap import Tap

"""
Train a transformer model.
"""


class Arguments(Tap):
    """
    Train a transformer model for translations.

    - Train a language on the full corpus:
      poetry run src/transformer_trainer2.py -- --source "en" --target "es"

    - Test the system on a small subset of the corpus:
      poetry run src/transformer_trainer2.py -- --source "en" --target "es" --small
    """

    source: str  # The source language, e.g. "en"
    target: str  # The target language, e.g. "es"
    small: bool = False  # Use a small test data set which is faster to load


args = Arguments().parse_args()

device = get_device()
p = HyperParameters(args.source, args.target)
tokens = load_tokenizers(p.source_language, p.target_language)

model = Transformer(
    p.d_embedding,
    p.source_vocab_size,
    p.source_vocab_size,
    tokens.pad,
    p.num_heads,
    p.num_layers,  # num_encoder_layers: int,
    p.num_layers,  # num_decoder_layers: int,
    p.feed_forward_size,  # feed_forward_size: int,
    p.dropout,
    p.max_seq_length,
    device,
).to(device)

# Set the model to training mode.
model.train()

# Generate random sample data
torch.manual_seed(1234)


criterion = nn.CrossEntropyLoss(ignore_index=tokens.pad)
optimizer = optim.Adam(
    model.parameters(),
    lr=p.learning_rate,
)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=0.1, patience=10, verbose=True
# )

data = load_data(p.source_language, p.target_language, small=args.small)

# Get the tokens for the "begin of a sentence" (bos) and "end of sentence" eos


# TODO - This could be made faster if this processing wasn't done on the fly.

sentence_processor = SentenceProcessor(tokens, p.max_seq_length, device)


def save_model(artifact_path: ArtifactPathFn) -> None:
    torch.save(
        model.state_dict(),
        artifact_path("model.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        artifact_path("optimizer.pt"),
    )


def load_model(artifact_path: ArtifactPathFn) -> None:
    if path.exists(artifact_path("model.pt")):
        print("Loading a saved model")
        model.load_state_dict(torch.load(artifact_path("model.pt")))

    if path.exists(artifact_path("optimizer.pt")):
        print("Loading a saved optimizer")
        optimizer.load_state_dict(torch.load(artifact_path("optimizer.pt")))


def train_step(data_slice: slice) -> float:
    source_data, target_data = sentence_processor.get_batch(data, data_slice)

    # The model expects the input transposed compared to how the sentence processor
    # builds it.
    source_data = source_data.t().contiguous()
    target_data = target_data.t().contiguous()

    torch.set_printoptions(threshold=10_000)

    # The model predicts the next token, so the last token will never be predicted.
    # The data is arranged like: [sentence_tokens, batch_size]
    target_data_for_model = target_data[:-1, :]

    output = model(source_data, target_data_for_model)

    # Perform backward propagation.
    optimizer.zero_grad()

    loss = criterion(
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        output.reshape(-1, output.shape[2]),
        # Also remove the start token.
        target_data[1:].reshape(-1),
    )

    loss.backward()

    # TODO - Is this needed?
    # Clip to avoid exploding gradient issues, makes sure grads are
    # within a healthy range
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    optimizer.step()

    return loss.item()


def epoch_done() -> None:
    sentence = "The very existence of space is a real puzzle."
    output_sentence, output_tokens = translate_sentence(
        model,
        sentence,
        tokens,
        device,
    )
    print("Source sentence", sentence)
    print("Target sentence", output_sentence)
    print("Target tokens", output_tokens)


# mean_loss = sum(losses) / len(losses)
# scheduler.step(mean_loss)

manager = TrainerManager("transformer", model, p)
manager.train(
    train_step,
    name=f"Translations Model: {args.source}-{args.target}",
    batch_size=p.batch_size,
    num_epochs=100,
    epoch_done=epoch_done,
    data_size=len(data),
    save_model=save_model,
    load_model=load_model,
)
