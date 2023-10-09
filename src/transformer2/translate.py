import torch
from transformer2.model import Transformer
from utils.data_loader import Tokens


def translate_sentence(
    model: Transformer,
    source_sentence: str,
    tokens: Tokens,
    device: torch.device,
    max_length: int = 50,
) -> (list[str], list[int]):
    source_tokens = tokens.source.encode_as_ids(source_sentence)
    source_tokens.insert(0, tokens.bos)
    source_tokens.append(tokens.eos)

    sentence_tensor = torch.LongTensor(source_tokens).unsqueeze(1).to(device)

    output_tokens = [tokens.bos]
    for _ in range(max_length):
        trg_tensor = torch.LongTensor(output_tokens).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        output_tokens.append(best_guess)

        if best_guess == tokens.eos:
            break

    output_sentence = tokens.target.decode(output_tokens)
    return (output_sentence, output_tokens)
