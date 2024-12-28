import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    # audio_list = [item['audio'].squeeze(0) for item in dataset_items]
    spectrogram_list = [item["spectrogram"].squeeze(0) for item in dataset_items]
    text_list = [item["text"] for item in dataset_items]
    text_encoded_list = [item["text_encoded"].squeeze(0) for item in dataset_items]
    audio_path_list = [item["audio_path"] for item in dataset_items]

    # # Pad audio tensors to the same length
    # audio_lengths = torch.tensor([audio.size(0) for audio in audio_list])
    # padded_audio = pad_sequence(audio_list, batch_first=True)

    # Pad spectrogram tensors to the same length along time dimension (last dimension)
    spectrogram_lengths = torch.tensor(
        [spectrogram.size(1) for spectrogram in spectrogram_list]
    )
    padded_spectrograms = torch.stack(
        [
            torch.nn.functional.pad(
                spectrogram,
                (0, max(spectrogram_lengths) - spectrogram.size(1)),
                mode="constant",
                value=0,
            )
            for spectrogram in spectrogram_list
        ]
    )

    # Pad text_encoded tensors to the same length
    text_encoded_lengths = torch.tensor(
        [text_encoded.size(0) for text_encoded in text_encoded_list]
    )
    padded_text_encoded = pad_sequence(text_encoded_list, batch_first=True)

    result_batch = {
        # 'audio': padded_audio,
        # 'audio_lengths': audio_lengths,
        "spectrogram": padded_spectrograms,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": padded_text_encoded,
        "text_encoded_length": text_encoded_lengths,
        "text": text_list,
        "audio_path": audio_path_list,
    }

    return result_batch
