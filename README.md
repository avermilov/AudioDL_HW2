# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

See the task assignment [here](https://github.com/NickKar30/SpeechAI/tree/main/hw2).

## Installation

Follow these steps to install the project:

0. Create and activate new environment using conda.

   ```bash
   conda create -n avermilov_audio_hw2 python=3.11 -y
   conda activate avermilov_audio_hw2
   ```

1. Install all required packages

   ```bash
   pip install https://github.com/kpu/kenlm/archive/master.zip # for KenLM language model support
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Reproduce

To reproduce the best model:
1. Download all the necessary weights (tokenizer, LM weights and vocab, best exp model for inference).
```bash
python3 download_weights.py
```
2. To reproduce the best trained model:
```bash
python3 train.py
```

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py
```

## Описание экспериментов

Пойдем по простому пути - подготовим эксперимент сразу со всеми возможными улучшениями. Список улучшений:
1. DeepSpeech2
2. Обучение на всех данных (train_all)
2. BPE токенизатор (обученный на наших данных, 1000 токенов)
3. Аугментации - Gain, ColoredNoise, TimeMasking, FrequencyMasking
4. Большой батч сайз - 64
5. Beam Search
6. Language Model (4-gram)

Неудивительным образом, этот эксперимент после обучения на 20 000 итераций выбил 24.2% WER на test-other. Также был проведен аналогичный эксперимент
без BPE токенизатора, который на удивление показал такой же результат (~24% WER на test-other). По итогам экспериментов также видим, что огромный буст к
качеству дает использование Beam Search и Language Model. Это в целом ожидаемо, так как сырые предсказания ctc декодирования дают множество опечаток
или ошибок, которые BS+LM очень эффективно правит.

## Метрики

Логи экспериментов (с/без bpe) доступны в [W&B](https://wandb.ai/artermiloff/asr_project/)

Ниже представлены итоговые результаты для двух способов инференса - argmax и beam search + language model:

| Dataset    | Method    | CER (%)  | WER (%)  |
|------------|-----------|----------|----------|
| test-clean | Argmax    | 3.9      | 13.0     |
|            | BS + LM   | **3.2**  | **8.5**  |
| test-other | Argmax    | 11.9     | 30.6     |
|            | BS + LM   | **11.4** | **24.2** |

Итого, выбит 24.2% WER на test-other (85 баллов), плюс сделаны бонусы LM (+10 баллов) и BPE (+10 баллов) = 105 баллов.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
