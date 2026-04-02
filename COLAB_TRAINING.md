# Google Colab Training Steps

## 1. New notebook banao

- Colab me `+ कोड` par click karo

## 2. GPU on karo

- `रनटाइम` -> `रनटाइम प्रकार बदलें`
- `Hardware accelerator` me `GPU` select karo
- `Save` karo

Check:

```python
!nvidia-smi
```

## 3. GitHub repo clone karo

```python
!git clone https://github.com/pagal4206/chatbaka.git
%cd chatbaka
```

## 4. Google Drive mount karo

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 5. Requirements install karo

```python
!pip install -q -r requirements.txt
```

## 6. Apni CSV upload karo agar repo me nahi hai

Ye step tabhi chahiye jab `data/custom/hinglish_conversations.csv` GitHub repo me already na ho.

```python
from google.colab import files
uploaded = files.upload()
```

Upload ke baad:

```python
import os, shutil
os.makedirs("data/custom", exist_ok=True)
shutil.move("hinglish_conversations.csv", "data/custom/hinglish_conversations.csv")
```

## 7. Dataset prepare karo

Sirf apni local CSV se dataset banana ho to:

```python
!python scripts/prepare_dataset.py --synthetic-max 0 --turn-pairs-max 0 --chat-format-max 0 --romanized-max 0
```

## 8. Training start karo

```python
!python scripts/train_lora.py --load-in-4bit --packing --output-dir /content/drive/MyDrive/chatbaka-model/hinglish-qwen25-3b-lora
```

Output yahan aayega:

- `/content/drive/MyDrive/chatbaka-model/hinglish-qwen25-3b-lora/final_adapter`

## 9. Optional merged model

```python
!python scripts/merge_adapter.py --output-dir /content/drive/MyDrive/chatbaka-model/hinglish-qwen25-3b-merged
```

Standalone model yahan save hoga:

- `/content/drive/MyDrive/chatbaka-model/hinglish-qwen25-3b-merged`

## 10. Baad me use kaise karna hai

- Agar adapter save kiya hai to base model + adapter dono chahiye
- Agar merged model save kiya hai to usse direct hosting par use kar sakte ho
- Telegram bot ke liye API server me model path set karke run karo
