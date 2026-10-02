# Recipe Generator: Llama 3.2 1B Fine-Tuned with QLoRA

Give the model a list of ingredients and it writes a structured recipe (title, ingredients, directions). It is Llama 3.2 1B Instruct fine-tuned with QLoRA, so the whole run fits on a free Google Colab T4 GPU.

## How it works
- **Base model:** `unsloth/Llama-3.2-1B-Instruct`
- **Dataset:** `tengomucho/all-recipes-split` (800 shuffled examples, cleaned and filtered for missing fields)
- **Split:** 80% train / 10% validation / 10% test
- **Quantization:** 4-bit NF4 with double quantization (bitsandbytes)
- **LoRA:** rank 16, alpha 32, dropout 0.05, applied to the q/k/v/o attention projections
- **Training:** 1 epoch, learning rate 2e-4, batch size 8 with gradient accumulation 2, fp16, max length 256
- **Evaluation:** test loss and perplexity on the held-out split

## Run it
1. Open `recipe_llama_finetune_fast.ipynb` in Google Colab.
2. Set the runtime to **T4 GPU** (`Runtime > Change runtime type`).
3. Run all cells. The notebook trains, evaluates, saves the LoRA adapter, and lets you try your own ingredients.

## Example
```python
generate_recipe(["rice", "chicken"])
```


## Results
Test perplexity: Recipe: Title: Garlic Rice
Ingredients:
rice
garlic
toasted almonds
salt
pepper
cumin
fresh parsley
toasted almonds
salt
pepper
cumin
fresh parsley
canned tomato paste
olive oil
garlic


Python, PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, Datasets, Google Colab
