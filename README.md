# DF-GAN

DF-GAN (Deep Fusion GAN) is a GAN model for text-to-image synthesis. It leverages the capabilities of pre-trained language models (such as BERT) to encode text and generate realistic images based on text descriptions. DF-GAN improves upon traditional GAN models by incorporating gradient penalties and mixed precision training, which stabilizes the training process and makes it more efficient.

## Why DF-GAN?

- **Stability**: By using gradient penalties, DF-GAN mitigates mode collapse and other common GAN training issues.
- **Efficiency**: Mixed precision training allows DF-GAN to leverage the power of GPUs more effectively, speeding up training without sacrificing model performance.
- **Text Encoding**: Incorporating pre-trained models like BERT for text encoding improves the quality and relevance of generated images.

## Training DF-GAN

### Prerequisites

1. **Set up GPU in Google Colab**
    - Navigate to the top menu and select `Runtime` -> `Change runtime type`.
    - Set `Hardware accelerator` to `GPU`.

2. **Install Dependencies**

    ```bash
    pip install torch torchvision transformers pycocotools matplotlib tqdm
    ```

3. **Download MS-COCO Dataset**
    - Download the MS-COCO dataset from the [COCO website](https://cocodataset.org/#download). You need the `train2017` images and the `train2017.json` annotations.

4. **Prepare Dataset**
    - Ensure the dataset is organized as follows:

      ```
      path_to/
          ├── train2017/
          │   ├── 000000000001.jpg
          │   ├── 000000000002.jpg
          │   └── ...
          └── annotations/
              └── train2017.json
      ```

### Running the Training Script

1. **Modify the DataLoader Path**
    - Update the dataset paths in the training script (`train_dfgan.py`):

      ```python
      dataset = COCODataset(img_dir='path_to/train2017', ann_file='path_to/annotations/train2017.json', transform=transform)
      ```

2. **Run the Training Script**
    - Save the training script as `train_dfgan.py` and run it:

      ```bash
      python train_dfgan.py
      ```

### Generating Images

1. **Run the Image Generation Script**
    - Save the image generation script as `generate_images.py` and run it:

      ```bash
      python generate_images.py
      ```

## Example Usage

```python
# Example usage in generate_images.py
text_description = "A description of the image you want to generate"
generated_image = generate_image_from_text(text_description)

# Plot the generated image
plt.imshow(make_grid(generated_image, normalize=True).permute(1, 2, 0).detach().cpu().numpy())
plt.axis('off')
plt.show()
```

### Output

Here is the output after training the model on only 10% of the dataset due to limited resources. The training took two nights to complete. The prompt was to generate "A table".

Output:

![table](https://github.com/rakeshutekar/Text-to-Image-using-DF-GAN/assets/48244158/cda8fc84-2db7-420b-84a8-60e291ff59b3)



Futhermore, Below is the plot showing the generator and discriminator loss during training. The plot indicates that the generator loss fluctuated significantly, while the discriminator loss remained relatively stable and low. To address this, we made several changes to the code, including reducing the learning rate, adding gradient penalties, and implementing gradient accumulation.

![Figure_1](https://github.com/rakeshutekar/Text-to-Image-using-DF-GAN/assets/48244158/bde7b6c3-e0d7-49f6-87fa-de7fe11cdf18)


