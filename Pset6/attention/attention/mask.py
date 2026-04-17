import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")

    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()

    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    visualize_attentions(tokens, result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    input_ids = inputs["input_ids"][0]  # shape: (seq_len,)

    positions = tf.where(tf.equal(input_ids, mask_token_id))

    if tf.shape(positions)[0] == 0:
        return None

    return int(positions[0][0])


def get_color_for_attention_score(attention_score):
    """
    Return a grayscale RGB tuple (0–255) for attention score.
    Higher score = lighter color.
    """
    score = float(attention_score.numpy())
    intensity = int(score * 255)

    return (intensity, intensity, intensity)


def visualize_attentions(tokens, attentions):
    """
    Produce diagrams for ALL layers and ALL heads.
    """
    for layer_index, layer in enumerate(attentions):
        for head_index, head in enumerate(layer[0]):
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                head
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing self-attention scores.
    """
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw tokens (columns + rows)
    for i, token in enumerate(tokens):

        # Column text (rotated)
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Row text
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw attention heatmap
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            y = PIXELS_PER_WORD + i * GRID_SIZE

            color = get_color_for_attention_score(attention_weights[i][j])

            draw.rectangle(
                (x, y, x + GRID_SIZE, y + GRID_SIZE),
                fill=color
            )

    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
