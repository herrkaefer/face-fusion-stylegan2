import gradio as gr
from PIL import Image
from inference import fuse_faces
import numpy as np
import traceback
import sys
import tempfile
import os

def preprocess_image(img):
    """Preprocess image to a reasonable size while maintaining aspect ratio"""
    if img is None:
        return None

    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img

    # Set maximum dimension while maintaining aspect ratio
    max_size = 800
    ratio = max_size / max(img_pil.size)
    if ratio < 1:  # Only resize if image is larger than max_size
        new_size = tuple([int(dim * ratio) for dim in img_pil.size])
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

    # Convert back to numpy array
    return np.array(img_pil)

def process_images(img1, img2, age_factor=0, smile_factor=0, variation_strength=0, random_seed=None):
    try:
        if img1 is None or img2 is None:
            raise ValueError("Please upload both parent images")

        # Preprocess input images
        img1 = preprocess_image(img1)
        img2 = preprocess_image(img2)

        print(f"Processing images")

        # Create temporary directory for input images if needed
        temp_dir = tempfile.mkdtemp()

        # Handle both file paths and image data
        if isinstance(img1, str):
            img1_path = img1
        else:
            img1_path = os.path.join(temp_dir, "parent1.jpg")
            Image.fromarray(img1).save(img1_path, quality=95)

        if isinstance(img2, str):
            img2_path = img2
        else:
            img2_path = os.path.join(temp_dir, "parent2.jpg")
            Image.fromarray(img2).save(img2_path, quality=95)

        print(f"Image paths: {img1_path}, {img2_path}")

        # Call the face fusion function with image paths and get the resulting PIL image
        child_pil = fuse_faces(
            img1_path,
            img2_path,
            random_seed=random_seed,
            age_factor=age_factor,
            smile_factor=smile_factor,
            variation_strength=variation_strength
        )

        print("Face fusion completed successfully")

        # Clean up temporary files if they were created
        if not isinstance(img1, str) or not isinstance(img2, str):
            import shutil
            shutil.rmtree(temp_dir)

        # Convert PIL image to numpy array for Gradio
        child_array = np.array(child_pil)

        return child_array, "Generation successful!"
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return None, error_msg

# Create Gradio interface
with gr.Blocks(title="Face Fusion App") as demo:
    gr.Markdown("# Face Fusion")
    gr.Markdown("Upload two parent face images to generate a child face.")

    # First row: Parent images side by side
    with gr.Row():
        input_image1 = gr.Image(
            label="Parent 1",
            type="numpy",
            height=300,
            show_download_button=False
        )
        input_image2 = gr.Image(
            label="Parent 2",
            type="numpy",
            height=300,
            show_download_button=False
        )

    # Second row: Child image
    with gr.Row():
        output_image = gr.Image(
            label="Generated Child",
            height=400,
            show_download_button=True
        )

    # Third row: Controls
    with gr.Row():
        with gr.Column(scale=1):
            # Empty column for spacing
            gr.Markdown("")
        with gr.Column(scale=2):
            age_slider = gr.Slider(-6, 6, value=-4, step=0.1, label="Age Factor")
            smile_slider = gr.Slider(-5, 5, value=0.8, step=0.1, label="Smile Factor")
            variation_slider = gr.Slider(0, 0.5, value=0.01, step=0.01, label="Variation Strength")
            seed_number = gr.Number(label="Random Seed (optional)", precision=0)
            generate_btn = gr.Button("Generate Child Face", size="lg")
            status_text = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=1):
            # Empty column for spacing
            gr.Markdown("")

    generate_btn.click(
        fn=process_images,
        inputs=[
            input_image1,
            input_image2,
            age_slider,
            smile_slider,
            variation_slider,
            seed_number
        ],
        outputs=[output_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
