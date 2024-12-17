import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

def extract_colors(image, num_colors):
    """
    Extract dominant colors from an image using KMeans clustering.
    """
    
    image = image.resize((150, 150))
    image_data = np.array(image)

    
    pixels = image_data.reshape(-1, 3)

    
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)


    colors = kmeans.cluster_centers_.astype(int)
    return colors

def rgb_to_hex(rgb):
    """
    Convert RGB values to Hexadecimal color codes.
    """
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def plot_colors(colors, save_as_file=False):
    """
    Plot a palette of colors and optionally save it as an image.
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, color in enumerate(colors):
        patch = patches.Rectangle((i, 0), 1, 1, facecolor=np.array(color) / 255)
        ax.add_patch(patch)

    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()

    if save_as_file:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        return buf
    else:
        return fig

def adjust_image(image, brightness, contrast):
    """
    Adjust the brightness and contrast of the image.
    """
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    return image

def main():
    st.title("🎨 Advanced Color Palette Generator")
    st.write("Upload an image, generate dominant colors, and interact with the palette.")

    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        
        st.subheader("🛠 Adjust Image")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
        adjusted_image = adjust_image(image, brightness, contrast)
        st.image(adjusted_image, caption="Adjusted Image", use_column_width=True)

        
        st.subheader("Extract Colors")
        num_colors = st.slider("Number of Colors", min_value=2, max_value=10, value=5)

        
        colors = extract_colors(adjusted_image, num_colors)
        hex_colors = [rgb_to_hex(color) for color in colors]
        
        
        st.write("Dominant Colors:")
        fig = plot_colors(colors)
        st.pyplot(fig)

        
        st.subheader("Color Details")
        for i, color in enumerate(colors):
            st.write(f"Color {i+1}: RGB {tuple(color)} | Hex {hex_colors[i]}")

        
        st.subheader("Save Palette")
        if st.button("Download Palette Image"):
            buf = plot_colors(colors, save_as_file=True)
            st.download_button(label="Download", data=buf, file_name="color_palette.png", mime="image/png")

        
        st.subheader("Copy Hex Codes")
        hex_text = "\n".join(hex_colors)
        st.code(hex_text, language="text")

        
        st.download_button("Download Hex Codes", data=hex_text, file_name="hex_codes.txt", mime="text/plain")

if __name__ == "__main__":
    main()
