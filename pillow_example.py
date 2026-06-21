from PIL import Image, ImageDraw, ImageFont


def main():
    width, height = 320, 220
    image = Image.new("RGB", (width, height), "skyblue")
    draw = ImageDraw.Draw(image)

    # Draw a gradient sky background.
    for y in range(height):
        color = (int(135 + y * 0.4), int(206 + y * 0.1), int(235 - y * 0.2))
        draw.line([(0, y), (width, y)], fill=color)

    draw.rectangle([40, 50, 280, 170], outline="white", width=3)
    draw.ellipse([120, 20, 200, 100], fill="yellow", outline="orange")

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except OSError:
        font = ImageFont.load_default()

    draw.text((90, 180), "Pillow Example", fill="white", font=font)

    image.save("pillow_example_output.png")
    print("Created pillow_example_output.png")


if __name__ == "__main__":
    main()
