from PIL import Image, ImageDraw

# Create a sequence of images
images = []
for i in range(10):
    img = Image.new('RGBA', (100, 100), (255, 0, 0, 0)) # Transparent background
    d = ImageDraw.Draw(img)
    # Draw a moving blue circle
    d.ellipse([i*5, 20, i*5+60, 80], fill=(0, 0, 255, 255))
    images.append(img)

# Save as GIF
images[0].save('assets/memes/67meme.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
print("Created assets/memes/67meme.gif")
