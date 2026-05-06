import matplotlib.pyplot as plt

def show_sample(img, text):
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(text)
    plt.axis("off")
    plt.show()