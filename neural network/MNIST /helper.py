import matplotlib.pyplot as plt
import numpy as np

def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(8,4), ncols=2)

    ax1.imshow(img.squeeze(), cmap='gray')
    ax1.axis('off')

    ax2.barh(np.arange(10), ps)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels([
        'T-shirt',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle Boot'
    ])
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.show()