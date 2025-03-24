import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


if __name__ == "__main__":
    img_01_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/01/cc0f8130.png")
    img_01_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/01/fcff_post_seg_cc0f8130_cc0f8130.png")
    img_02_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/02/66301bd1.png")
    img_02_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/02/434c_post_seg_66301bd1_66301bd1.png")
    img_03_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/03/31a93a94.png")
    img_03_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/03/a298_post_seg_31a93a94_31a93a94.png")
    img_04_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/04/40a2e6a8.png")
    img_04_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/04/9660_post_seg_6409a46a_6409a46a.png")
    img_05_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/05/a8b3b5d4.png")
    img_05_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/05/52cc_post_seg_a8b3b5d4_a8b3b5d4.png")
    img_06_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/06/dd6422f8.png")
    img_06_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/06/5a0a_post_seg_dd6422f8_dd6422f8.png")

    plt.figure(figsize=(10, 10), dpi=400)
    
    plt.subplot(1,6,1)
    plt.imshow(img_01_1)
    plt.axis("off")
    plt.subplot(1,6,2)
    plt.imshow(img_02_1)
    plt.axis("off")
    plt.subplot(1,6,3)
    plt.imshow(img_03_1)
    plt.axis("off")
    plt.subplot(1,6,4)
    plt.imshow(img_04_1)
    plt.axis("off")
    plt.subplot(1,6,5)
    plt.imshow(img_05_1)
    plt.axis("off")
    plt.subplot(1,6,6)
    plt.imshow(img_06_1)
    plt.axis("off")
    """
    plt.subplot(1,6,1)
    plt.imshow(img_01_2)
    plt.axis("off")
    plt.subplot(1,6,2)
    plt.imshow(img_02_2)
    plt.axis("off")
    plt.subplot(1,6,3)
    plt.imshow(img_03_2)
    plt.axis("off")
    plt.subplot(1,6,4)
    plt.imshow(img_04_2)
    plt.axis("off")
    plt.subplot(1,6,5)
    plt.imshow(img_05_2)
    plt.axis("off")
    plt.subplot(1,6,6)
    plt.imshow(img_06_2)
    plt.axis("off")
    """
    plt.tight_layout()
    plt.savefig("plot_img_1.png", bbox_inches = "tight", pad_inches = 0)
    plt.show()
    
