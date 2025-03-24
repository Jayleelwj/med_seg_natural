import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


if __name__ == "__main__":
    img_01_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/01/4231c706.png")
    img_01_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/01/d168_post_seg_4231c706_4231c706.png")
    img_02_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/02/120d2bf4.png")
    img_02_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/02/e155_post_seg_120d2bf4_120d2bf4.png")
    img_03_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/03/c2f0120f.png")
    img_03_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/03/a0f3_post_seg_c2f0120f_c2f0120f.png")
    img_04_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/04/6ff7e4db.png")
    img_04_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/04/8349_post_seg_6ff7e4db_6ff7e4db.png")
    img_05_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/05/a658a991.png")
    img_05_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/05/88ae_post_seg_a658a991_a658a991.png")
    img_06_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/06/a8718964.png")
    img_06_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/lung_img/06/acf7_post_seg_a8718964_a8718964.png")

    plt.figure(figsize=(10, 10), dpi=400)
    
    """
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
    
    plt.tight_layout()
    plt.savefig("plot_img_2.png", bbox_inches = "tight", pad_inches = 0)
    plt.show()
    
