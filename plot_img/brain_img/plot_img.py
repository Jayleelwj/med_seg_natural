import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


if __name__ == "__main__":
    img_01_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/01/dcd90058.png")
    img_01_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/01/c39c_post_seg_dcd90058_dcd90058.png")
    img_02_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/02/3affbff7.png")
    img_02_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/02/d1b4_post_seg_3affbff7_3affbff7.png")
    img_03_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/03/3affbff7.png")
    img_03_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/03/4fbc_post_seg_3affbff7_3affbff7.png")
    img_04_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/04/e8224416.png")
    img_04_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/04/a4eb_post_seg_e8224416_e8224416.png")
    img_05_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/05/fa8dce99.png")
    img_05_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/05/9916_post_seg_fa8dce99_fa8dce99.png")
    img_06_1 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/06/8e4bee18.png")
    img_06_2 = Image.open("/Users/wenjie/Downloads/ly_openai/MedSAM/plot_img/brain_img/06/6db7_post_seg_8e4bee18_8e4bee18.png")

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
    
