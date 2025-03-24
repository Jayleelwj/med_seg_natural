import numpy as np
import os
import ast

txt_input = input().split(', ')
print(txt_input)
img_path, txt_prompt = txt_input[0], txt_input[1]
print(txt_prompt)
txt_prompt = txt_prompt.replace(" ", ",")
print(txt_prompt)
txt_prompt = ast.literal_eval(txt_prompt)
print(txt_prompt[0])