import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from skimage import io, transform
from torch.nn import functional as F
import gradio as gr
import uuid
from PIL import Image, ImageDraw, ImageOps, ImageFont
#SAM library
from segment_anything import sam_model_registry
#langchain library
from langchain.agents.initialize import initialize_agent 
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI  # Changed from OpenAI
from langchain.prompts import PromptTemplate

matplotlib.pyplot.switch_backend('Agg') 

import re

root = "/Users/liwenjie/Documents/mm_openai"

MEDICAL_CHATGPT_PREFIX = """You are a medical image analysis assistant. You help analyze medical images using specialized tools.
Your responses should be clear and focused on the image analysis tasks. When using tools:
1. Always use the exact image filename provided
2. Follow the tool's format requirements exactly
3. Provide clear, concise responses about what you observe

Available tools:"""

MEDICAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: [one of: {tool_names}]
Action Input: [input for the tool]
Observation: [result from the tool]

After successful segmentation:
Thought: The image has been successfully segmented
Final Answer: [describe what was segmented and where]

Remember:
1. Always use exact tool names
2. Stop after successful segmentation
3. Provide clear description in Final Answer
4. Don't continue observations after successful segmentation"""

MEDICAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist. You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Medical ChatGPT is a text language model, Medical ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Medical ChatGPT, Medical ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

def get_new_image_name(org_img_name, func_name="post_seg"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    #convert the image name to uuid string
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
    return os.path.join(head, new_file_name)

def prompts(name, description):
    def decorator(func):
        func.name =name
        func.description = description
        return func
    return decorator

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    
    # Convert list of messages to string if necessary
    if isinstance(history_memory, list):
        # Convert the list of messages to a string format
        history_text = ""
        for message in history_memory:
            # Assuming each message has a type and content
            if hasattr(message, 'type') and hasattr(message, 'content'):
                history_text += f"\n{message.type}: {message.content}"
            else:
                # Fallback for simple string messages
                history_text += f"\n{str(message)}"
        history_memory = history_text.strip()
    
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    
    if n_tokens < keep_last_n_words:
        return history_memory
    
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    
    return '\n' + '\n'.join(paragraphs)

#load the model
class MedSeg:
    #def __init__(self,device="cpu",box=[10,25,400,500]):
    def __init__(self,device="mps"):
        self.device = device
        self.model_path = os.path.join(root,"MedSAM_model","medsam_vit_b.pth")
        self.color_index = 0  # Add color index tracking
        self.colors = [  # Define a list of distinct colors [R,G,B,A]
            [251/255, 252/255, 30/255, 0.6],   # Yellow
            [252/255, 111/255, 30/255, 0.6],   # Orange
            [30/255, 252/255, 50/255, 0.6],    # Green
            [30/255, 252/255, 252/255, 0.6],   # Cyan
            [252/255, 30/255, 242/255, 0.6],   # Pink
            [99/255, 30/255, 252/255, 0.6],    # Purple
            [252/255, 30/255, 30/255, 0.6],    # Red
            [30/255, 127/255, 252/255, 0.6],   # Blue
        ]
        # Add mask history to store all masks for each image
        self.mask_history = {}  # {image_path: [(mask, box_np, color_index)]}
    
    def get_next_color(self):
        """Get next color from the color list and cycle through"""
        color = self.colors[self.color_index]
        self.color_index = (self.color_index + 1) % len(self.colors)
        return color, self.color_index

    def get_bbox(self, img_path, box=None, position=None):
        """
        Get bounding box coordinates based on position description or explicit coordinates
        Args:
            img_path: str, path to the input image
            position: str, one of ['upper left', 'upper right', 'lower left', 'lower right', 'center']
            box: list of 4 integers [x1, y1, x2, y2]
        """
        # Get image dimensions from the input image
        img = Image.open(img_path)
        WIDTH, HEIGHT = img.size
        
        if box is not None:
            return np.array([box])
        
        # Define regions based on position descriptions and actual image dimensions
        regions = {
            'upper left': [0, 0, WIDTH//2, HEIGHT//2],
            'upper right': [WIDTH//2, 0, WIDTH, HEIGHT//2],
            'lower left': [0, HEIGHT//2, WIDTH//2, HEIGHT],
            'lower right': [WIDTH//2, HEIGHT//2, WIDTH, HEIGHT],
            'center': [WIDTH//4, HEIGHT//4, 3*WIDTH//4, 3*HEIGHT//4],
            'default': [WIDTH//4, HEIGHT//4, 3*WIDTH//4, 3*HEIGHT//4]  # Changed default to center
        }
        
        # If position is specified, use corresponding region
        if position and position.lower() in regions:
            return np.array([regions[position.lower()]])
        
        # Default fallback
        return np.array([regions['default']])

    #load the model from the checkpoint
    def load_model(self):
        medsam_model = sam_model_registry['vit_b'](checkpoint=self.model_path)
        medsam_model.to(self.device)
        return medsam_model.eval()
    
    def load_image(self,image_path):
        #2d image only
        img_np =io.imread(image_path)
        if len(img_np.shape)==2:
            img_3c = np.repeat(img_np[:,:,None],3,axis=-1)
        else:
            img_3c = img_np
        H,W,_ = img_3c.shape
        img_1024 = transform.resize(
            img_3c,(1024,1024),order=3,preserve_range=True,anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024-img_1024.min())/np.clip(img_1024.max()-img_1024.min(),a_min=1e-8,a_max=None)
        img_1024 = (torch.tensor(img_1024).float().permute(2,0,1).unsqueeze(0).to(self.device))
        return img_1024,H,W
    
    def show_mask(self,mask,ax):
        color = self.get_next_color()[0]  # Get next color in sequence
        h,w = mask.shape[-2:]
        mask_image = mask.reshape(h,w,1)*np.array(color).reshape(1,1,-1)
        ax.imshow(mask_image)

    def show_mask_bbox(self,mask,inputs, box_np):
        fig, ax = plt.subplots()
        img_np = io.imread(inputs)
        ax.imshow(img_np)
        
        # Show mask with next color in sequence
        self.show_mask(mask, ax)
        
        # Show bounding box in green
        self.box_np = box_np
        x0, y0 = self.box_np[0][0], self.box_np[0][1]
        w, h = self.box_np[0][2]-self.box_np[0][0], self.box_np[0][3]-self.box_np[0][1]
        ax.add_patch(plt.Rectangle((x0,y0), w, h, edgecolor="green", facecolor=(0,0,0,0), lw=2))
        
        plt.tight_layout()
        return fig

    def show_masks(self, img_path):
        """Show all masks for an image"""
        if img_path not in self.mask_history:
            return None
        
        fig, ax = plt.subplots()
        img_np = io.imread(img_path)
        ax.imshow(img_np)
        
        # Show all masks with their respective colors
        for mask, box_np, color_idx in self.mask_history[img_path]:
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * np.array(self.colors[color_idx]).reshape(1, 1, -1)
            ax.imshow(mask_image)
            
            # Show bounding box
            x0, y0 = box_np[0][0], box_np[0][1]
            w, h = box_np[0][2]-box_np[0][0], box_np[0][3]-box_np[0][1]
            ax.add_patch(plt.Rectangle((x0,y0), w, h, edgecolor="green", facecolor=(0,0,0,0), lw=2))
        
        plt.tight_layout()
        return fig

    @prompts(name="Segment the desired area",
             description="useful when you want to segment the desired area from provided medical image."
             "like: segment the image based on given area,"
             "or please segment the assigned image,"
             "or segment all disease issues on this image,"
             "or segment the assigned image given the area [100 200 100 200],"
             "The input to this tool should be a comma separated string of two, representing the image path and the user description.")

    #def medsam_inference(self,image_path):
    def inference(self, inputs):
        txt_input = inputs.split(", ")
        if len(txt_input)==1:
            img_path = txt_input[0]
            self.box_np = self.get_bbox(img_path, position='default')
        else:
            img_path, txt_prompt = txt_input[0], txt_input[1]
            
            # Check for position keywords in txt_prompt
            position_keywords = {
                'upper left': ['upper left', 'top left', 'upleft', 'topleft'],
                'upper right': ['upper right', 'top right', 'upright', 'topright'],
                'lower left': ['lower left', 'bottom left', 'lowleft', 'bottomleft'],
                'lower right': ['lower right', 'bottom right', 'lowright', 'bottomright'],
                'center': ['center', 'middle', 'central']
            }
            
            # Try to match position from text
            matched_position = None
            txt_prompt_lower = txt_prompt.lower()
            for position, keywords in position_keywords.items():
                if any(keyword in txt_prompt_lower for keyword in keywords):
                    matched_position = position
                    break
            
            try:
                # First check if it's a position-based request
                if matched_position:
                    self.box_np = self.get_bbox(img_path, position=matched_position)
                # Then check if it's explicit coordinates
                elif txt_prompt.strip().startswith('[') and txt_prompt.strip().endswith(']'):
                    numbers = [int(x.strip()) for x in txt_prompt[1:-1].split(',')]
                    if len(numbers) == 4:
                        self.box_np = self.get_bbox(img_path, box=numbers)
                    else:
                        raise ValueError("Box coordinates must contain exactly 4 numbers")
                else:
                    # Use default bbox if no position or coordinates found
                    self.box_np = self.get_bbox(img_path, position='default')
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing input: {e}. Using default bbox.")
                self.box_np = self.get_bbox(img_path, position='default')
        
        #txt_prompt, img_path = inputs.split(",")
        #img_path, _ = inputs.split(",")
        #self.box_np = self.get_bbox(box=txt_prompt)
        medsam_model = self.load_model()
        img_1024,H,W = self.load_image(img_path)
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024)
        box_1024 = self.box_np/np.array([W,H,W,H])*1024
        
        # Ensure box_1024 is the correct type before conversion
        box_1024 = box_1024.astype(np.float32)  # Convert to float32
        box_torch = torch.as_tensor(box_1024, device=self.device)
        
        if len(box_torch.shape)==2:
            box_torch = box_torch[:,None,:]  # b,1,4
        
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)

            low_res_logits, _ = medsam_model.mask_decoder(image_embeddings = image_embedding,
            image_pe = medsam_model.prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings = dense_embeddings,multimask_output=False)
        
            low_res_pred = torch.sigmoid(low_res_logits)
            low_res_pred = F.interpolate(
            low_res_pred,
            size=(H,W),
            mode="bilinear",
            align_corners=False
        )
            low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (low_res_pred>0.5).astype(np.uint8)
        #print(io.imread(inputs))
        orig = Image.fromarray(io.imread(img_path))
        updated_image_path = get_new_image_name(img_path)
        #plt.imshow(updated_image_path)
        #plt.savefig(updated_image_path)
        orig.save(updated_image_path)
        fig = self.show_mask_bbox(medsam_seg, updated_image_path, self.box_np)
        plt.savefig(updated_image_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)  # Close the figure to free memory
        
        print(f"Processing Medical Segmentation Task.\nInput Image: {img_path}. Output Image:{updated_image_path}")
        return updated_image_path 


OPENAI_API_KEY = "YOUR_OPENAI_KEY"


class ConversationBot:
    def __init__(self,load_dict):
        print("Initializing Medial ChatGPT, load_dict={}".format(load_dict))
        self.models = {}
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance,e)
                    self.tools.append(Tool(
                        name = func.name,
                        description=func.description,
                        func = func
                    ))
        
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            temperature=1,
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4"
        )
        
        # Initialize memory with a more robust configuration
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='output',
            return_messages=True  # This enables better handling of chat messages
        )
    
    def init_agent(self,lang):
        self.memory.clear()  # Clear memory when starting new conversation
        if lang=="English":
            # Create a custom prompt template that includes all required variables
            prompt = PromptTemplate(
                input_variables=["input", "chat_history", "agent_scratchpad"],
                template=(
                    MEDICAL_CHATGPT_PREFIX + "\n\n" +
                    MEDICAL_CHATGPT_FORMAT_INSTRUCTIONS + "\n\n" +
                    "Previous conversation history:\n{chat_history}\n\n" +
                    "New input: {input}\n\n" +
                    "Since Medical ChatGPT is a text language model, Medical ChatGPT must use tools to observe images rather than imagination.\n" +
                    "The thoughts and observations are only visible for Medical ChatGPT, Medical ChatGPT should remember to repeat important information in the final response for Human.\n" +
                    "Thought: Do I need to use a tool? {agent_scratchpad}"
                )
            )
            
            place = "Enter text and upload an image"
            label_clear = "Clear"
            
            self.agent = initialize_agent(
                self.tools,
                self.llm,
                agent="conversational-react-description",
                verbose=True,
                memory=self.memory,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                agent_kwargs={
                    'prompt': prompt
                }
            )
            
            return gr.update(visible=True), gr.update(visible=False), gr.update(placeholder=place), gr.update(value=label_clear)

    def run_text(self, text, state):
        # Get current memory state
        current_buffer = self.agent.memory.load_memory_variables({})['chat_history']
        truncated_buffer = cut_dialogue_history(current_buffer, keep_last_n_words=500)
        
        # Clear and rebuild memory with truncated history
        self.agent.memory.clear()
        
        if truncated_buffer:
            messages = truncated_buffer.split('\nHuman: ')
            for msg in messages[1:]:  # Skip first empty split
                if '\nAI: ' in msg:
                    human_msg, ai_msg = msg.split('\nAI: ')
                    self.agent.memory.save_context(
                        {"input": human_msg.strip()},
                        {"output": ai_msg.strip()}
                    )
        
        # Process the new message
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        
        # Format the response for display
        response = re.sub('(image/[-\\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.load_memory_variables({})['chat_history']}")
        
        return state, state

    def run_image(self, image, state, txt, lang):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        img = Image.open(image.name)
        img = img.convert('RGB')
        img.save(image_filename, "PNG")

        if lang=="English":
            description = self.models['MedSeg'].inference(image_filename)
            Human_prompt = (
                f'\nHuman: provide a figure named {image_filename}. '
                f'The description is: {description}. '
                'This information helps you to understand this image, but you should use tools to finish following tasks, '
                'rather than directly imagine from my description. If you understand, say "Received". \n'
            )
            AI_prompt = "Received. I understand the image has been provided and I'll use the available tools to analyze it when needed."

        # Update memory with the image context
        self.agent.memory.save_context(
            {"input": Human_prompt},
            {"output": AI_prompt}
        )
        
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.load_memory_variables({})['chat_history']}")
        
        return state, state, f'{txt} {image_filename} '


load_dict = {"MedSeg":"mps"}
c_bot = ConversationBot(load_dict)
with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    lang = gr.Radio(choices=['English'],value=None, label='language')
    chatbot = gr.Chatbot(elem_id="chatbot",label=" Medical ChatGPT")
    state = gr.State([])
    with gr.Row(visible=False) as input_raws:
        with gr.Column(scale=7):
            txt = gr.Textbox(show_label=True,
                             placeholder="Enter text and upload an image, then press Enter",
                             container=False)
            btn = gr.UploadButton(label="Upload",file_types=["image"])
        with gr.Column(scale=0.15,min_width=0):
            clear=gr.Button("Clear")
        #with gr.Column(scale=0.15,min_width=0):
            #btn = gr.UploadButton(label="Upload",file_types=["image"])
        
    lang.change(c_bot.init_agent, [lang], [input_raws, lang, txt, clear])
    txt.submit(c_bot.run_text,[txt,state],[chatbot,state])
    txt.submit(lambda: "",None,txt)
    btn.upload(c_bot.run_image,[btn,state,txt,lang],[chatbot, state,txt])
    clear.click(c_bot.memory.clear)
    clear.click(lambda:[], None, chatbot)
    clear.click(lambda: [], None, state)
demo.launch()


