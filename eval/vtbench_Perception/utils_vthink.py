from PIL import Image
from qwen_vl_utils import process_vision_info # Ensure this utility script is available
import re
from sandbox import execute_code_in_sandbox
import os
import copy
from transformers import  AutoProcessor
MAX_ITERATIONS = 1 # Safety break for the loop
MAX_NEW_TOKENS_PER_STEP = 8192 * 2 # Max tokens for each generation step


SYSTEM_PROMPT = '''You are an expert AI assistant that translates user instructions into Python code for marking specific points on an image. Your primary task is to generate code that marks one or more points as requested.

Instructions:

Mandatory Code Generation: You must respond by writing executable Python code. Do not provide answers in plain text or refuse to write code.

Strict Task Limitation: Mark Points Only: Your sole function is to mark points. You must refuse to perform any other image manipulation task like drawing continuous lines, cropping, rotating, or applying filters. Your generated code should only contain logic for marking points.

Library and Method: Use the Pillow (PIL) library, especially the ImageDraw module. To mark a point, draw a small, visible shape like a filled circle or a cross at the target coordinates. A small ellipse is often a good choice. Avoid using draw.point(), as the resulting pixel is usually too small to be seen.

Coordinate Accuracy: Crucially, you must first determine the appropriate coordinates based on the user's request and the provided image size/context. Do not use placeholder or example coordinates (e.g., (10, 10)) if the user's request implies a specific location (e.g., "mark the center," "put a dot in the top-right corner").

Code Wrapping: All Python code snippets must be wrapped in <code> and ````python` tags as shown below: 
    <code>
    ```python
    # Your Python code here
    ```
    </code>

Input/Output: Your code should assume it receives the input image path via a variable named image_path. It must save the modified image to a new file named 'image output path' and, as the final step, print the path of this new file.

Focus: Your code should only perform the point marking action requested by the user.'''
def remove_unpickable_values(dictionary):
    import pickle

    def is_pickable(obj):
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, TypeError, AttributeError):
            return False


def generate_prompt_final_qa(user_question, user_image_path):
    # Construct the prompt based on the given requirements
    try:
        with Image.open(user_image_path) as img:
            user_image_size = f"{img.width}x{img.height}"
    except Exception as e:
        user_image_size = "Unable to determine (error reading image)"

    prompt = f"""
<image>\n
{user_question}\n
You should translates user instructions into Python code for marking specific points on an image. Your primary task is to generate code that marks one or more points as requested.

If your code processes the image, use: 
    image_path = '{user_image_path}'

### **Output Format (strict adherence required):**

***Code Wrapping:** All Python code snippets **must** be wrapped in `<code>` and ````python` tags as shown below:
    <code>
    ```python
    # Your Python code here
    ```
    </code>
    """
    return prompt




SPECIAL_STRING_LIST=["</code>", "</answer>","</sandbox_output>","<sandbox_output>"]

def run_evaluation(user_question: str, initial_image_url_or_path: str, aux_outpath, model, processor):
    """
    Runs the iterative evaluation for a given question and image.
    """
    current_image_path_for_code = initial_image_url_or_path # This is the path sandbox code will use
    
    # Conversation history for the model
    # The first image is always the one provided by the user for the initial prompt context
    # Subsequent images are results of code execution
    conversation_history = []
    
    # conversation_history.append({'role': 'system', 'content': [{"type": "text", "text": SYSTEM_PROMPT}]})
    conversation_history.append({'role': 'system', 'content': [{"type": "text", "text": SYSTEM_PROMPT}]})


    # The very first "user" turn contains the initial problem setup.
    # The prompt from utils.py guides the model's overall behavior.
    initial_prompt_text = generate_prompt_final_qa(user_question, current_image_path_for_code)
    
    conversation_history.append({
        "role": "user",
        "content": [
            {"type": "image", "image": initial_image_url_or_path}, # The visual context
            {"type": "text", "text": initial_prompt_text},         # The task and instructions
        ]
    })
    first_path = "null"
    # Conditions for generation pause: reach special tokens like </code> (code generation end) or </answer>
    for iteration in range(MAX_ITERATIONS):
        # saved for multiple rounds of thinking and calling.
        generated_content = []
        # Maintain context for sandbox environment. 
        previous_execution_context = {}
        
        text_prompt_for_model = processor.apply_chat_template(
            conversation_history, # Pass the entire history
            tokenize=False,
            add_generation_prompt=(iteration==0)
        )

        
        if iteration != 0:
            if text_prompt_for_model.endswith("<|im_end|>\n"):
                text_prompt_for_model = text_prompt_for_model[:-len("<|im_end|>\n")]
                        
        # `process_vision_info` needs the message list that `apply_chat_template` would process
        # to correctly identify images and their positions.
        image_inputs, video_inputs = process_vision_info(conversation_history)
        inputs = processor(
            text=[text_prompt_for_model],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate model response
        print("Generating model response...")
        # bkup context. roll back when we fail to execute the generated code.
        last_execution_context = copy.deepcopy(remove_unpickable_values(previous_execution_context))
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS_PER_STEP,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id, # Qwen typically uses eos as pad
            stop_strings=SPECIAL_STRING_LIST,
            tokenizer=processor.tokenizer
        )
        
        # Decode only the newly generated tokens
        input_token_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[0, input_token_len:]
        generated_text_segment = processor.decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(f"LLM (segment {iteration+1}):\n{generated_text_segment}")
                
        # Case 1: directly give answer
        if "<answer>" in generated_text_segment:
            print(f"Case 1: directly give answer")
            generated_content.append(
                {"type": "text", "text": generated_text_segment},
            )
        
        
        # Case 2: Code block generated.
        # parse current result. Two cases: reach </code> or reach </answer>
        # code_regex = re.compile(r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>', re.IGNORECASE)
        code_regex = re.compile(r"'''python\s*([\s\S]*?)\s*'''", re.IGNORECASE)
        code_match = code_regex.search(generated_text_segment)
        if code_match:
            code_match = code_match
        else:
            code_regex2 = re.compile(r"<code>\s*([\s\S]*?)\s*</code>", re.IGNORECASE)
            code_match = code_regex2.search(generated_text_segment)
        
        # execute code and return result.
        if code_match:
            code_to_execute = code_match.group(1).strip()
            
            print(f"\033[31m--- Found Code Block ---\n{code_to_execute}\n-------------------------\033[0m")

            processed_img_paths, captured_stdout, error_msg, current_execution_context = execute_code_in_sandbox(
                code_to_execute, current_image_path_for_code,temp_output_dir=aux_outpath,
                previous_execution_context=previous_execution_context
            )
            print("------------------------------")
            print("------------------------------")
            print("------------------------------")
            print("------------------------------")
            print("------------------------------")
            print(processed_img_paths)
            print("------------------------------")
            print("------------------------------")
            print("------------------------------")
            print("------------------------------")
            print("------------------------------")
            if not processed_img_paths:
                previous_execution_context = last_execution_context
                print(f'{error_msg}')
                generated_content += [
                    {"type": "text", "text": generated_text_segment},
                    {"type": "text", "text": "<sandbox_output>"}
                ]
                generated_content.append({"type": "text", "text": error_msg})
                generated_content.append({"type": "text", "text": "</sandbox_output>"})
                conversation_history.append({"role": "assistant", "content": generated_content})
                continue      
            
            has_valid_images = False
            generated_content += [
                                {"type": "text", "text": generated_text_segment},
                                {"type": "text", "text": "<sandbox_output>"}
                            ]
            first_path = processed_img_paths[0]
            if os.path.exists(first_path):
                # Iterate through each path in the list
                for img_path in processed_img_paths:
                    if os.path.exists(img_path):
                        if not has_valid_images: # Add text segments only once per sandbox output block
                            has_valid_images = True
                        generated_content.append({"type": "image", "image": img_path})                            
            else:
                generated_content.append({"type": "text", "text": first_path})

            if has_valid_images or not os.path.exists(first_path):
                generated_content.append({"type": "text", "text": "</sandbox_output>"})
            else:
                # pandayin: a failed code execution/generation doesn't count as an intermedia step.
                print('skip this generation due to error and adapt the temperature')
                continue
        else:
            # wo code. wo </answer>, assume repetition generated, break.
            if "<answer>" not in generated_text_segment:
                print('Warning: wo code. wo </answer>')
                generated_content.append(
                {"type": "text", "text": generated_text_segment},
                )
                if conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": generated_content})
                # If the last message was 'assistant', append to its last text content item
                elif conversation_history[-1]["role"] == "assistant":
                    conversation_history.append({"role": "assistant", "content": generated_content})
        
                print(generated_text_segment)
                break
                    

        
        # Update conversation_history with the latest generated segment
        # If the last message was 'user', start a new 'assistant' message
        if conversation_history[-1]["role"] == "user":
            conversation_history.append({"role": "assistant", "content": generated_content})
        # If the last message was 'assistant', append to its last text content item
        elif conversation_history[-1]["role"] == "assistant":
            conversation_history.append({"role": "assistant", "content": generated_content})
        
        
        # --- Check for final answer tag if no code was processed in this segment ---
        if "<answer>" in generated_text_segment:
            print("\033[32m--- Final answer tag found. ---\033[0m")
            break
        
        # If the model produced an EOS token and no code/answer, it might be finished
        if generated_ids[0][-1] == processor.tokenizer.eos_token_id:
            print("\033[32m--- Model generated EOS and no further actions (code/answer). Assuming completion. ---\033[0m")
            break
        

    print(f"\n--- End of processing (max iterations: {MAX_ITERATIONS}, actual: {iteration + 1}) ---")
    
    # Print the full conversation
    print("\n=== Full Conversation History ===")
    final_assistant_response = ""
    for msg_idx, msg in enumerate(conversation_history):
        print(f"Role: {msg['role']}")
        current_content_str = ""
        for content_item_idx, item in enumerate(msg['content']):
            if item['type'] == 'text':
                print(f"  Content Text Part {content_item_idx+1}:\n{item['text']}")
                current_content_str += item['text']
            elif item['type'] == 'image':
                print(f"  Content Image Part {content_item_idx+1}: {item['image']}")
                current_content_str += f"\n[IMAGE: {item['image']}]\n" # For combined log
        if msg['role'] == 'assistant':
            final_assistant_response += current_content_str # Get the last full response from assistant
        print("--------------------")

    # Extract content within <answer> tags from the final assistant response
    answer_match = re.search(r"<answer>(.*?)<answer>", final_assistant_response, re.DOTALL)
    if answer_match:
        final_answer = answer_match.group(1).strip()
        print(f"\nExtracted Final Answer:\n{final_answer}")
    else:
        final_answer = "No answer tag found in the final output."
        print(f"\n{final_answer}\nFull assistant response was:\n{final_assistant_response}")

    return final_assistant_response, final_answer, first_path
