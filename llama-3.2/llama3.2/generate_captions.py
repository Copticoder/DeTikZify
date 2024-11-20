import base64
import requests
from datasets import load_dataset
from io import BytesIO
import json
import time
import logging
import os 

# Configure logging
logging.basicConfig(filename='process_log2.log', level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger()

# Function to log messages and flush immediately
def log_and_print(message):
   print(message)
   log.info(message)

# Load the dataset
ds = load_dataset("nllg/datikz", split="test")

# API configuration
API_KEY = "YOUR_API_KEY_HERE"
headers = {
   "Content-Type": "application/json",
   "api-key": API_KEY,
}

ENDPOINT = "YOUR_ENDPOINT_URL_HERE"

# Constants
TOKEN_LIMIT_PER_MINUTE = 30000
# Load existing captions if the file exists
captions = {}
# if os.path.exists("captions2.json"):
#     with open("captions2.json", "r") as f:
#         captions = json.load(f)

# Dictionary to store captions with URI as the key
captions = {}
total_input_tokens = 0
total_output_tokens = 0
tokens_in_last_minute = 0
start_time = time.time()

# Variables for progress tracking
request_times = []
num_images = len(ds)

# Loop over only the first third of the dataset
# start_idx = 2 * (num_images // 3)
start_idx = 442
end_idx = num_images  # End at the last item in the dataset

# Loop over the last third of the dataset
for idx, item in enumerate(ds.select(range(start_idx, end_idx)), start=start_idx):
   image = item["image"]
   uri = item["uri"]

   # Convert the image to JPEG and then to base64
   buffered = BytesIO()
   image.save(buffered, format="JPEG")
   jpeg_image_data = buffered.getvalue()
   encoded_image = base64.b64encode(jpeg_image_data).decode('ascii')

   # Prepare the payload with the encoded image
   payload = {
       "messages": [
           {
               "role": "system",
               "content": [
                   {
                       "type": "text",
                       "text": "You are an AI specialized in generating captions for scientific figures which would help to write the tikz code for these figures"
                   }
               ]
           },
           {
               "role": "user",
               "content": [
                   {
                       "type": "text",
                       "text": "\n"
                   },
                   {
                       "type": "image_url",
                       "image_url": {
                           "url": f"data:image/jpeg;base64,{encoded_image}"
                       }
                   },
                   {
                       "type": "text",
                       "text": "give me a caption describing this image which would help me write the tikz code for it"
                   }
               ]
           },
       ],
       "temperature": 0.3,
       "top_p": 0.95,
       "max_tokens": 800
   }

   # Retry logic for generating caption
   while True:
       try:
           request_start = time.time()
           response = requests.post(ENDPOINT, headers=headers, json=payload)
           response.raise_for_status()
           response_json = response.json()
           
           # Extract caption
           caption = response_json["choices"][0]["message"]["content"]
           log_and_print(f"Image {idx}: Caption generated successfully.")

           # Add the URI and caption to the dictionary
           captions[uri] = caption

           # Save the dictionary to JSON file after each addition
           with open("captions_test2.json", "w") as f:
               json.dump(captions, f, indent=4)

           # Extract and count tokens
           input_tokens = response_json["usage"]["prompt_tokens"]
           output_tokens = response_json["usage"]["completion_tokens"]
           total_input_tokens += input_tokens
           total_output_tokens += output_tokens

           # Update tokens in the last minute
           tokens_in_last_minute += input_tokens + output_tokens

           # Track request time for progress estimation
           request_end = time.time()
           request_time = request_end - request_start
           request_times.append(request_time)

           # Print progress
           elapsed_time = time.time() - start_time
           avg_time_per_request = sum(request_times) / len(request_times)
           remaining_requests = end_idx - (idx + 1)
           estimated_time_remaining = avg_time_per_request * remaining_requests
           log_and_print(f"Iteration {idx} - Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")
           log_and_print(f"Total Input Tokens So Far: {total_input_tokens}, Total Output Tokens So Far: {total_output_tokens}")
           log_and_print(f"Elapsed Time: {elapsed_time:.2f}s, Estimated Time Remaining: {estimated_time_remaining:.2f}s")

           # Check if we exceeded the rate limit
           if tokens_in_last_minute > TOKEN_LIMIT_PER_MINUTE:
               # Wait until a minute has passed since the start time
               wait_time = 60 - elapsed_time
               if wait_time > 0:
                   log_and_print(f"Token limit exceeded. Waiting for {wait_time:.2f} seconds.")
                   time.sleep(wait_time)
               # Reset tracking for the next minute
               tokens_in_last_minute = 0
               start_time = time.time()
           
           # Exit the retry loop on success
           break

       except requests.RequestException as e:
           if response.status_code == 400:
               log_and_print(f"Skipping image {idx} due to 400 Bad Request.")
               break  # Skip this image if it's a 400 error

           elif response.status_code == 429:
               log_and_print(f"Rate limit exceeded for image {idx}. Retrying...")
               time.sleep(5)  # Wait before retrying

           else:
               log_and_print(f"Failed to generate caption for image {idx}. Retrying... Error: {e}")
               time.sleep(5)  # Optional delay between retries to avoid excessive requests

log_and_print(f"Final Total Input Tokens: {total_input_tokens}, Final Total Output Tokens: {total_output_tokens}")
log_and_print("Captions saved to captions.json")