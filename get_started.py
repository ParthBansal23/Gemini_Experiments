# -*- coding: utf-8 -*-
"""
Get_started.py

This script demonstrates how to use the Google Gemini API for various tasks including:
- Text and multimodal prompting
- Token counting
- System instructions
- Safety filters
- Multi-turn chat
- Content streaming
- File handling
- And more
"""

import os
from dotenv import load_dotenv
import requests
import pathlib
from PIL import Image
import io
import json
import time
import asyncio
from google import genai
from google.genai import types
from pydantic import TypeAdapter, BaseModel, Field
from typing import Optional

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running this script.")

# Initialize SDK client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Choose a model
MODEL_ID = "gemini-2.5-flash"  # You can change this to other model IDs

# Helper function to print response text
def print_response(response):
    if hasattr(response, 'text') and response.text:
        print(response.text)
    else:
        print(response)

def save_image(image_data, filename):
    """Helper function to save image data to a file"""
    image = Image.open(io.BytesIO(image_data))
    image.save(filename)
    print(f"Image saved as {filename}")

def download_file(url, filename):
    """Helper function to download files"""
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

# Example 1: Basic text generation
def basic_text_generation():
    print("\n--- Basic Text Generation ---")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents="What's the largest planet in our solar system?"
    )
    print_response(response)

# Example 2: Count tokens
def count_tokens():
    print("\n--- Count Tokens ---")
    response = client.models.count_tokens(
        model=MODEL_ID,
        contents="What's the highest mountain in Africa?",
    )
    print(response)

# Example 3: Multimodal prompting
def multimodal_example(request: 'MultimodalGenerationRequest'):
    print("\n--- Multimodal Example ---")
    img_path = pathlib.Path(request.local_image_filename)
    download_file(request.image_url, img_path)
    
    # Open and process image
    image = Image.open(img_path)
    image.thumbnail((512, 512))
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            image,
            request.text_prompt
        ]
    )
    image.show()
    print_response(response)

# Example 4: Configure model parameters
def configure_parameters(request: 'ConfiguredGenerationRequest'):
    print("\n--- Configure Model Parameters ---")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=request.prompt,
        config=request.config_params.to_sdk_config()
    )
    print_response(response)

# 5. Configure safety filters
def configure_safety_filters():
    print("\n--- Configure Safety Filters ---")
    prompt = """
    Write a list of 2 disrespectful things that I might say to the universe after stubbing my toe in the dark.
    """
    # You may need to adjust these enums if the SDK expects them
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ]
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            safety_settings=safety_settings,
        ),
    )
    print_response(response)

# 6. Multi-turn chat
def multi_turn_chat():
    print("\n--- Multi-turn Chat ---")
    system_instruction = """
    You are an expert software developer and a helpful coding assistant.
    You are able to generate high-quality code in any programming language.
    """
    chat_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
    chat = client.chats.create(
        model=MODEL_ID,
        config=chat_config,
    )

    """Use `chat.send_message` to pass a message back and receive a response."""
    # First message
    response = chat.send_message("Write a function that checks if a year is a leap year.")
    print_response(response)

    # Second message
    response = chat.send_message("Okay, write a unit test of the generated function.")
    print_response(response)
    # Save and resume chat
    history_adapter = TypeAdapter(list[types.Content])
    chat_history = chat.get_history()
    json_history = history_adapter.dump_json(chat_history)
    history = history_adapter.validate_json(json_history)
    # Convert each Content object to dict for SDK compatibility (ContentOrDict)
    history_dicts = [c.model_dump() for c in chat_history]
    new_chat = client.chats.create(
        model=MODEL_ID,
        config=chat_config,
        history=history_dicts,
    )
    response = new_chat.send_message("What was the name of the function again?")
    print_response(response)

# 7. Generate JSON
def generate_json():
    print("\n--- Generate JSON ---")
    class Recipe(BaseModel):
        recipe_name: str
        recipe_description: str
        recipe_ingredients: list[str]

    # New Pydantic models for function inputs
    class MultimodalGenerationRequest(BaseModel):
        image_url: str
        text_prompt: str
        local_image_filename: str = "jetpack_downloaded.png" # Default name

    class GenerationCustomConfig(BaseModel):
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        candidate_count: Optional[int] = Field(default=None, ge=1)
        seed: Optional[int] = None
        stop_sequences: Optional[list[str]] = None
        presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
        frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)

        def to_sdk_config(self) -> types.GenerateContentConfig:
            return types.GenerateContentConfig(**self.model_dump(exclude_none=True))

    class ConfiguredGenerationRequest(BaseModel):
        prompt: str
        config_params: GenerationCustomConfig

    response = client.models.generate_content(
        model=MODEL_ID,
        contents="Provide a popular cookie recipe and its ingredients.",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Recipe,
        ),
    )
    if response.text:
        print(json.dumps(json.loads(response.text), indent=4))
    else:
        print("No JSON response.")

# 8. Generate Images
def generate_images():
    print("\n--- Generate Images ---")
    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents='Hi, can create a 3d rendered image of a pig with wings and a top hat flying over a happy futuristic scifi city with lots of greenery?',
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image']
        )
    )
    # Check for candidates and content before accessing parts
    candidates = getattr(response, 'candidates', None)
    if candidates and getattr(candidates[0], 'content', None):
        for part in getattr(candidates[0].content, 'parts', []) or []:
            if hasattr(part, 'text') and part.text:
                print(part.text)
            elif hasattr(part, 'inline_data') and part.inline_data:
                mime = part.inline_data.mime_type
                print(f"Image mime type: {mime}")
                data = part.inline_data.data
                if data:
                    try:
                        img = Image.open(io.BytesIO(data))
                        img.show()
                    except Exception as e:
                        print(f"Could not display image: {e}")

# 9. Generate content stream
def generate_content_stream():
    print("\n--- Generate Content Stream ---")
    for chunk in client.models.generate_content_stream(
        model=MODEL_ID,
        contents="Tell me a story about a lonely robot who finds friendship in a most unexpected place."
    ):
        if hasattr(chunk, 'text') and chunk.text:
            print(chunk.text, end="")
    print()

# 10. Async content generation
async def async_content_generation():
    print("\n--- Async Content Generation ---")
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents="Compose a song about the adventures of a time-traveling squirrel."
    )
    print_response(response)

# 11. File uploads (image, text, pdf, audio)
def file_upload_examples():
    print("\n--- File Upload Examples ---")
    # Image
    img_url = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png"
    img_path = pathlib.Path('jetpack.png')
    img_bytes = requests.get(img_url).content
    img_path.write_bytes(img_bytes)
    file_upload = client.files.upload(file=img_path)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[file_upload, "Write a short and engaging blog post based on this picture."]
    )
    print_response(response)
    # Text
    text_url = "https://storage.googleapis.com/generativeai-downloads/data/a11.txt"
    text_path = pathlib.Path('a11.txt')
    text_bytes = requests.get(text_url).content
    text_path.write_bytes(text_bytes)
    file_upload = client.files.upload(file=text_path)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[file_upload, "Can you give me a summary of this information please?"]
    )
    print_response(response)
    # PDF
    pdf_url = "https://storage.googleapis.com/generativeai-downloads/data/Smoothly%20editing%20material%20properties%20of%20objects%20with%20text-to-image%20models%20and%20synthetic%20data.pdf"
    pdf_path = pathlib.Path('article.pdf')
    pdf_bytes = requests.get(pdf_url).content
    pdf_path.write_bytes(pdf_bytes)
    file_upload = client.files.upload(file=pdf_path)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[file_upload, "Can you summarize this file as a bulleted list?"]
    )
    print_response(response)
    # Audio
    audio_url = "https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3"
    audio_path = pathlib.Path('audio.mp3')
    audio_bytes = requests.get(audio_url).content
    audio_path.write_bytes(audio_bytes)
    file_upload = client.files.upload(file=audio_path)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[file_upload, "Listen carefully to the following audio file. Provide a brief summary"]
    )
    print_response(response)

# Example: Upload and process a video file
def video_file_example():
    print("\n--- Video File Example ---")
    VIDEO_URL = "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
    video_file_name = "BigBuckBunny_320x180.mp4"
    # Download the video file using requests
    video_response = requests.get(VIDEO_URL)
    with open(video_file_name, 'wb') as f:
        f.write(video_response.content)
    print(f"Downloaded {video_file_name}")
    # Upload the file using the API
    video_file = client.files.upload(file=video_file_name)
    print(f"Completed upload: {video_file.uri}")
    # Wait for processing
    while video_file.state == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        if video_file.name:
            video_file = client.files.get(name=video_file.name)
        else:
            print("No video file name available.")
            break
    if video_file.state == "FAILED":
        raise ValueError(video_file.state)
    if video_file.uri:
        print(f'Video processing complete: {video_file.uri}')
    else:
        print('Video processing complete: (no URI)')
    # Ask Gemini about the video
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[video_file, "Describe this video."]
    )
    print(response.text)

# Example: Process a YouTube link
def youtube_link_example():
    print("\n--- YouTube Link Example ---")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents= types.Content(
            parts=[
                types.Part(text="Summarize this video."),
                types.Part(
                    file_data=types.FileData(file_uri='https://www.youtube.com/watch?v=WsEQjeZoEng')
                )
            ]
        )
    )
    print(response.text)

# Example: Use URL context
def url_context_example():
    print("\n--- URL Context Example ---")
    prompt = """
    Compare recipes from https://www.food.com/recipe/homemade-cream-of-broccoli-soup-271210
    and from https://www.allrecipes.com/recipe/13313/best-cream-of-broccoli-soup/,
    list the key differences between them.
    """
    tools = [types.Tool(url_context=types.UrlContext())]
    config = types.GenerateContentConfig(tools=tools)
    response = client.models.generate_content(
        contents=[prompt],
        model=MODEL_ID,
        config=config
    )
    print(response.text)

# Example: Use context caching
def context_caching_example():
    print("\n--- Context Caching Example ---")
    system_instruction = """
      You are an expert researcher who has years of experience in conducting systematic literature surveys and meta-analyses of different topics.
      You pride yourself on incredible accuracy and attention to detail. You always stick to the facts in the sources provided, and never make up new facts.
      Now look at the research paper below, and answer the following questions in 1-2 sentences.
    """
    urls = [
        'https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
    ]
    # Download files
    pdf_bytes = requests.get(urls[0]).content
    pdf_path = pathlib.Path('2312.11805v3.pdf')
    pdf_path.write_bytes(pdf_bytes)
    pdf_bytes = requests.get(urls[1]).content
    pdf_path = pathlib.Path('2403.05530.pdf')
    pdf_path.write_bytes(pdf_bytes)
    # Upload the PDFs using the File API
    uploaded_pdfs = []
    uploaded_pdfs.append(client.files.upload(file='2312.11805v3.pdf'))
    uploaded_pdfs.append(client.files.upload(file='2403.05530.pdf'))
    # Create a cache with a 60 minute TTL
    cached_content = client.caches.create(
        model=MODEL_ID,
        config=types.CreateCachedContentConfig(
            display_name='research papers',
            system_instruction=system_instruction,
            contents=uploaded_pdfs,
            ttl="3600s",
        )
    )
    print("Cache created.")
    # List available cache objects
    for cache in client.caches.list():
        print(cache)
    # Use a cache
    response = client.models.generate_content(
        model=MODEL_ID,
        contents="What is the research goal shared by these research papers?",
        config=types.GenerateContentConfig(cached_content=cached_content.name)
    )
    print(response.text)
    # Delete a cache
    if cached_content.name:
        result = client.caches.delete(name=cached_content.name)
        print("Cache deleted.")
    else:
        print("No cache name to delete.")

# Example: Get text embeddings
def get_text_embeddings():
    print("\n--- Get Text Embeddings ---")
    TEXT_EMBEDDING_MODEL_ID = "gemini-embedding-exp-03-07"
    response = client.models.embed_content(
        model=TEXT_EMBEDDING_MODEL_ID,
        contents=[
            "How do I get a driver's license/learner's permit?",
            "How do I renew my driver's license?",
            "How do I change my address on my driver's license?"
        ],
        config=types.EmbedContentConfig(output_dimensionality=512)
    )
    if hasattr(response, 'embeddings') and response.embeddings:
        embeddings = response.embeddings
        if embeddings:
            print(f"Number of embeddings: {len(embeddings)}")
            values = getattr(embeddings[0], 'values', None)
            if values:
                print(f"Length of first embedding: {len(values)}")
                print(f"First 4 values: {values[:4]}")
            else:
                print("No values in first embedding.")
        else:
            print("No embeddings returned.")
    else:
        print("No embeddings returned.")

# Example: Function Calling
def function_calling_example():
    print("\n--- Function Calling Example ---")
    get_destination = types.FunctionDeclaration(
        name="get_destination",
        description="Get the destination that the user wants to go to",
        parameters={
            "type": "OBJECT",
            "properties": {
                "destination": {
                    "type": "STRING",
                    "description": "Destination that the user wants to go to",
                },
            },
        },
    )

    destination_tool = types.Tool(
        function_declarations=[get_destination],
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents="I'd like to travel to Paris.",
        config=types.GenerateContentConfig(
            tools=[destination_tool],
            temperature=0,
            ),
    )

    if response.candidates and \
       len(response.candidates) > 0 and \
       response.candidates[0].content and \
       len(response.candidates[0].content.parts) > 0 and \
       hasattr(response.candidates[0].content.parts[0], 'function_call'):
        
        function_call = response.candidates[0].content.parts[0].function_call
        print("\nFunction call details (response.candidates[0].content.parts[0].function_call):")
        print(function_call)
    else:
        print("Could not extract function call from the response or function_call attribute is missing.")

# Main function
def main():
    basic_text_generation()
    count_tokens()
    multimodal_example(
        MultimodalGenerationRequest(
            image_url="https://storage.googleapis.com/generativeai-downloads/data/jetpack.png",
            text_prompt="Write a short and engaging blog post based on this picture.",
            local_image_filename="jetpack.png" # Explicitly setting to original name for consistency
        )
    )
    configure_parameters(
        ConfiguredGenerationRequest(
            prompt="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
            config_params=GenerationCustomConfig(
                temperature=0.4,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
                stop_sequences=["STOP!"],
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
        )
    )
    configure_safety_filters()
    multi_turn_chat()
    generate_json()
    generate_images()
    generate_content_stream()
    asyncio.run(async_content_generation())
    file_upload_examples()
    video_file_example()
    youtube_link_example()
    url_context_example()
    context_caching_example()
    get_text_embeddings()
    function_calling_example()

if __name__ == "__main__":
    main()


"""## Next Steps

### Useful API references:

Check out the [Google GenAI SDK](https://github.com/googleapis/python-genai) for more details on the new SDK.

### Related examples

For more detailed examples using Gemini models, check the [Quickstarts folder of the cookbook](https://github.com/google-gemini/cookbook/tree/main/quickstarts/). You'll learn how to use the [Live API](./Get_started_LiveAPI.ipynb), juggle with [multiple tools](../examples/LiveAPI_plotting_and_mapping.ipynb) or use Gemini 2.0 [spatial understanding](./Spatial_understanding.ipynb) abilities.

Also check the [Gemini thinking models](./Get_started_thinking.ipynb) that explicitly showcases its thoughts summaries and can manage more complex reasonings.
"""