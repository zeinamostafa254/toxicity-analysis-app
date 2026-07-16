def load_caption_model():
    # Only import heavy libraries INSIDE the function
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(image, processor, model):
    # Logic...
    return caption
