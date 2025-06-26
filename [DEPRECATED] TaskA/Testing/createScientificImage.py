import requests
import urllib.parse
import os
import time


save_dir = "face_images"
os.makedirs(save_dir, exist_ok=True)
face_prompts = [
    "Close-up portrait of a smiling elderly woman, soft lighting",
    "Cropped photo of a teenage boy with freckles and messy hair",
    "Close-up of a young African girl with braided hair",
    "Face of a middle-aged Asian man, neutral expression",
    "Portrait of a baby boy with big eyes and chubby cheeks",
    "Elderly man with wrinkles, close-up face, high detail",
    "Smiling Indian woman with traditional jewelry, close crop",
    "Close-up of a red-haired child laughing",
    "Teen girl with round glasses, cropped face photo",
    "Bald man with beard, close-up portrait",
    "Old Asian woman with kind eyes, face-focused photo",
    "Middle-aged Hispanic man, close-up headshot",
    "Freckled teenage girl, curly hair, face image",
    "Black man with dreadlocks and serious expression, cropped tightly",
    "White woman with blonde hair and piercing eyes",
    "Child with mixed ethnicity, colorful background, face only",
    "Man with a scar across his cheek, high detail portrait",
    "Artistic close-up of a woman with face paint",
    "Close crop of elderly man smiling with missing tooth",
    "Teen boy with headphones, intense look",
    "Toddler girl with flower crown, close-up face",
    "Old man with long white beard, side-lit portrait",
    "Profile of woman in soft sunset light, cropped tight",
    "Girl with blue dyed hair and nose piercing",
    "Bearded man with monocle, vintage style close-up",
    "Woman with vitiligo, close-up headshot",
    "Black woman with afro and bright makeup",
    "Child with chocolate-covered face, smiling",
    "Man in traditional Middle Eastern attire, close face",
    "Elderly Native American man with feathers",
    "Teen boy with acne and serious look",
    "Asian baby girl with short hair and chubby face",
    "Woman with striking green eyes, head turned slightly",
    "Close-up of crying child, emotional expression",
    "Man with cyberpunk tattoos and neon lighting",
    "Cute girl in winter hat, face close-up",
    "Smiling grandpa with thick glasses, warm tones",
    "Woman in hijab, calm and confident expression",
    "Elderly nun with kind wrinkles, cropped portrait",
    "Baby sleeping peacefully, extreme close-up",
    "Teen with braces smiling confidently",
    "Artist with face smudged in paint, headshot",
    "Asian teen girl in glasses, face-focused photo",
    "Old man with eye patch, strong expression",
    "Woman with silver hair and nose ring, headshot",
    "Dark-skinned child with curious eyes, close-up",
    "Biker with helmet lifted, close portrait",
    "Laughing girl in school uniform, cropped image",
    "Father and daughter cropped together, smiling",
    "Worried old man looking into distance",
    "Surprised baby boy with big smile",
    "Woman with tears and smeared mascara",
    "Young man with bleached hair and piercings",
    "Girl with flowers in her hair, soft expression",
    "Boy with messy mud on face, happy",
    "Woman with intense stare, high contrast",
    "Close-up of a boy with Down syndrome, smiling",
    "Portrait of a young woman with alopecia",
    "Laughing man in a cowboy hat",
    "Serious businesswoman, formal suit close-up",
    "Face of woman with freckles under sunlight",
    "Kid with superhero face paint, cropped close",
    "Sikh man with turban, traditional portrait",
    "Boy in rain with water on his face",
    "Senior woman with grey afro smiling",
    "Angry young man in street fashion",
    "Close-up of teen girl biting lip playfully",
    "Baby with a pacifier, looking curious",
    "Woman in forest light, serene expression",
    "Punk teen with green mohawk",
    "Face of a blind man with dark glasses",
    "Laughing Asian grandma, happy eyes",
    "Close-up of child blowing a kiss",
    "Young man meditating, peaceful expression",
    "Teen girl with dyed purple hair and freckles",
    "Close crop of female boxer, bruised but smiling",
    "Child eating watermelon, juice on face",
    "Face of a wise elderly monk",
    "Close-up of military veteran in uniform",
    "Boy with golden retriever licking his cheek",
    "Young girl in ballet costume, smiling",
    "Close-up of boy with hearing aid",
    "Sad expression on middle-aged man",
    "Woman with headset, video gamer expression",
    "Child peeking behind curtain, shy smile",
    "Old lady with colorful sari and joyful grin",
    "Teen with gothic makeup and black lipstick",
    "Man with tribal face paint, intense look",
    "Smiling elderly couple cropped together",
    "Girl with freckles under sunhat",
    "Close-up of exhausted athlete post-run",
    "Woman with flower petal on her cheek",
    "Young monk with shaved head, serene face",
    "Toddler in glasses with toothy smile",
    "Close-up of boy in hoodie at night",
    "Face of a traditional flamenco dancer",
    "Teen girl with braces and big smile",
    "Old man laughing with joy, close crop",
    "Close-up of female scientist in lab goggles",
    "Boy in glasses, surprised expression",
    "Woman in rain, water dripping on face",
    "Portrait of person with ambiguous gender, stylish look",
]


for idx, prompt in enumerate(face_prompts, start=1):
    print(f"Fetching image {idx}/100: {prompt}")
    
    params = {
        "width": 250,
        "height": 250,
        "seed": 42,
        "model": "flux",
        "token": "fEWo70t94146ZYgk",
        "referrer": "elixpoart",
        "private": True,
        "nologo": True,
    }

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    try:
        response = requests.get(url, params=params, timeout=300)
        response.raise_for_status()
        filename = os.path.join(save_dir, f"face_{idx:03}.jpg")
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Saved: {filename}")
        time.sleep(2) 

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image {idx}: {e}")
