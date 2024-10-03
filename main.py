import os
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch
from openai import OpenAI
from config import OPENAI_API_KEY

# OpenAI クライアントの初期化
client = OpenAI(api_key=OPENAI_API_KEY)

# 画像分類のためのパイプラインを設定
pipe = pipeline("image-classification", model="google/vit-base-patch16-224")

# 画像プロセッサとモデルを設定
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

def get_image_from_path(image_path):
    return Image.open(image_path)

def classify_image(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

def generate_story_gpt4o_mini(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "あなたは創造的な物語作家です。日本語で短い物語を書いてください。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def translate_classification(classification):
    # 英語の分類を日本語に翻訳するためのプロンプト
    prompt = f"Translate the following image classification result to Japanese: {classification}"
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

def image_storyteller(image_path):
    # 画像を取得
    image = get_image_from_path(image_path)
    
    # 画像を分類
    classification = classify_image(image)
    japanese_classification = translate_classification(classification)
    print(f"画像の内容: {japanese_classification}")
    
    # 分類結果に基づいてストーリーを生成
    prompt = f"{japanese_classification}についての短くて魅力的な物語を作成してください。物語は創造的で想像力豊かで、すべての年齢層に適したものにしてください。約300文字程度でお願いします。"
    story = generate_story_gpt4o_mini(prompt)
    print("生成されたストーリー:")
    print(story)

# アプリケーションの使用例
image_path = os.path.expanduser("pic/sample.jpg")
image_storyteller(image_path)