# simple_gemini_test.py
import google.generativeai as genai
import os
GEMINI_API_KEY="AIzaSyB11KCkWC322TYpOX4LRb9E7l7KT_mfde8"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Tell me a short joke.")
        print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")
