from flask import Blueprint, render_template, request, redirect, url_for, flash
import google.generativeai as genai
from google_api_key import google_api_key
import os
import mimetypes

# Create a blueprint
medical_bp = Blueprint('medical', __name__)

genai.configure(api_key=google_api_key)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_prompt = (
    "You are a professional medical image analyst and health advisor with expertise in diagnostics, disease interpretation, and wellness planning. "
    "Analyze the uploaded medical image thoroughly to detect any possible conditions, diseases, or abnormalities. "
    "If any condition is identified, provide a structured and concise report including:\n"
    "1. **Detected Condition(s)**: Name and short explanation.\n"
    "2. **Medical Summary**: Reasoning behind your diagnosis based on image features.\n"
    "3. **Recommended Diet**: Nutritional guidelines suitable for the condition.\n"
    "4. **Medication (if applicable)**: Suggested common medications or treatment direction (non-prescription).\n"
    "5. **Workout/Physical Activity Advice**: Exercises or precautions suitable for the patient’s condition.\n"
    "6. **Additional Suggestions**: Lifestyle changes or follow-up suggestions.\n"
    "Also, extract and list any visible text found in the image, such as dates, labels, or patient notes."
)



model = genai.GenerativeModel(
    model_name="gemma-3-27b-it", # or use this model  learnlm-2.0-flash-experimental
    generation_config=generation_config,
    safety_settings=safety_settings
)


@medical_bp.route('/medical_analysis', methods=['GET', 'POST'])
def medical_analysis():
    response_text = None
    uploaded_image_path = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file:
            # Save the uploaded file temporarily
            uploaded_image_path = os.path.join('static', 'uploads', file.filename)
            os.makedirs(os.path.dirname(uploaded_image_path), exist_ok=True)
            file.save(uploaded_image_path)

            # Read binary content of the file
            file.stream.seek(0)
            image_data = file.read()

            try:
                # Create prompt parts: image + system prompt text
                mime_type, _ = mimetypes.guess_type(file.filename)
                prompt_parts = [
                    {"mime_type": mime_type or "image/jpeg", "data": image_data},
                    {"text": system_prompt}
                ]

                # Generate response from Gemini
                response = model.generate_content(prompt_parts)

                if response and hasattr(response, 'text'):
                    response_text = response.text
                else:
                    # print("Model response is empty or invalid.")
                    flash("Model didn't return a response. Please try again.", "danger")

            except Exception as e:
                print("Error during analysis:", str(e))
                flash(f"Error during analysis: {str(e)}", "danger")

        else:
            # print("No file uploaded.")
            flash("Please upload a medical report or image.", "warning")

    return render_template('medical_analysis.html', response_text=response_text, image_path=uploaded_image_path)
















# from flask import Blueprint, render_template, request, flash
# from google_api_key import google_api_key  # Your OpenRouter API key
# import base64
# import requests
# import os
# import mimetypes

# medical_bp = Blueprint('medical', __name__)

# generation_config = {
#   "temperature": 1,
#   "top_p": 0.95,
#   "top_k": 0,
#   "max_output_tokens": 2000,  # Reduced to fit free-tier token limit
# }


# # Use your OpenRouter API key here
# OPENROUTER_API_KEY = google_api_key

# # System prompt for medical image analysis
# system_prompt = (
#     "You are an expert medical image analyst. Analyze the uploaded image for possible "
#     "conditions, abnormalities, and provide a brief medical summary. Highlight any visible text in the image as well."
# )

# @medical_bp.route('/medical_analysis', methods=['GET', 'POST'])
# def medical_analysis():
#     response_text = None
#     uploaded_image_path = None

#     if request.method == 'POST':
#         file = request.files.get('file')

#         if file:
#             uploaded_image_path = os.path.join('static', 'uploads', file.filename)
#             os.makedirs(os.path.dirname(uploaded_image_path), exist_ok=True)
#             file.save(uploaded_image_path)

#             file.stream.seek(0)
#             image_data = file.read()
#             base64_image = base64.b64encode(image_data).decode('utf-8')

#             try:
#                 headers = {
#                     "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#                     "Content-Type": "application/json",
#                     "HTTP-Referer": "http://localhost:5000"  # Replace with your deployed domain if needed
#                 }

#                 mime_type, _ = mimetypes.guess_type(file.filename)

#                 payload = {
#                     "model": "google/gemini-pro-1.5",  # ✅ Valid model for image + text
#                     "messages": [
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": system_prompt},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {
#                                         "url": f"data:{mime_type};base64,{base64_image}"
#                                     }
#                                 }
#                             ]
#                         }
#                     ]
#                 }

#                 response = requests.post(
#                     "https://openrouter.ai/api/v1/chat/completions",
#                     headers=headers,
#                     json=payload
#                 )

#                 try:
#                     data = response.json()
#                     if "choices" in data and data["choices"]:
#                         response_text = data["choices"][0]["message"]["content"]
#                     else:
#                         print("Unexpected API response:", data)
#                         flash("Unexpected API response. Check console for details.", "danger")
#                 except Exception as e:
#                     print("Failed to parse response JSON:", str(e))
#                     flash("Failed to process API response. Please try again.", "danger")


#             except Exception as e:
#                 print("Error during analysis:", str(e))
#                 flash(f"Error during analysis: {str(e)}", "danger")
#         else:
#             flash("Please upload a medical report or image.", "warning")

#     return render_template('medical_analysis.html', response_text=response_text, image_path=uploaded_image_path)
