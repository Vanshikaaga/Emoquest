from google import generativeai as genai
import json

# Initialize the Gemini client with your API key
genai.configure(api_key="AIzaSyBmlu40tXUPVGjaTFo0V0h0RpUuNrT3KFk")  # Replace with your actual API key

def get_gemini_response_legacy(json_data):
    model_name = "gemini-2.0-flash"  # Or another available model in your version
    model = genai.GenerativeModel(model_name)

    prompt = f"""
You are an expert in psychological behavioral analysis through affective computing and personality modeling. Based on the provided structured behavioral data, analyze the individual's affective behavior and extract psychologically meaningful patterns and traits.

The data consists of four behavioral dimensions:
1. **Avoidant Behavior**: Interaction style with emotional or perceived threats.
2. **Startle Responsiveness**: Sensitivity to sudden or unexpected stimuli.
3. **Confrontational Reactivity**: Emotional expression during frustration or confrontation.
4. **Expressive Positivity**: Tendency to express positive emotions externally.

### Your Objective:
Use this data to generate a psychologically insightful, user-friendly report. The report should help the individual better understand their emotional and behavioral tendencies, how these patterns influence their daily life and relationships, and which areas might benefit from further awareness or development.

---

### 📄 Report Format to Follow:


The report should be written in second person, addressing the individual directly. Use the following structure:
First, Give scores for user's Confrontational Reactivity behaviour,Startle Responsivenes behaviour, Expressive Positivity, Defensive Avoidant Behavior  give this rating from 0 to 10,  0 being the left end corner.
#### 1. **Emotional Behavior Profile **
A short, readable description of the person’s affective behavior style. It should include:
- How they tend to process and express emotions.
- How they are likely to respond in personal, social, and professional situations.
- Brief notes on strengths and areas of growth.
- Tone should be warm, supportive, and reflective.

#### 2. **Overall Behavioral Patterns**
- Summarize dominant behavioral tendencies (e.g., avoidance, impulsiveness, expressiveness, restraint).
- Describe how these patterns influence emotional regulation, social dynamics, and adaptability.

#### 3. **Behavioral Tendencies & Underlying Psychological Traits**
- Identify inferred psychological traits (e.g., emotional sensitivity, self-control, suppression, resilience).
- Ground them in observable tendencies across the four behavioral axes.
- Use psychologically relevant language but keep it readable.

#### 4. **Concerns or Red Flags **
- Briefly mention any potential behavioral patterns that may pose a challenge to the individual's well-being or relationships.
- Include constructive, non-alarming suggestions if certain behaviors (e.g., emotional suppression, avoidant coping, heightened reactivity) could be worth exploring further.
- Keep the tone constructive and solutions-oriented.
The reccomendation that you give should be brief, to the point and easy to understand

---

Important Notes:
- Do **not** mention anything about "games" or "EmoQuest." Focus entirely on the person and their behavioral data.
- Keep the tone **formal, brief, reflective, and encouraging**.
- Avoid listing raw metrics. Your goal is to **translate data into insight**.
- at the start of report donot mention any line like okay i will analyze, simply start with report in desired format with first displaying scores of Confrontational Reactivity behaviour,Startle Responsivenes behaviour, Expressive Positivity, Defensive Avoidant Behavior in this format \*\*Startle Responsiveness Behavior:\*\*\s*\d+(?:/10)?
Here is the data for analysis:
{json.dumps(json_data, indent=2)}
"""

    try:
        response = model.generate_content(prompt)
        full_response = ""
        for chunk in response.parts:
            full_response += chunk.text
        print(full_response)
        return full_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage (assuming you have 'consolidated_data' as discussed earlier)
# with open('your_data.json', 'r') as f:
#     loaded_data = json.load(f)
#     # --- Transformation Logic (as discussed before) ---
#     consolidated_data = { ... }
#
#     if consolidated_data:
#         report_text = get_gemini_response_legacy(consolidated_data)
#         # ... process the report_text ...
#     else:
#         print("Could not consolidate data.")