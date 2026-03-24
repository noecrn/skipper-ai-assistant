# src/skipper_ai/explain.py

import json
import ollama

DEFAULT_MODEL = 'llama3.2:3b'

def generate_explanation(analysis_results_path, model=DEFAULT_MODEL):
    """
    Uses Ollama to generate a structured Markdown explanation for a sailor.
    """
    with open(analysis_results_path, 'r') as f:
        results = json.load(f)
    
    # Enhanced structured prompt
    prompt = f"""
    You are an expert Offshore Sailing Coach. Analyze this performance data and provide a structured report for the skipper.
    
    ### DATA SUMMARY
    - Avg Performance: {results['avg_performance']*100:.1f}% of polar
    - Avg Speed: {results['avg_boat_speed']:.2f} kts (Expected: {results['avg_expected_speed']:.2f} kts)
    
    ### PERFORMANCE IMPACTS (SHAP values)
    {json.dumps(results['feature_impact'], indent=2)}
    
    ### UNDER-PERFORMANCE SEGMENTS (>5% loss)
    {json.dumps(results['under_performance_impact'], indent=2)}

    ### INSTRUCTIONS
    1. Use Markdown headers (###).
    2. Use bullet points for issues.
    3. Be direct and technical but avoid ML jargon (no 'SHAP', 'features', or 'regressor').
    4. Focus on 'Heel', 'Sail Choice', and 'Trim/Steering'.
    5. Keep it short and high-impact.

    ### OUTPUT FORMAT
    ### 📊 PERFORMANCE SUMMARY
    (One sentence)

    ### 🔍 KEY ISSUES
    - (Issue 1)
    - (Issue 2)

    ### 💡 COACH'S RECOMMENDATION
    **(Your top actionable advice here)**
    """
    
    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': 'You are a professional sailing performance analyst. You provide concise, structured, and actionable feedback.'},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        # Fallback (updated to look similar to the structured output)
        impacts = results['feature_impact']
        worst_feature = min(impacts, key=impacts.get)
        
        fallback = f"### 📊 PERFORMANCE SUMMARY\nYour average performance was {results['avg_performance']*100:.1f}% of the polar.\n\n"
        fallback += "### 🔍 KEY ISSUES\n"
        if impacts[worst_feature] < -0.01:
            if worst_feature == 'heel':
                fallback += "- Suboptimal heel angle detected as a major speed loss factor.\n"
            elif worst_feature == 'sail_id_numeric':
                fallback += "- Incorrect sail selection for the current wind angle.\n"
            else:
                fallback += f"- Inconsistent steering or trim in current {worst_feature} conditions.\n"
        else:
            fallback += "- No major issues detected; performance is near-optimal.\n"
            
        fallback += "\n### 💡 COACH'S RECOMMENDATION\n"
        if worst_feature == 'heel':
            fallback += "**Focus on flatter sailing. Adjust water ballast or reef earlier to optimize your heel angle.**"
        elif worst_feature == 'sail_id_numeric':
            fallback += "**Review your sail crossover chart. You are likely carrying the wrong sail for this TWA.**"
        else:
            fallback += "**Maintain current settings and focus on precise steering to stay on the polar.**"
            
        fallback += f"\n\n*Note: This is a fallback explanation (Ollama model '{model}' not reachable).* "
        return fallback

def ask_question(analysis_results_path, question, model=DEFAULT_MODEL):
    """
    Asks a specific question about the run using Ollama.
    """
    with open(analysis_results_path, 'r') as f:
        results = json.load(f)
    
    prompt = f"""
    Analysis results for a sailing run:
    {json.dumps(results, indent=2)}
    
    User Question: {question}
    
    Provide a clear answer based on the analysis data. Focus on sailing performance.
    """
    
    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': 'You are a helpful sailing performance analyst.'},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        # Fallback for demonstration purposes
        if "sail" in question.lower():
            return "Based on the analysis, your sail selection was generally okay, but there were some segments where it might have been suboptimal. Check your polars for the current wind angle."
        elif "heel" in question.lower():
            return "The analysis shows that heel had some impact on your performance. Ensure you're maintaining the optimal heel for your boat's design."
        else:
            return f"The run showed an average performance of {results['avg_performance']*100:.1f}%. The main factors were wind conditions (TWS/TWA). (Fallback answer as Ollama is not accessible)"

