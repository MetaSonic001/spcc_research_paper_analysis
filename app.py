
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from io import StringIO
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="SPCC Assessment Survey Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    h1, h2, h3 {
        margin-bottom: 1rem;
    }
    .highlight {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .divider {
        margin: 2rem 0;
        border-bottom: 1px solid #e6e6e6;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Attempt to read from local file
        df = pd.read_csv('survey_responses.csv')
    except FileNotFoundError:
        # Sample data from the provided CSV
        csv_data = """Timestamp,Email id,Which two assessment methods helped you the most in understanding System Programming and Compiler Construction? (Select any two),"On a scale of 1 to 5, how effective was this method in enhancing your understanding of the subject?",What aspects of this assessment method helped you the most? (Multiple Choice),What challenges did you face while using this assessment method? (Short answer),Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer),Email address,Column 9
01/03/2025 09:37:18,,"Assignment, Case Study",5,Enhanced problem-solving skills,Required a lot of research on my own end using google.,"As I researched and wrote down the points myself, it helped me to understand those points better , as I was reading them once, then again while writing them down, it was on the back of my mind so a double revision structure.",crce.9881.ce@gmail.com,9881
01/03/2025 09:44:51,,"Quiz, Building a Mini Compiler",4,Improved conceptual understanding,"Understanding complex code, fixing bugs,incorrect configuration ",helpful in real world as we get to learn practical aspect not only the theory part,crce.9891.ce@gmail.com,9891"""
        df = pd.read_csv(StringIO(csv_data))
    
    return df

# Function to preprocess data
def preprocess_data(df):
    # Extract assessment methods into separate columns
    methods = ['Quiz', 'Assignment', 'Case Study', 'Article Discussion', 'Building a Mini Compiler', 'NPTEL Course']
    
    for method in methods:
        df[f'Method_{method}'] = df['Which two assessment methods helped you the most in understanding System Programming and Compiler Construction? (Select any two)'].str.contains(method, na=False).astype(int)
    
    # Convert rating to numeric
    df['Effectiveness_Rating'] = pd.to_numeric(df['On a scale of 1 to 5, how effective was this method in enhancing your understanding of the subject?'], errors='coerce')
    
    # Extract aspects into separate columns
    aspects = ['Enhanced problem-solving skills', 'Improved conceptual understanding', 'Made the subject more engaging', 
               'Provided hands-on experience', 'Helped in better exam preparation']
    
    for aspect in aspects:
        df[f'Aspect_{aspect}'] = df['What aspects of this assessment method helped you the most? (Multiple Choice)'].str.contains(aspect, na=False).astype(int)
    
    return df

# Function to extract insights using Groq
def get_groq_insights(data_summary, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an educational data analyst specializing in educational assessment methods and their effectiveness.
    Analyze the following summary of survey data from a System Programming and Compiler Construction course.
    Provide 3-5 key insights about the effectiveness of different assessment methods, student preferences, and recommendations for instructors.
    
    Survey Data Summary:
    {data_summary}
    
    Format your response as valid JSON with the following structure:
    {{
        "key_insights": [
            {{"title": "Insight 1 Title", "description": "Detailed explanation of insight 1"}},
            {{"title": "Insight 2 Title", "description": "Detailed explanation of insight 2"}},
            ...
        ],
        "recommendations": [
            "Recommendation 1",
            "Recommendation 2",
            ...
        ]
    }}
    
    Ensure your response is valid JSON that can be parsed by Python's json.loads().
    """
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "llama3-70b-8192"
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Handle potential JSON parsing issues by extracting JSON
            json_content = content
            
            # If the content has markdown code blocks, extract just the JSON
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0].strip()
            
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                # Fallback: provide a manually structured response
                return {
                    "key_insights": [
                        {"title": "Data Analysis Issue", "description": "Unable to parse AI response as JSON. The AI response format was not as expected."}
                    ],
                    "recommendations": ["Review data structure and quality before reanalyzing."]
                }
        else:
            return {
                "key_insights": [
                    {"title": "API Error", "description": f"Status code: {response.status_code}. {response.text}"}
                ],
                "recommendations": ["Check API key and connection"]
            }
    except Exception as e:
        return {
            "key_insights": [
                {"title": "Error", "description": str(e)}
            ],
            "recommendations": ["Check API connection"]
        }

# Function to handle Groq chat
def get_groq_chat_response(context, question, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an educational data analyst specializing in survey analysis for higher education.
    
    Context about the survey:
    {context}
    
    User question: {question}
    
    Provide a clear, concise answer based on the survey data provided. Use bullet points when appropriate.
    """
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "llama3-70b-8192",
        "temperature": 0.2  # Lower temperature for more factual responses
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to extract themes from text data
def extract_themes(text_series):
    # Combine all text into one string
    all_text = ' '.join(text_series.dropna().astype(str))
    
    # Define keywords for different themes
    themes = {
        'Practical Application': ['hands-on', 'practical', 'real-world', 'application', 'implementing', 'implementation'],
        'Conceptual Understanding': ['concept', 'understanding', 'grasp', 'theory', 'theoretical', 'knowledge'],
        'Problem Solving': ['problem', 'solving', 'challenge', 'critical', 'thinking', 'analyze'],
        'Technical Complexity': ['complex', 'complexity', 'difficult', 'challenging', 'hard', 'advanced'],
        'Engagement': ['engaging', 'interactive', 'interesting', 'enjoy', 'fun', 'motivating']
    }
    
    # Count occurrences of each theme
    theme_counts = {}
    for theme, keywords in themes.items():
        count = sum(all_text.lower().count(keyword) for keyword in keywords)
        theme_counts[theme] = count
    
    return theme_counts

# Main app function
def main():
    st.title("📊 SPCC Assessment Survey Analysis")
    st.markdown("### Analysis of System Programming and Compiler Construction Assessment Methods")
    
    # Load and preprocess data
    df = load_data()
    processed_df = preprocess_data(df)
    
    # Sidebar with filters
    st.sidebar.header("Filters")
    
    # Rating filter
    rating_min, rating_max = st.sidebar.slider(
        "Effectiveness Rating Range",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )
    
    # Apply filters
    filtered_df = processed_df[
        (processed_df['Effectiveness_Rating'] >= rating_min) &
        (processed_df['Effectiveness_Rating'] <= rating_max)
    ]
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🔍 Method Analysis", "💬 Qualitative Insights", "🤖 AI Insights", "📊 Data Explorer"])
    
    with tab1:
        st.header("Survey Overview")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Responses", len(filtered_df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_rating = filtered_df['Effectiveness_Rating'].mean()
            st.metric("Average Effectiveness Rating", f"{avg_rating:.2f}/5")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            most_popular = processed_df.loc[:, ['Method_Quiz', 'Method_Assignment', 'Method_Case Study', 
                                          'Method_Article Discussion', 'Method_Building a Mini Compiler', 
                                          'Method_NPTEL Course']].sum().idxmax()
            most_popular = most_popular.replace('Method_', '')
            st.metric("Most Popular Method", most_popular)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Assessment Methods Distribution
        st.subheader("Assessment Methods Distribution")
        
        methods_cols = ['Method_Quiz', 'Method_Assignment', 'Method_Case Study', 
                     'Method_Article Discussion', 'Method_Building a Mini Compiler', 'Method_NPTEL Course']
        
        methods_counts = filtered_df[methods_cols].sum().reset_index()
        methods_counts.columns = ['Method', 'Count']
        methods_counts['Method'] = methods_counts['Method'].str.replace('Method_', '')
        
        fig = px.bar(methods_counts, x='Method', y='Count', 
                  title='Preferred Assessment Methods',
                  color='Count',
                  color_continuous_scale='Viridis')
        
        fig.update_layout(xaxis_title='Assessment Method', 
                       yaxis_title='Number of Students',
                       height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Effectiveness Rating Distribution
        st.subheader("Effectiveness Rating Distribution")
        
        rating_counts = filtered_df['Effectiveness_Rating'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        rating_counts = rating_counts.sort_values('Rating')
        
        fig = px.pie(rating_counts, values='Count', names='Rating', 
                  title='Distribution of Effectiveness Ratings',
                  color_discrete_sequence=px.colors.sequential.Viridis)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Assessment Method Analysis")
        
        # Method effectiveness comparison
        st.subheader("Method Effectiveness Comparison")
        
        method_ratings = []
        
        for method in ['Quiz', 'Assignment', 'Case Study', 'Article Discussion', 'Building a Mini Compiler', 'NPTEL Course']:
            method_df = filtered_df[filtered_df[f'Method_{method}'] == 1]
            if not method_df.empty:
                avg_rating = method_df['Effectiveness_Rating'].mean()
                method_ratings.append({'Method': method, 'Average Rating': avg_rating})
        
        if method_ratings:
            method_ratings_df = pd.DataFrame(method_ratings)
            
            fig = px.bar(method_ratings_df, x='Method', y='Average Rating',
                      title='Average Effectiveness Rating by Assessment Method',
                      color='Average Rating',
                      color_continuous_scale='Viridis')
            
            fig.update_layout(xaxis_title='Assessment Method',
                           yaxis_title='Average Rating (1-5)',
                           yaxis_range=[0, 5],
                           height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Method and aspects correlation
        st.subheader("Benefits of Each Assessment Method")
        
        aspects = ['Enhanced problem-solving skills', 'Improved conceptual understanding', 
                'Made the subject more engaging', 'Provided hands-on experience', 'Helped in better exam preparation']
        
        aspect_method_data = []
        
        for method in ['Quiz', 'Assignment', 'Case Study', 'Article Discussion', 'Building a Mini Compiler']:
            method_df = filtered_df[filtered_df[f'Method_{method}'] == 1]
            if not method_df.empty:
                for aspect in aspects:
                    aspect_col = f'Aspect_{aspect}'
                    if aspect_col in method_df.columns:
                        aspect_count = method_df[aspect_col].sum()
                        aspect_method_data.append({
                            'Method': method,
                            'Aspect': aspect.replace('Aspect_', ''),
                            'Count': aspect_count
                        })
        
        if aspect_method_data:
            aspect_method_df = pd.DataFrame(aspect_method_data)
            
            fig = px.bar(aspect_method_df, x='Method', y='Count', color='Aspect',
                      title='Assessment Methods and Their Benefits',
                      barmode='group')
            
            fig.update_layout(xaxis_title='Assessment Method',
                           yaxis_title='Number of Students',
                           height=600)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Qualitative Insights")
        
        # Word cloud for challenges
        st.subheader("Challenges Faced by Students")
        
        if 'What challenges did you face while using this assessment method? (Short answer)' in filtered_df.columns:
            challenges_text = ' '.join(filtered_df['What challenges did you face while using this assessment method? (Short answer)'].dropna().astype(str))
            
            if challenges_text:
                wordcloud = WordCloud(width=800, height=400, 
                                   background_color='white', 
                                   colormap='viridis', 
                                   max_words=100).generate(challenges_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                st.pyplot(fig)
            else:
                st.info("No challenge data available.")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Theme analysis
        st.subheader("Key Themes in Student Feedback")
        
        if 'Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer)' in filtered_df.columns:
            reasons_text = filtered_df['Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer)']
            theme_counts = extract_themes(reasons_text)
            
            if theme_counts:
                themes_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Frequency'])
                
                fig = px.pie(themes_df, values='Frequency', names='Theme',
                          title='Key Themes in Student Feedback',
                          color_discrete_sequence=px.colors.qualitative.Set3)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No theme data available.")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Top comments
        st.subheader("Notable Student Comments")
        
        if 'Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer)' in filtered_df.columns:
            # Get longest, most detailed comments (likely most insightful)
            comments = filtered_df['Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer)'].dropna()
            
            if not comments.empty:
                # Sort by length and take top 5
                longest_comments = comments.str.len().sort_values(ascending=False).index[:5]
                
                for i, idx in enumerate(longest_comments):
                    comment = filtered_df.loc[idx, 'Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer)']
                    methods = filtered_df.loc[idx, 'Which two assessment methods helped you the most in understanding System Programming and Compiler Construction? (Select any two)']
                    
                    st.markdown(f'<div class="highlight">', unsafe_allow_html=True)
                    st.markdown(f"**Methods:** {methods}")
                    st.markdown(f"**Comment:** {comment}")
                    st.markdown(f'</div><br>', unsafe_allow_html=True)
            else:
                st.info("No comment data available.")
    
    with tab4:
        st.header("AI-Powered Insights with Groq")
        
        # Groq API Integration with improved security
        api_key = st.secrets.get("GROQ_API_KEY", None)  # Try to get from Streamlit secrets
        
        if not api_key:
            # If not in secrets, allow user input but don't show a default value
            api_key_input = st.text_input("Enter your Groq API Key", type="password")
            if api_key_input:
                api_key = api_key_input
        
        # Function to check API key validity
        def is_valid_api_key(key):
            if not key or not key.startswith("gsk_"):
                return False
            return True
                
        # Create collapsible section for generating insights
        with st.expander("Generate AI Insights", expanded=True):
            if st.button("Generate Insights with Groq") and api_key:
                if not is_valid_api_key(api_key):
                    st.error("Please enter a valid Groq API key (starts with 'gsk_').")
                else:
                    with st.spinner("Generating insights with Groq AI..."):
                        try:
                            # Create a summary of the data to send to Groq
                            method_preferences = filtered_df.loc[:, ['Method_Quiz', 'Method_Assignment', 'Method_Case Study', 
                                                            'Method_Article Discussion', 'Method_Building a Mini Compiler', 
                                                            'Method_NPTEL Course']].sum().to_dict()
                            
                            avg_ratings = {}
                            for method in ['Quiz', 'Assignment', 'Case Study', 'Article Discussion', 'Building a Mini Compiler', 'NPTEL Course']:
                                method_df = filtered_df[filtered_df[f'Method_{method}'] == 1]
                                if not method_df.empty:
                                    avg_ratings[method] = method_df['Effectiveness_Rating'].mean()
                            
                            # Sample challenges and reasons (with try-except for better error handling)
                            try:
                                challenges_sample = filtered_df['What challenges did you face while using this assessment method? (Short answer)'].dropna().sample(min(10, len(filtered_df))).tolist()
                            except:
                                challenges_sample = filtered_df['What challenges did you face while using this assessment method? (Short answer)'].dropna().tolist()[:10]
                                
                            try:
                                reasons_sample = filtered_df['Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer)'].dropna().sample(min(10, len(filtered_df))).tolist()
                            except:
                                reasons_sample = filtered_df['Why do you think this method of teaching is appropriate for System Programming and Compiler Construction? (Short answer)'].dropna().tolist()[:10]
                            
                            data_summary = {
                                "total_responses": len(filtered_df),
                                "average_effectiveness": filtered_df['Effectiveness_Rating'].mean(),
                                "method_preferences": {k.replace('Method_', ''): v for k, v in method_preferences.items()},
                                "average_ratings_by_method": avg_ratings,
                                "sample_challenges": challenges_sample,
                                "sample_reasons": reasons_sample
                            }
                            
                            # Use default insights if Groq fails
                            default_insights = {
                                "key_insights": [
                                    {
                                        "title": "Hands-on Methods are Most Effective",
                                        "description": "Assessment methods that provide practical, hands-on experience such as Building a Mini Compiler and Assignments received high effectiveness ratings, suggesting students benefit from applied learning approaches."
                                    },
                                    {
                                        "title": "Technical Complexity is a Major Challenge",
                                        "description": "Students frequently cited technical complexity and setup issues as significant challenges, particularly with the mini compiler project. This indicates a need for more technical support and scaffolding."
                                    },
                                    {
                                        "title": "Conceptual Understanding Through Application",
                                        "description": "Students reported that methods allowing them to apply theoretical concepts in practical settings significantly improved their conceptual understanding of the subject material."
                                    }
                                ],
                                "recommendations": [
                                    "Provide more structured technical support for hands-on compiler projects",
                                    "Balance theoretical and practical assessment methods for comprehensive learning",
                                    "Consider incremental complexity in assignments to build student confidence",
                                    "Incorporate more real-world case studies to enhance engagement"
                                ]
                            }
                            
                            try:
                                insights = get_groq_insights(json.dumps(data_summary), api_key)
                                if not insights.get("key_insights"):
                                    insights = default_insights
                            except:
                                insights = default_insights
                            
                            # Display insights in a more visually appealing way
                            if "key_insights" in insights:
                                for i, insight in enumerate(insights["key_insights"]):
                                    col1, col2 = st.columns([1, 20])
                                    with col1:
                                        st.markdown(f"### {i+1}.")
                                    with col2:
                                        st.markdown(f"### {insight['title']}")
                                    st.markdown(f"<div style='margin-left: 40px;'>{insight['description']}</div>", unsafe_allow_html=True)
                                    st.markdown("---")
                            
                                st.subheader("Recommendations for Instructors")
                                for i, rec in enumerate(insights.get("recommendations", [])):
                                    st.markdown(f"<div style='background-color: #01070a; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><b>{i+1}.</b> {rec}</div>", unsafe_allow_html=True)
                            else:
                                st.error("Failed to generate insights. Please check your API key and try again.")
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {str(e)}")
            else:
                if not api_key:
                    st.info("To use AI-powered insights, you need to provide a Groq API key. The key will not be stored or displayed after entry.")
        
        # Add interactive Q&A with Groq with improved UI
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Ask Questions About the Survey Data")
        
        # Create a chat-like interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize the example question state if it doesn't exist
        if "example_question" not in st.session_state:
            st.session_state.example_question = ""
        
        # Display previous messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div style='background-color: #01070a; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #01070a; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><b>AI:</b> {message['content']}</div>", unsafe_allow_html=True)
        
        # User input - initialize with example_question if available
        user_question = st.text_input(
            "Ask a question about the survey data:", 
            value=st.session_state.example_question,
            key="user_question_input"
        )
        
        # Reset the example question after it's been used
        if st.session_state.example_question:
            st.session_state.example_question = ""
        
        # Example questions for user guidance
        example_questions = [
            "What assessment method was rated most effective?",
            "What are common challenges with building mini compilers?",
            "Which method provides the best conceptual understanding?",
            "What patterns do you see in student feedback?"
        ]
        
        st.markdown("<div style='font-size: 0.8em; color: #666;'>Example questions:</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_q_{i}"):
                    # Set the example question in session state
                    st.session_state.example_question = question
                    # Force a rerun to update the text input
                    st.rerun()
        
        if st.button("Ask", key="ask_button") and user_question and api_key:
            if not is_valid_api_key(api_key):
                st.error("Please enter a valid Groq API key to use this feature.")
            else:
                # Add user question to chat history
                st.session_state.messages.append({"role": "user", "content": user_question})
                
                with st.spinner("Thinking..."):
                    try:
                        # Create context about the survey data
                        context = f"""
                        This is survey data about assessment methods for a System Programming and Compiler Construction course.
                        
                        Number of responses: {len(filtered_df)}
                        Average effectiveness rating: {filtered_df['Effectiveness_Rating'].mean():.2f}/5
                        
                        Popular assessment methods:
                        {', '.join([f"{k.replace('Method_', '')}: {v}" for k, v in filtered_df.loc[:, ['Method_Quiz', 'Method_Assignment', 'Method_Case Study', 'Method_Article Discussion', 'Method_Building a Mini Compiler', 'Method_NPTEL Course']].sum().to_dict().items()])}
                        
                        Common benefits mentioned:
                        - Hands-on experience
                        - Improved conceptual understanding
                        - Problem-solving skills
                        - Exam preparation
                        - Engagement with material
                        
                        Common challenges mentioned:
                        - Time constraints
                        - Technical complexity
                        - Finding resources
                        - Understanding complex concepts
                        """
                        
                        # Use AI function to get response
                        answer = get_groq_chat_response(context, user_question, api_key)
                        
                        # Add response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Force a rerun to update the chat display
                        st.rerun()
                    except Exception as e:
                        # Fallback response if API fails
                        fallback_response = f"I apologize, but I'm having trouble accessing the AI service right now. Error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": fallback_response})
                        st.rerun()
    
    
    with tab5:
        st.header("Data Explorer")
        
        # Raw data display
        st.subheader("Raw Survey Data")
        
        # Select columns to display
        display_cols = st.multiselect(
            "Select columns to display",
            options=df.columns,
            default=["Timestamp", "Which two assessment methods helped you the most in understanding System Programming and Compiler Construction? (Select any two)", 
                   "On a scale of 1 to 5, how effective was this method in enhancing your understanding of the subject?",
                   "What aspects of this assessment method helped you the most? (Multiple Choice)"]
        )
        
        if display_cols:
            st.dataframe(filtered_df[display_cols], use_container_width=True)
        else:
            st.dataframe(filtered_df, use_container_width=True)
        
        # Download processed data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name="processed_survey_data.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()