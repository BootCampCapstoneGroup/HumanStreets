from google import genai

# Initialize the client
client = genai.Client(api_key="AIzaSyDCeArBfPxL0pZYmiv6tzGdTi8TDtrFUDE")

# Use the correct model string
response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Explain quantum physics to a five-year-old."
)

print(response.text)

