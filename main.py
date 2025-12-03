import os
import requests
import json
from typing import Annotated, Literal, TypedDict
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# --- Configuration & Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# --- Helper Functions ---

def get_amadeus_token():
    """Authenticates with Amadeus to get a temporary access token."""
    if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
        return None
    
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET
    }
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"Error fetching Amadeus token: {e}")
        return None

# --- Tools Definitions ---

@tool
def check_weather(city: str):
    """
    Fetches the current weather and upcoming forecast for a specific city.
    Useful for helping the user decide the best day to visit.
    """
    # Step 1: Geocode the city to get lat/lon (Using Open-Meteo Geocoding for zero-config runnability)
    # In a production Google-centric app, you would use Google Geocoding API here.
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    
    try:
        geo_res = requests.get(geo_url).json()
        if not geo_res.get("results"):
            return f"Could not find coordinates for {city}."
        
        location = geo_res["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        
        # Step 2: Fetch Weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,precipitation_probability_max,weathercode",
            "timezone": "auto"
        }
        
        w_res = requests.get(weather_url, params=params).json()
        
        daily = w_res.get("daily", {})
        times = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])
        probs = daily.get("precipitation_probability_max", [])
        
        forecast_report = f"Weather Forecast for {city}:\n"
        for i in range(min(5, len(times))):
            forecast_report += f"- {times[i]}: {temps[i]}°C, Rain Chance: {probs[i]}%\n"
            
        return forecast_report

    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def google_places_search(query: str, location: str = None):
    """
    Searches for places, restaurants, hidden gems, or tourist spots using Google Places API.
    'query' should be what to look for (e.g., "Italian restaurants", "Hidden gems").
    'location' is the city or area name.
    """
    if not GOOGLE_API_KEY:
        return "Error: Google API Key not found."

    # Using the Text Search (New) API logic or standard Text Search
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.userRatingCount"
    }
    
    text_query = f"{query} in {location}" if location else query
    payload = {"textQuery": text_query}

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        
        places = data.get("places", [])
        if not places:
            return "No places found."
            
        result = f"Top results for '{text_query}':\n"
        for place in places[:5]:
            name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "No address")
            rating = place.get("rating", "N/A")
            result += f"- {name} (Rating: {rating}/5): {address}\n"
            
        return result
    except Exception as e:
        return f"Error connecting to Google Places: {e}"

@tool
def get_map_view(location: str, zoom: int = 14):
    """
    Generates a Google Maps Static API URL for a given location.
    Returns a URL that the user can click to see the map.
    """
    if not GOOGLE_API_KEY:
        return "Error: Google API Key not found."
    
    # URL Encode the location
    import urllib.parse
    encoded_loc = urllib.parse.quote(location)
    
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={encoded_loc}&zoom={zoom}&size=600x400&maptype=roadmap&key={GOOGLE_API_KEY}"
    
    return f"Here is a map view of {location}: {url}"

@tool
def find_hotels(city: str):
    """
    Finds hotels in a specific city using the Amadeus API.
    """
    token = get_amadeus_token()
    if not token:
        return "Error: Could not authenticate with Amadeus API. Check API keys."

    try:
        # Step 1: Find the City IATA code
        city_url = "https://test.api.amadeus.com/v1/reference-data/locations"
        headers = {"Authorization": f"Bearer {token}"}
        city_params = {"subType": "CITY", "keyword": city}
        
        city_res = requests.get(city_url, headers=headers, params=city_params).json()
        
        if not city_res.get("data"):
            return f"Could not find IATA code for city: {city}"
            
        iata_code = city_res["data"][0]["iataCode"]
        
        # Step 2: Search for hotels in that city
        hotel_url = f"https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
        hotel_params = {"cityCode": iata_code}
        
        hotel_res = requests.get(hotel_url, headers=headers, params=hotel_params).json()
        
        hotels = hotel_res.get("data", [])
        if not hotels:
            return f"No hotels found in {city} ({iata_code})."
            
        output = f"Hotels found in {city} ({iata_code}):\n"
        for hotel in hotels[:5]:
            name = hotel.get("name", "Unknown Hotel")
            hotel_id = hotel.get("hotelId", "")
            output += f"- {name} (ID: {hotel_id})\n"
            
        return output

    except Exception as e:
        return f"Error querying Amadeus: {e}"

@tool
def get_destination_photo(query: str):
    """
    Fetches a travel photo for a location using Unsplash API.
    """
    if not UNSPLASH_ACCESS_KEY:
        return "Error: Unsplash API Key not found."

    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "client_id": UNSPLASH_ACCESS_KEY,
        "per_page": 1,
        "orientation": "landscape"
    }
    
    try:
        res = requests.get(url, params=params).json()
        results = res.get("results", [])
        if not results:
            return "No photos found."
        
        photo = results[0]
        desc = photo.get("description") or photo.get("alt_description") or "Travel photo"
        image_url = photo.get("urls", {}).get("regular")
        credit = photo.get("user", {}).get("name")
        
        return f"![{desc}]({image_url})\n*Photo by {credit} on Unsplash*"
    except Exception as e:
        return f"Error fetching photo: {e}"

# --- LangGraph Setup ---

# List of tools the agent can use
tools = [check_weather, google_places_search, get_map_view, find_hotels, get_destination_photo]

# Initialize the LLM (GPT-4o or GPT-3.5-turbo)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define State with 'add_messages' reducer to preserve history
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Define the Agent Node
def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Add Edges
workflow.add_edge(START, "agent")

# Conditional Edge: If the agent calls a tool, go to 'tools', otherwise END
workflow.add_conditional_edges(
    "agent",
    tools_condition
)

# Edge back from tools to agent (to interpret results)
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile(checkpointer=MemorySaver())

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("🌍 AI Travel Agent Initialized. Type 'quit' to exit.")
    print("You can ask about weather, hotels, places to eat, or maps for any location.")
    
    # Generate a random thread ID for session memory
    thread_id = "user-session-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        print("🤖 Agent is thinking...", end="", flush=True)
        
        # Stream events from the graph
        events = app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values"
        )
        
        # Print the final response
        final_message = None
        for event in events:
            if "messages" in event:
                final_message = event["messages"][-1]
                
        print("\r" + " " * 30 + "\r", end="") # Clear "thinking" text
        
        if final_message and final_message.content:
            print(f"🤖 Agent: {final_message.content}")
        elif final_message and final_message.tool_calls:
             # This happens if the stream ends exactly on a tool call (rare in 'values' mode logic above, but good safety)
             print(f"🤖 Agent: (Executing tools: {[tc['name'] for tc in final_message.tool_calls]})")