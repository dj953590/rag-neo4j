import os
from dotenv import load_dotenv
from langchain.chains import SequentialChain, LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dynaconf import settings

api_groq_key = settings.GROQ_API_KEY

# Initialize the GROQ language model
llm = ChatGroq(temperature=0.5, model_name=f"llama-3.1-70b-versatile", api_key=api_groq_key)

# Conversation memory for persistent context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

destination_prompt = PromptTemplate(
    input_variables=["interests"],
    template="Your role is travel assistant. you are tasked to provide list of "
             "best travel destinations for someone interested in {interests}?"
)

destination_chain = LLMChain(
    prompt=destination_prompt,
    memory=memory,
    llm=llm
)

hotel_prompt = PromptTemplate(
    input_variables=["destination"],
    template="Please book a 4-star hotel in {destination}."
)

hotel_chain = LLMChain(
    prompt=hotel_prompt,
    memory=memory,
    llm=llm
)

IT_TEMPLATE = """
Your role is travel assistant. you are tasked to Create detailed a 3-day travel itinerary for 
{destination}. Please DO NOT LIST OUT ANY WRONG DESTINATIONS IF YOU DONT KNOW SAY SORRY I CAN'T FIND ONE
Format your response based on number of days of travel : 
    Day 1: Explore {destination}'s landmarks. 
    Day 2: List famous Local cuisine tasting. 
    Day 3: List Relax and leisure activities
"""
itinerary_prompt = PromptTemplate(
    input_variables=["destination"],
    template=IT_TEMPLATE
)

itinerary_chain = LLMChain(
    prompt=itinerary_prompt,
    memory=memory,
    llm=llm
)


def recommend_destinations(interests: str, destination_chain: LLMChain) -> str:
    """
    Recommends travel destinations based on the user's interests.

    This function takes a string of user interests and returns a string
    containing recommended destinations that match those interests.

    Parameters:
    interests (str): A string describing the user's travel interests.
                     Multiple interests can be included, separated by commas.

    Returns:
    str: A string containing recommended destinations based on the given interests.
         The recommendations include the destination names and brief descriptions
         of why they match the user's interests.

    Example:
    >>> recommend_destinations("culture, beaches, nature")
    "Based on your interests in culture, beaches, nature, consider visiting Kyoto (for culture), Bali (for beaches), or Iceland (for nature)."
    """
    destination_response = destination_chain.run(interests=interests)
    return destination_response


destination_tool = Tool(
    name="DestinationRecommender",
    func=lambda interests: recommend_destinations(interests, destination_chain=destination_chain),
    description="Recommends destinations based on user interests.",
)


def book_hotel(destination: str, hotel_chain: LLMChain) -> str:
    """
    Book a hotel in a specified destination.

    This function takes a string representing the destination and returns a string
    containing a confirmation message for the hotel booking.
    Parameters:
        destination (str): The name of the destination for which the hotel is being booked.
        Returns:
            str: A confirmation message for the hotel booking in the given destination.
            Example:
                >>> book_hotel("Bali")
                "Hotel booked in Bali at a 4-star property with great reviews."
    """
    hotel_response = hotel_chain.run(destination=destination)
    return hotel_response


hotel_tool = Tool(
    name="HotelBooking",
    func=lambda destination: book_hotel(destination, hotel_chain=hotel_chain),
    description="Books hotels in a specified destination."
)


def generate_itinerary(destination: str, itinerary_chain: LLMChain) -> str:
    """
    Generates a 3-day itinerary for a given destination.

    Parameters:
        destination (str): The name of the destination for which the itinerary is being generated.
        Returns:
            str: A 3-day itinerary for the given destination.
            Example:
                >>> generate_itinerary("Bali")
                "Day 1: Explore Bali's landmarks. Day 2: Local cuisine tasting. Day 3: Relax and leisure activities."
    """
    it_response = itinerary_chain.run(destination=destination)
    return it_response


itinerary_tool = Tool(
    name="ItineraryPlanner",
    func=lambda destination: generate_itinerary(destination, itinerary_chain=itinerary_chain),
    description="Provides a 3-day itinerary for a given destination."
)

tools = [destination_tool, hotel_tool, itinerary_tool]


# Conditional logic for tool execution
def decide_branch(user_input: str) -> str:
    if "recommend" in user_input:
        return "destination"
    elif "book" in user_input:
        return "hotel"
    elif "itinerary" in user_input:
        return "itinerary"
    return "default"


execution_logic = {
    "default": {
        "tools": [],
        "description": "Default branch for unrecognized inputs."
    },
    "destination": {
        "tools": [destination_tool],
        "description": "Branch for 'recommend' inputs."
    },
    "hotel": {
        "tools": [hotel_tool],
        "description": "Branch for 'book' inputs."
    },
    "itinerary": {
        "tools": [itinerary_tool],
        "description": "Branch for 'itinerary' inputs."
    }
}

agent = initialize_agent(
    tools=[destination_tool, hotel_tool, itinerary_tool],
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    llm=llm,
    memory=memory,
    verbose=True,
    execution_control_logic=execution_logic,
)

print("Welcome to the AI Travel Assistant!")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = agent.run(user_input)
    print(f"Assistant: {response}")
