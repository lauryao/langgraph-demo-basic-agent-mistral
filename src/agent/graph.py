#from dotenv import load_dotenv
#load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState, add_messages
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from dataclasses import dataclass, field
from langchain_core.prompts import PromptTemplate
from typing import cast, Annotated, Dict, List

@dataclass
class TripPreferenceSchema: trip: dict[str, bool | None] = field(default_factory=lambda: { 
# By default we like beach and sport trips 
       "plage": True, 
       "montagne": None, 
       "ville": False, 
       "sport": True, 
       "detente": False, 
       "acces_handicap": None 
      })

# --- 2. Trip data → Document store & vector store setup ---

TRIP_LIST = [
    {"nom": "Randonnee camping en Lozere", "labels": ["sport","montagne","campagne"], "acces_handicap": "non"},
    {"nom": "5 etoiles a Chamonix option cfondue", "labels": ["montagne","detente"], "acces_handicap": "oui"},
    {"nom": "5 etoiles a Chamonix option ski", "labels": ["montagne","sport"], "acces_handicap": "non"},
    {"nom": "Palavas de paillotes en paillotes", "labels": ["plage","ville","detente","paillote"], "acces_handicap": "oui"},
    {"nom": "5 etoiles en rase campagne", "labels": ["campagne","detente"], "acces_handicap": "oui"},
]

#Define the State class
class TravelState(MessagesState):
    pass
    
    

# Setup LLM & structured extractor
llm = init_chat_model("codestral-latest", model_provider="mistralai", )
structured_extractor = llm.with_structured_output(TripPreferenceSchema)

async def ExtractPreferences(state: TravelState) -> Dict[str, List[AIMessage]]:
    user_text = state["messages"][-1].content  
    prompt_trip_criteria="""
    Comme agent de voyage donnez svp les criteres preferes saisis par l\'utilisateur
    Message: {user_text}
    Les criteres possibles de l\'agence de voyage sont: {possible_trips}
    """

    # we define a prompt template
    template = PromptTemplate(input_variables=["user_text","possible_trips"],
                              template=prompt_trip_criteria
                              )

    #liste des preferences
    trip_preferences = list(TripPreferenceSchema().trip.keys())
    
    #llm structured extractor call here 
    res= await structured_extractor.ainvoke(template.invoke({"user_text": user_text, "possible_trips": trip_preferences}))

    #We return the list of criteria defined by the llm according to what the user has entered
    return {
            "messages": [
                AIMessage(
                    content=[res],
                )
            ]
        }


    

def RecommendTrips(state: TravelState) -> Dict[str, List[AIMessage]]:
    #we extract the last message (list of criteria)
    pref_extracted = state['messages'][-1].content
    print("pref_extracted", pref_extracted)

    #Initialize to False
    criteria = False

    #Search if at least one criteria has been met
    for key,value in pref_extracted[0]["trip"].items():
        if value == True:
            criteria = True

    # If at least one criteria has been met
    if criteria:
        prompt_recom_trip="""
        Selon les critères saisis par l utilisateur
        his criteria for a trip are : {prefered_criteria}

        can you give us the ideal trips in this list of trips according to the criterias
        {list_of_trips} and ask the customer to define other criteria to?
        """

        template = PromptTemplate(input_variables=["prefered_criteria","list_of_trips"],
                              template=prompt_recom_trip
                              )
        res= llm.invoke(template.invoke({"prefered_criteria": pref_extracted[0]["trip"], "list_of_trips": TRIP_LIST}))

        return  {"messages": [res]}
    else: #If we return a message

        return {
            "messages": [
                AIMessage(                    
                    content="Sorry, I could not find any criteria for you. Can you anter your criteria for a trip please?",
                )
            ]
        }


#We define the graph builder
builder = StateGraph(state_schema=TravelState)

#add the extract preferences node
builder.add_node("ExtractPreferences", ExtractPreferences)
#add the recommend trip node
builder.add_node("RecommendTrips", RecommendTrips)
#add the edge from START to extract preferences
builder.add_edge(START, "ExtractPreferences")
#add the edge from extract preferences to recommend trips
builder.add_edge("ExtractPreferences", "RecommendTrips")
#add the edge from recommend trips to END
builder.add_edge("RecommendTrips",END)

#we compile the buiilder to create the graph
graph = builder.compile(name="My Travel Agency")





