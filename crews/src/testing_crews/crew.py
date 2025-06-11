from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from testing_crews.tools.tavily_search_tool import TavilySearchTool
from testing_crews.tools.human_input_tool import HumanInputTool
from testing_crews.models import TripDetails, AttractionsSearchResult
from typing import Tuple, Union, Dict, Any
from crewai import TaskOutput
import json

@CrewBase
class TestingCrews():
    """TestingCrews crew"""

    @agent
    def detail_extractor(self) -> Agent:
        return Agent(
            role="Detail Extractor",
            goal="Extract trip details from user query and collect missing mandatory information using the Human Input Collector tool",
            backstory="You're a precise detail extractor who extracts information from user queries. When mandatory information is missing (destination, duration, start_date, budget), you use the Human Input Collector tool to gather the missing details from the user. You never make up or assume any data.",
            tools=[HumanInputTool()],
        )

    @agent
    def attractions_searcher(self) -> Agent:
        return Agent(
            role="Attractions Searcher",
            goal="Find attractions that match the trip duration and budget using maximum 2 targeted searches",
            backstory="You are a smart travel researcher who tailors attraction recommendations based on trip length and budget. For short trips, you focus on must-see highlights. For longer trips, you find diverse experiences. You always consider the budget - suggesting free attractions for budget travelers and premium experiences for high-budget trips. You perform efficient, targeted searches and stop once you have the right number of attractions for the trip duration.",
        )

    @agent
    def trip_planner(self) -> Agent:
        return Agent(
            role="Trip Planner",
            goal="Plan a trip to the given location",
            backstory="You are a travel enthusiast who is very good at planning trips to a given location. You are excellent at planning a trip from day to day basis with detailed information about the attractions, restaurants, and activities. You are also very good at providing information about the trip in a clear and concise manner.",
        )

    @task
    def extraction_task(self) -> Task:
        return Task(
            description="""
            Here is the user's query:
            {query}

            Extract trip details from the user's query. If any mandatory fields are missing, use the Human Input Collector tool to gather them.
            
            MANDATORY FIELDS (Required for trip planning):
            - destination: Where the user wants to travel
            - duration: How long the trip will last (can be in any format: "5 days", "2 weeks", "1 month", etc.)
            - start_date: When the trip begins
            - budget: The budget range or amount for the trip
            
            OPTIONAL FIELDS (Extract if mentioned):
            - interests: Activities, attractions, or experiences they're interested in
            - group_size: Number of people traveling
            - accommodation_type: Preferred type of accommodation
            
            PROCESS:
            1. First, extract all available information from the user's query
            2. If any mandatory fields are missing, use the Human Input Collector tool with:
               - missing_fields: comma-separated list of missing mandatory fields
               - current_data: JSON string of currently extracted data
            3. Update your extraction with the information collected from the user
            4. Ensure all mandatory fields are present before completing the task
            
            IMPORTANT: Duration should be extracted as a string exactly as mentioned (e.g., "5 days", "2 weeks", "1 month")
            """,
            expected_output="Complete trip details with all mandatory fields (destination, duration, start_date, budget) filled in",
            agent=self.detail_extractor(),
            output_pydantic=TripDetails,
        )

    @task
    def attractions_search_task(self) -> Task:
        return Task(
            description="""
            Search for attractions in the destination based on the trip duration and budget from the extracted trip details.
            
            DYNAMIC SEARCH REQUIREMENTS (based on trip details):
            
            DURATION-BASED ATTRACTION COUNT:
            - Short trips (1-3 days): Find 3-5 key attractions
            - Medium trips (4-7 days): Find 6-10 attractions  
            - Long trips (8+ days or weeks/months): Find 10-15 attractions
            
            BUDGET-BASED ATTRACTION TYPES:
            - Budget/Low budget: Focus on free attractions, parks, walking tours, local markets
            - Medium budget: Mix of free and paid attractions, museums, local restaurants
            - High budget: Premium attractions, fine dining, exclusive experiences, luxury activities
            
            SEARCH PROCESS:
            1. Analyze the trip duration to determine target number of attractions
            2. Consider the budget to prioritize appropriate attraction types
            3. Do ONE comprehensive search: "top attractions in [destination] for [duration] trip [budget] budget"
            4. If needed, do ONE additional targeted search based on budget level
            5. STOP after maximum 2 searches
            6. Compile results prioritizing attractions that match the duration and budget
            
            SEARCH EXAMPLES:
            - "top attractions in Paris for 5 days medium budget"
            - "budget-friendly attractions in Tokyo for 2 weeks"
            - "luxury experiences in Dubai for 3 days high budget"
            
            DO NOT:
            - Ignore the trip duration when selecting attractions
            - Suggest expensive attractions for budget trips
            - Suggest too few attractions for long trips
            - Search more than 2 times total
            
            STOPPING CRITERIA:
            - Stop when you have appropriate number of attractions for the trip duration
            - Stop after 2 searches regardless of results
            - Stop if search results are repetitive
            """,
            expected_output="List of attractions appropriate for the trip duration and budget, with names, locations, brief descriptions, and estimated costs where relevant.",
            agent=self.attractions_searcher(),
            tools=[TavilySearchTool()],
            output_pydantic=AttractionsSearchResult,
        )

    @task
    def trip_plan_task(self) -> Task:
        return Task(
            description="Create a detailed day-by-day trip plan using the extracted trip details and found attractions. Plan activities for each day with specific timings, considering travel time between locations.",
            expected_output="A detailed trip plan with day-wise itinerary including timings, attractions to visit, and activities for each day",
            agent=self.trip_planner(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TestingCrews crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
