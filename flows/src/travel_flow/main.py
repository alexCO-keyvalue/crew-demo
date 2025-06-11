#!/usr/bin/env python
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Agent, Crew, Task, Process
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from travel_flow.tools.tavily_search_tool import TavilySearchTool
from travel_flow.tools.human_input_tool import HumanInputTool
from travel_flow.models import TripDetails, AttractionsSearchResult


# Define our flow state to maintain data across nodes
class TripPlanningState(BaseModel):
    user_query: str = ""
    trip_details: Optional[TripDetails] = None
    missing_fields: List[str] = []
    attractions_result: Optional[AttractionsSearchResult] = None
    final_trip_plan: str = ""
    needs_missing_details: bool = False


class TripPlanningFlow(Flow[TripPlanningState]):
    """Flow for comprehensive trip planning with detail extraction, validation, and itinerary generation"""

    @start()
    def get_user_input(self):
        """Get the initial trip query from the user"""
        print("\n=== Welcome to AI Trip Planner ===\n")
        
        self.state.user_query = input("Enter your trip query (destination, dates, budget, interests, etc.): ")
        
        print(f"\nProcessing your trip request: {self.state.user_query}\n")

    @listen(get_user_input)
    def extract_trip_details(self):
        """Extract trip details from user query using the detail extractor agent"""
        print("🔍 Extracting trip details from your query...")
        
        # Create detail extractor agent
        detail_extractor = Agent(
            role="Detail Extractor",
            goal="Extract trip details from user query and collect missing mandatory information using the Human Input Collector tool",
            backstory="You're a precise detail extractor who extracts information from user queries. When mandatory information is missing (destination, duration, start_date, budget), you use the Human Input Collector tool to gather the missing details from the user. You never make up or assume any data.",
        )
        
        # Create extraction task
        extraction_task = Task(
            description=f"""
            Here is the user's query:
            {self.state.user_query}

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
            agent=detail_extractor,
            output_pydantic=TripDetails,
        )
        
        # Create and run crew
        extraction_crew = Crew(
            agents=[detail_extractor],
            tasks=[extraction_task],
            process=Process.sequential,
            verbose=True,
        )
        
        result = extraction_crew.kickoff()
        
        # Parse the result to get TripDetails
        if hasattr(result, 'pydantic') and result.pydantic:
            self.state.trip_details = result.pydantic
        else:
            # Fallback parsing if needed
            try:
                trip_data = json.loads(result.raw) if isinstance(result.raw, str) else result.raw
                self.state.trip_details = TripDetails(**trip_data)
            except Exception as e:
                print(f"⚠️ Error parsing trip details: {e}")
                # Create empty TripDetails if parsing fails
                self.state.trip_details = TripDetails()
        
        print(f"✅ Trip details extracted: {self.state.trip_details}")

    @router(extract_trip_details)
    def validate_trip_details(self):
        """Validate if all mandatory trip details are present"""
        print("🔍 Validating trip details...")
        
        mandatory_fields = ["destination", "duration", "start_date", "budget"]
        missing = []
        
        for field in mandatory_fields:
            value = getattr(self.state.trip_details, field, None)
            if not value or value == "":
                missing.append(field)
        
        self.state.missing_fields = missing
        self.state.needs_missing_details = len(missing) > 0
        
        if missing:
            print(f"❌ Missing mandatory details: {', '.join(missing)}")
            return "collect_missing_details_no_loop"
        else:
            print("✅ All mandatory trip details are present!")
            return "search_attractions_no_loop"

    @listen("collect_missing_details_no_loop")
    def collect_missing_details(self):
        """Collect missing mandatory details from the user"""
        print("📝 Collecting missing trip details...")
        
        # Create detail collector agent
        detail_collector = Agent(
            role="Detail Collector",
            goal="Collect missing mandatory trip information from the user",
            backstory="You specialize in gathering missing information from users in a friendly and efficient manner.",
            tools=[HumanInputTool()],
        )
        
        # Create a context with missing fields information
        missing_fields_str = ", ".join(self.state.missing_fields)
        current_data = self.state.trip_details.model_dump_json() if self.state.trip_details else "{}"
        
        collection_task = Task(
            description=f"""
            The user has provided this original query: {self.state.user_query}
            
            Missing mandatory fields: {missing_fields_str}
            Current extracted data: {current_data}
            
            Use the Human Input Collector tool to gather the missing mandatory information from the user.
            Make sure to collect all missing fields: {missing_fields_str}
            
            Return the complete trip details with all mandatory fields filled.
            """,
            expected_output="Complete trip details with all mandatory fields (destination, duration, start_date, budget) filled in",
            agent=detail_collector,
            output_pydantic=TripDetails,
        )
        
        # Create and run crew
        collection_crew = Crew(
            agents=[detail_collector],
            tasks=[collection_task],
            process=Process.sequential,
            verbose=True,
        )
        
        result = collection_crew.kickoff()
        
        # Update trip details with collected information
        if hasattr(result, 'pydantic') and result.pydantic:
            self.state.trip_details = result.pydantic
        
        print("✅ Missing details collected successfully!")

        

    @listen(or_("search_attractions_no_loop", "collect_missing_details"))
    def search_attractions(self):
        """Search for attractions based on the trip details"""
        print("🔍 Searching for attractions...")
        
        if not self.state.trip_details:
            print("❌ No trip details available for attractions search")
            return None
        
        # Create attractions searcher agent
        attractions_searcher = Agent(
            role="Attractions Searcher",
            goal="Find attractions that match the trip duration and budget using maximum 2 targeted searches",
            backstory="You are a smart travel researcher who tailors attraction recommendations based on trip length and budget. For short trips, you focus on must-see highlights. For longer trips, you find diverse experiences. You always consider the budget - suggesting free attractions for budget travelers and premium experiences for high-budget trips. You perform efficient, targeted searches and stop once you have the right number of attractions for the trip duration.",
            tools=[TavilySearchTool()],
        )
        
        # Create attractions search task
        attractions_task = Task(
            description=f"""
            Search for attractions in {self.state.trip_details.destination} based on the trip duration ({self.state.trip_details.duration}) and budget ({self.state.trip_details.budget}).
            
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
            3. Do ONE comprehensive search: "top attractions in {self.state.trip_details.destination} for {self.state.trip_details.duration} trip {self.state.trip_details.budget} budget"
            4. If needed, do ONE additional targeted search based on budget level
            5. STOP after maximum 2 searches
            6. Compile results prioritizing attractions that match the duration and budget
            
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
            agent=attractions_searcher,
            output_pydantic=AttractionsSearchResult,
        )
        
        # Create and run crew
        attractions_crew = Crew(
            agents=[attractions_searcher],
            tasks=[attractions_task],
            process=Process.sequential,
            verbose=True,
        )
        
        result = attractions_crew.kickoff()
        
        # Parse the attractions result
        if hasattr(result, 'pydantic') and result.pydantic:
            self.state.attractions_result = result.pydantic
        else:
            try:
                attractions_data = json.loads(result.raw) if isinstance(result.raw, str) else result.raw
                self.state.attractions_result = AttractionsSearchResult(**attractions_data)
            except Exception as e:
                print(f"⚠️ Could not parse attractions result: {e}")
                # Create a basic result structure
                self.state.attractions_result = AttractionsSearchResult(
                    destination=self.state.trip_details.destination,
                    attractions=[],
                    total_found=0,
                    search_date=datetime.now().strftime('%Y-%m-%d')
                )
        
        print(f"✅ Found {len(self.state.attractions_result.attractions) if self.state.attractions_result else 0} attractions")

    @listen(search_attractions)
    def generate_trip_plan(self):
        """Generate the final trip itinerary"""
        print("📅 Generating your personalized trip plan...")
        
        # Create trip planner agent
        trip_planner = Agent(
            role="Trip Planner",
            goal="Plan a trip to the given location",
            backstory="You are a travel enthusiast who is very good at planning trips to a given location. You are excellent at planning a trip from day to day basis with detailed information about the attractions, restaurants, and activities. You are also very good at providing information about the trip in a clear and concise manner.",
        )
        
        # Prepare attractions info for the planner
        attractions_info = ""
        if self.state.attractions_result and self.state.attractions_result.attractions:
            attractions_info = "\n".join([
                f"- {attraction.name}: {attraction.description} (Location: {attraction.location})"
                for attraction in self.state.attractions_result.attractions
            ])
        else:
            attractions_info = "No specific attractions found, please research popular attractions for the destination."
        
        # Create trip planning task
        planning_task = Task(
            description=f"""
            Create a detailed day-by-day trip plan using the following information:
            
            TRIP DETAILS:
            - Destination: {self.state.trip_details.destination}
            - Duration: {self.state.trip_details.duration}
            - Start Date: {self.state.trip_details.start_date}
            - Budget: {self.state.trip_details.budget}
            - Group Size: {self.state.trip_details.group_size or 'Not specified'}
            - Interests: {', '.join(self.state.trip_details.interests) if self.state.trip_details.interests else 'General sightseeing'}
            
            AVAILABLE ATTRACTIONS:
            {attractions_info}
            
            Create a comprehensive day-by-day itinerary that includes:
            1. Daily schedule with specific timings
            2. Attractions to visit each day
            3. Recommended restaurants for meals
            4. Transportation suggestions between locations
            5. Budget considerations for each day
            6. Tips and recommendations
            
            Make sure the plan is realistic, considering travel time between locations and the specified budget.
            """,
            expected_output="A detailed trip plan with day-wise itinerary including timings, attractions to visit, and activities for each day",
            agent=trip_planner,
        )
        
        # Create and run crew
        planning_crew = Crew(
            agents=[trip_planner],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=True,
        )
        
        result = planning_crew.kickoff()
        
        self.state.final_trip_plan = result.raw if hasattr(result, 'raw') else str(result)
        
        print("✅ Trip plan generated successfully!")


    @listen(generate_trip_plan)
    def save_trip_plan(self):
        """Save the complete trip plan to files"""
        print("💾 Saving your trip plan...")
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Save trip details as JSON
        if self.state.trip_details:
            with open("output/trip_details.json", "w") as f:
                json.dump(self.state.trip_details.model_dump(), f, indent=2)
        
        # Save attractions as JSON
        if self.state.attractions_result:
            with open("output/attractions.json", "w") as f:
                json.dump(self.state.attractions_result.model_dump(), f, indent=2)
        
        # Save final trip plan as markdown
        if self.state.final_trip_plan:
            trip_title = f"Trip to {self.state.trip_details.destination}" if self.state.trip_details else "Your Trip Plan"
            
            plan_content = f"# {trip_title}\n\n"
            plan_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if self.state.trip_details:
                plan_content += "## Trip Details\n\n"
                plan_content += f"- **Destination:** {self.state.trip_details.destination}\n"
                plan_content += f"- **Duration:** {self.state.trip_details.duration}\n"
                plan_content += f"- **Start Date:** {self.state.trip_details.start_date}\n"
                plan_content += f"- **Budget:** {self.state.trip_details.budget}\n"
                if self.state.trip_details.group_size:
                    plan_content += f"- **Group Size:** {self.state.trip_details.group_size}\n"
                if self.state.trip_details.interests:
                    plan_content += f"- **Interests:** {', '.join(self.state.trip_details.interests)}\n"
                plan_content += "\n"
            
            plan_content += "## Your Itinerary\n\n"
            plan_content += self.state.final_trip_plan
            
            with open("output/complete_trip_plan.md", "w") as f:
                f.write(plan_content)
        
        print("\n🎉 Trip planning completed!")
        print("📁 Files saved:")
        print("   - output/trip_details.json")
        print("   - output/attractions.json") 
        print("   - output/complete_trip_plan.md")
        
        return "Trip planning flow completed successfully"


def kickoff():
    """Run the trip planning flow"""
    flow = TripPlanningFlow()
    plot()
    flow.kickoff()
    print("\n=== Flow Complete ===")
    print("Your personalized trip plan is ready!")
    print("Check the output directory for all generated files.")


def plot():
    """Generate a visualization of the flow"""
    flow = TripPlanningFlow()
    flow.plot("trip_planning_flow")
    print("Flow visualization saved to trip_planning_flow.html")


if __name__ == "__main__":
    kickoff()
