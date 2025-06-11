from crewai.tools import BaseTool
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, Field

class HumanInputSchema(BaseModel):
    """Input schema for human input tool"""
    missing_fields: str = Field(..., description="Comma-separated list of missing mandatory fields")
    current_data: str = Field(..., description="Current extracted data in JSON format")

class HumanInputTool(BaseTool):
    name: str = "Human Input Collector"
    description: str = "Collects missing mandatory trip information from the user through interactive prompts"
    args_schema: Type[BaseModel] = HumanInputSchema

    def _run(self, missing_fields: str, current_data: str) -> str:
        """
        Collect missing mandatory fields from user input
        
        Args:
            missing_fields: Comma-separated list of missing fields
            current_data: Current extracted data
            
        Returns:
            Updated trip data with user-provided information
        """
        try:
            import json
            
            # Parse current data
            try:
                data = json.loads(current_data) if current_data else {}
            except json.JSONDecodeError:
                data = {}
            
            # Parse missing fields
            fields_list = [field.strip() for field in missing_fields.split(',')]
            
            print("\n" + "="*60)
            print("🚨 MISSING TRIP INFORMATION")
            print("="*60)
            print("Some mandatory information is missing for your trip planning.")
            print("Please provide the following details:\n")
            
            # Define field mappings for cleaner prompts
            field_prompts = {
                'destination': "📍 Destination (where you want to travel)",
                'duration': "📅 Duration (e.g., '5 days', '2 weeks', '1 month')",
                'start_date': "📅 Start Date (when your trip begins)",
                'budget': "💰 Budget (your budget range or amount)"
            }
            
            # Collect all missing fields at once
            missing_inputs = {}
            for field in fields_list:
                # Clean field name to match our mapping
                field_clean = field.lower().replace('(', '').replace(')', '').strip()
                
                # Find the matching field type
                field_key = None
                for key in field_prompts.keys():
                    if key in field_clean or key.replace('_', ' ') in field_clean:
                        field_key = key
                        break
                
                if field_key and (not data.get(field_key) or str(data.get(field_key, '')).strip() == ''):
                    prompt = field_prompts[field_key]
                    user_input = input(f"{prompt}: ").strip()
                    if user_input:
                        missing_inputs[field_key] = user_input
            
            # Update data with collected inputs
            data.update(missing_inputs)
            
            print("\n✅ Thank you! Continuing with trip planning...")
            print("="*60 + "\n")
            
            # Return updated data as JSON string
            return json.dumps(data, indent=2)
            
        except Exception as e:
            error_msg = f"Error collecting user input: {str(e)}"
            print(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})

    async def _arun(self, missing_fields: str, current_data: str) -> str:
        """Async version of the tool"""
        return self._run(missing_fields, current_data) 