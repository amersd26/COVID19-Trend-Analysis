"""
COVID-19 AI Chatbot - Interactive Demo
======================================
This is a standalone demonstration of the professional AI chatbot
integrated into the COVID-19 Analytics Dashboard.

Run this file to interact with the chatbot directly!
"""

import pandas as pd
from datetime import datetime


class CovidChatbot:
    """Professional AI chatbot for COVID-19 data analysis"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        
    def format_number(self, num):
        """Format large numbers with commas"""
        if pd.isna(num):
            return "N/A"
        return f"{int(num):,}"
    
    def analyze_query(self, query):
        """Analyze user query and generate intelligent response"""
        query_lower = query.lower()
        
        # Cases Analysis
        if any(word in query_lower for word in ["cases", "how many", "total", "case"]):
            return self._get_cases_insight(query_lower)
        
        # Deaths Analysis
        elif any(word in query_lower for word in ["deaths", "death", "mortality", "fatality"]):
            return self._get_deaths_insight(query_lower)
        
        # Vaccination Analysis
        elif any(word in query_lower for word in ["vaccin", "immuniz", "vaccine"]):
            return self._get_vaccination_insight(query_lower)
        
        # Country Comparison
        elif any(word in query_lower for word in ["compare", "versus", "vs", "between", "country"]):
            return self._get_comparison_insight(query_lower)
        
        # Trends
        elif any(word in query_lower for word in ["trend", "peak", "wave", "increase", "decrease"]):
            return self._get_trend_insight(query_lower)
        
        # Testing
        elif any(word in query_lower for word in ["test", "testing", "tests"]):
            return self._get_testing_insight(query_lower)
        
        # Statistics
        elif any(word in query_lower for word in ["average", "mean", "highest", "lowest", "max", "min", "stat"]):
            return self._get_statistics_insight(query_lower)
        
        # Help & Tips
        elif any(word in query_lower for word in ["help", "how", "what", "guide", "tips", "suggest"]):
            return self._get_help_response(query_lower)
        
        # Default
        else:
            return self._get_general_insight()
    
    def _get_cases_insight(self, query):
        """Generate case-related insights"""
        total = self.format_number(self.df.total_cases.max())
        daily_avg = self.format_number(self.df.new_cases.mean())
        peak = self.format_number(self.df.new_cases.max())
        countries = self.df.location.nunique()
        
        return f"""
📊 **COVID-19 Cases Overview**

• **Total Global Cases:** {total}
• **Average Daily Cases:** {daily_avg}
• **Peak Daily Cases:** {peak}
• **Countries Affected:** {countries}

The data shows significant variation in case counts over time, with multiple waves 
representing different COVID-19 variants and regional outbreaks.
"""
    
    def _get_deaths_insight(self, query):
        """Generate death-related insights"""
        total_deaths = self.format_number(self.df.total_deaths.max())
        daily_deaths = self.format_number(self.df.new_deaths.mean())
        peak_deaths = self.format_number(self.df.new_deaths.max())
        
        case_death_ratio = (self.df.total_deaths.max() / self.df.total_cases.max() * 100) if self.df.total_cases.max() > 0 else 0
        
        return f"""
💀 **COVID-19 Mortality Analysis**

• **Total Deaths:** {total_deaths}
• **Average Daily Deaths:** {daily_deaths}
• **Peak Daily Deaths:** {peak_deaths}
• **Case Fatality Ratio:** {case_death_ratio:.2f}%

Mortality rates have improved over time due to better treatments, vaccination programs, 
and improved healthcare infrastructure across nations.
"""
    
    def _get_vaccination_insight(self, query):
        """Generate vaccination insights"""
        vaccinated = self.format_number(self.df.people_vaccinated.max())
        fully_vaccinated = self.format_number(self.df.people_fully_vaccinated.max())
        
        return f"""
💉 **Global Vaccination Progress**

• **Partially Vaccinated:** {vaccinated}
• **Fully Vaccinated:** {fully_vaccinated}

Vaccination campaigns have been crucial in reducing severe cases and deaths.
Global immunization efforts continue to evolve with updated vaccines targeting 
new variants.
"""
    
    def _get_trend_insight(self, query):
        """Generate trend analysis"""
        recent_cases = self.df.nlargest(7, 'date')['new_cases'].mean()
        early_cases = self.df.nsmallest(7, 'date')['new_cases'].mean()
        
        trend = "⬇️ DECREASING" if recent_cases < early_cases else "⬆️ INCREASING"
        
        return f"""
📈 **COVID-19 Trend Analysis**

**Current Trend:** {trend}

The pandemic has shown cyclical patterns with peaks and troughs corresponding to:
- New variant emergence
- Seasonal factors
- Policy changes and interventions
- Vaccination rates
- Population immunity levels

Multiple waves have been observed, with each subsequent wave showing different 
characteristics based on variant transmissibility and population immunity.
"""
    
    def _get_comparison_insight(self, query):
        """Generate country comparison insights"""
        top_cases = self.df.groupby('location')['total_cases'].max().nlargest(5)
        
        response = """
🌍 **Country Comparison**

**Top 5 Countries by Total Cases:**
"""
        for country, cases in top_cases.items():
            response += f"\n• {country}: {self.format_number(cases)}"
        
        response += """

Case counts vary significantly based on:
- Population size
- Population density
- Healthcare system capacity
- Policy responses
- Testing availability
"""
        return response
    
    def _get_testing_insight(self, query):
        """Generate testing insights"""
        total_tests = self.format_number(self.df.total_tests.max())
        
        return f"""
🧪 **Testing Analysis**

• **Total Tests Conducted:** {total_tests}

Testing capacity has been vital for:
- Early case detection
- Isolation and quarantine decisions
- Epidemiological tracking
- Variant identification

Increased testing often correlates with higher confirmed cases due to improved detection.
"""
    
    def _get_statistics_insight(self, query):
        """Generate statistical insights"""
        avg_cases = self.format_number(self.df.new_cases.mean())
        avg_deaths = self.format_number(self.df.new_deaths.mean())
        median_cases = self.format_number(self.df.new_cases.median())
        
        return f"""
📊 **Statistical Summary**

**Daily Cases:**
• Average: {avg_cases}
• Median: {median_cases}

**Daily Deaths:**
• Average: {avg_deaths}

These statistics represent global aggregated data showing the scale and 
severity of the pandemic over time.
"""
    
    def _get_help_response(self, query):
        """Provide help and guidance"""
        return """
🤖 **How I Can Help**

I can answer questions about:
• **Cases & Deaths** - Ask about case counts, death tolls, or mortality rates
• **Vaccination** - Inquire about vaccination programs and coverage
• **Countries** - Compare statistics between nations
• **Trends** - Understand pandemic waves and patterns
• **Testing** - Learn about testing volume and capacity
• **Statistics** - Get aggregated data and averages

**Example Questions:**
- "How many total COVID cases are there?"
- "What's the vaccination rate?"
- "Compare cases between countries"
- "Show me the pandemic trends"
- "Show me testing statistics"

Try asking any question about the COVID-19 data!
"""
    
    def _get_general_insight(self):
        """Provide general information"""
        return f"""
ℹ️ **COVID-19 Analytics Assistant**

I'm here to help you explore and understand the COVID-19 data!

**Quick Stats:**
• Total Countries: {self.df.location.nunique()}
• Date Range: {self.df.date.min().date()} to {self.df.date.max().date()}
• Total Records: {self.format_number(len(self.df))}

Feel free to ask me anything about the pandemic data. You can inquire about 
specific metrics, trends, comparisons, or request analysis on particular aspects.
"""


def print_header():
    """Print a nice header"""
    print("\n" + "="*80)
    print("🦠 COVID-19 AI ASSISTANT - INTERACTIVE CHATBOT DEMO 🦠".center(80))
    print("="*80)
    print("\n💡 This is a professional AI chatbot integrated into the dashboard.")
    print("   Type your questions about COVID-19 data and get instant insights!\n")


def print_separator():
    """Print a separator"""
    print("-"*80)


def main():
    """Main chatbot interaction loop"""
    
    # Load data
    print("📂 Loading COVID-19 data...")
    try:
        df = pd.read_csv("covid19_cleaned.csv")
        df["date"] = pd.to_datetime(df["date"])
        print(f"✅ Successfully loaded {len(df)} records!")
    except FileNotFoundError:
        print("❌ Error: covid19_cleaned.csv not found!")
        return
    
    # Initialize chatbot
    chatbot = CovidChatbot(df)
    
    # Print welcome message
    print_header()
    
    # Conversation loop
    while True:
        print_separator()
        
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n🤖 Chatbot: Goodbye! Thanks for using COVID-19 AI Assistant. Stay safe! 👋")
                break
            
            # Check for help command
            if user_input.lower() == 'help':
                print("\n📋 **AVAILABLE COMMANDS:**")
                print("   • 'exit' or 'quit' - Close the chatbot")
                print("   • 'clear' - Clear the screen")
                print("   • 'help' - Show this message")
                print("\n   Or just ask any question about COVID-19 data!")
                continue
            
            # Check for clear command
            if user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Generate response
            response = chatbot.analyze_query(user_input)
            print(f"\n🤖 Chatbot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n🤖 Chatbot: Goodbye! Thanks for using COVID-19 AI Assistant. Stay safe! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("   Please try again with a different question.")


if __name__ == "__main__":
    main()
