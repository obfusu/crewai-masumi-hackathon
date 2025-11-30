from crewai import Agent, Crew, Task, LLM
from logging_config import get_logger

class ResearchCrew:
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        self.crew = self.create_crew()
        self.logger.info("ResearchCrew initialized")

    def create_crew(self):
        self.logger.info("Creating agent")
        llm = LLM(
            model="ollama/gemma3:1b",
            base_url="http://localhost:11434"
        )
        
        agent = Agent(
            role='Invoice Analyst',
            goal='Find and analyze key information from invoice',
            backstory='Expert at extracting information',
            verbose=self.verbose,
            llm=llm
        )

        crew = Crew(
            agents=[agent],
            tasks=[
                Task(
                    description='{text}',
                    expected_output='Detailed findings about the invoice',
                    agent=agent
                ),
            ]
        )
        self.logger.info("Crew setup completed")
        return crew