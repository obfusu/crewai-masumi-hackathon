import pdfplumber
from crewai import Agent, Crew, Task, LLM
from logging_config import get_logger
from crew_definition import ResearchCrew

# 1. PDF Loader
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

class Logic:
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
        invoiceText = load_pdf("invoice.pdf")
        
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
                    description='{invoiceText}',
                    expected_output='Detailed research findings about the topic',
                    agent=researcher
                ),

            ]
        )
        self.logger.info("Crew setup completed")
        return crew