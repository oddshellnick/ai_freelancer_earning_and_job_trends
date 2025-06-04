import sys
import warnings
import subprocess
from data_handler import DataHandler
from langchain_ollama import ChatOllama
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent


def setup_ollama(model_name: str) -> ChatOllama:
	"""
	Sets up the Ollama environment by checking installation and pulling the specified model.

	This function first verifies if the Ollama command-line tool is installed.
	If not, it prints a warning and exits the script.
	Then, it attempts to pull the specified language model. If the pull fails,
	it prints a warning and exits.
	Finally, it returns an initialized ChatOllama instance.

	Args:
		model_name (str): The name of the Ollama model to set up (e.g., "llama3.1:8b").

	Returns:
		ChatOllama: An instance of the ChatOllama model ready for use.
	"""
	
	try:
		subprocess.run(
				["ollama", "--version"],
				check=True,
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL
		)
	except subprocess.CalledProcessError:
		warnings.warn(
				"Error: Ollama is not installed. Please install it from https://ollama.com/ and try again."
		)
		sys.exit(1)
	
	try:
		subprocess.run(["ollama", "pull", model_name], check=True)
	except subprocess.CalledProcessError:
		warnings.warn(
				f"Error pulling model '{model_name}'. Check your internet connection and model name."
		)
		sys.exit(1)
	
	return ChatOllama(model=model_name)


class AI_Handler:
	"""
	Manages the AI agent which interacts with a DataHandler to answer data-related queries.

	Initializes a DataHandler to load and process data and builds a Langgraph
	agent using an Ollama model. The agent is equipped with tools from the
	DataHandler to perform data analysis and retrieval tasks based on user queries.

	Attributes:
		model_name (str): The name of the Ollama model used by the agent.
		data_handler (DataHandler): An instance of the DataHandler class, containing the dataset and analysis methods.
		graph (CompiledGraph): The compiled LangGraph agent ready to process queries.
	"""

	def __init__(self, model_name: str):
		"""
		Initializes the AI_Handler, loading the dataset and building the AI graph.

		Args:
			model_name (str): The name of the Ollama model to use (e.g., "llama3.1:8b").
		"""

		self.model_name = model_name
		self.data_handler = DataHandler()
		self.graph = self.build_ollama_graph()
	
	def build_ollama_graph(self) -> CompiledGraph:
		"""
		Builds and compiles a Langgraph agent using the specified Ollama model
		and methods from the DataHandler as tools.

		The agent is configured with a ReAct prompt to utilize the provided
		data analysis tools to answer user questions.

		Returns:
			CompiledGraph: The compiled Langgraph agent.
		"""

		model = setup_ollama(model_name=self.model_name)
		
		return create_react_agent(
				model=model,
				tools=[
					self.data_handler.get_all_data_types,
					self.data_handler.get_average_values_by_data,
					self.data_handler.get_sum_values_by_data,
					self.data_handler.get_average_value,
					self.data_handler.get_sum_value,
					self.data_handler.get_data_count_with_value,
					self.data_handler.get_data_count_with_value_percentage,
					self.data_handler.get_values_percentage,
					self.data_handler.get_value_above_or_below,
					self.data_handler.get_min_values_by_data,
					self.data_handler.get_min_values_by_data,
					self.data_handler.get_data_count,
					self.data_handler.get_data_count_percentage,
				],
				prompt="""
                You are an assistant that provides helpful responses to user queries.
                Provide responses to user in language that user uses.
                """
		)
	
	def invoke_agent(self, query: str) -> str:
		"""
		Invokes the compiled Langgraph agent with a user query.

		Sends the query to the agent and returns the content of the agent's
		final message as a string. If any exception occurs during the
		invocation process, a generic error message is returned.

		Args:
			query (str): The user's query as a string.

		Returns:
			str: The content of the agent's final message or a generic error message
				if an exception occurred.
		"""

		try:
			return self.graph.invoke({"messages": [{"role": "user", "content": query}]})["messages"][-1].content
		except (Exception,):
			return "I'm sorry, I couldn't process your request."
