# ai_freelancer_earning_and_job_trends: An AI-powered tool for analyzing freelancer earnings and job trends from a dataset.

This project utilizes a local AI model (via Ollama) integrated with Langchain and Langgraph to interact with a dataset of freelancer earnings. It allows users to ask natural language questions about the data and receive insights related to job categories, platforms, experience levels, client regions, payment methods, earnings, hourly rates, and job success rates. The system is designed to understand user queries, leverage a variety of specialized data analysis tools, and provide accurate, context-aware responses.


## Technologies

| Name         | Badge                                                                                                                                                      | Description                                                                                                                                                |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python       | [![Python](https://img.shields.io/badge/Python%2DPython?style=flat&logo=python&color=%231f4361)](https://www.python.org/)                                  | Core programming language for the entire project.                                                                                                          |
| Ollama       | [![Ollama](https://img.shields.io/badge/Ollama%2DOllama?style=flat&logo=ollama&color=%23dc6416)](https://ollama.com/)                                      | Used to run local large language models (e.g., `qwen3:8b` specified in `main.py`) that power the AI agent.                                                 |
| Langchain    | [![LangChain](https://img.shields.io/badge/LangChain%2DLangChain?style=flat&logo=langchain&color=%231c3c3c)](https://www.langchain.com/)                   | Framework used for its `ChatOllama` wrapper (from `langchain_ollama`) to interface with the Ollama-served LLM.                                             |
| LangGraph    | [![LangGraph](https://img.shields.io/badge/LangGraph%2DLangGraph?style=flat&logo=langchain&color=%23053d5b)](https://python.langchain.com/docs/langgraph/) | Utilized via `create_react_agent` to build the AI agent as a stateful graph, enabling ReAct prompting and dynamic tool use based on `DataHandler` methods. |
| Pandas       | [![Pandas](https://img.shields.io/badge/Pandas%2DPandas?style=flat&logo=pandas&color=%23130654)](https://pandas.pydata.org/)                               | Essential library for data manipulation and analysis; used in `data_handler.py` to load, process, and query the `freelancer_earnings_bd.csv` dataset.      |
| NumPy        | [![NumPy](https://img.shields.io/badge/NumPy%2DNumPy?style=flat&logo=numpy&color=%23013243)](https://numpy.org/)                                           | Used for numerical operations, specifically in `data_handler.py` for handling array outputs from pandas operations.                                        |


## Key Features

*   **Natural Language Interaction:** Ask questions about freelancer data in plain English (or other languages, as the agent aims to respond in the user's language).
*   **Local LLM Powered:** Leverages local large language models through Ollama, ensuring data privacy and offline capabilities (once the model is downloaded).
*   **Advanced Agent Framework:** Built with Langchain and Langgraph for robust agent creation, tool usage, and ReAct-based reasoning.
*   **Conversation History:** The AI agent maintains a cache of recent messages (configurable size via `messages_cache_size`) to enable more coherent multi-turn conversations.
*   **Comprehensive Data Analysis:** The `DataHandler` class provides a rich set of tools for:
    *   Retrieving unique values from categorical columns (`Job_Category`, `Platform`, `Experience_Level`, `Client_Region`, `Payment_Method`).
    *   Calculating aggregate statistics (average, sum, min, max) for numerical columns (`Job_Completed`, `Earnings_USD`, `Hourly_Rate`, `Job_Success_Rate`).
    *   Grouping data by various dimensions to calculate statistics for specific segments.
    *   Computing counts and percentages of data segments.
    *   Filtering data based on single or multiple complex comparison conditions (e.g., earnings greater than a certain value).
    *   Analyzing value distributions within numerical data using binning.
*   **Automated Ollama Setup:** The application includes logic to check for Ollama installation and automatically pull the required language model if not already present.
*   **Modular Design:**
    *   `ai_handler.py`: Manages AI agent setup, configuration, and interaction logic.
    *   `data_handler.py`: Encapsulates all data loading, processing, validation, and analysis operations, exposing them as tools for the AI agent.
    *   `main.py`: Simple entry point to run the interactive command-line application.


## Installation

1.  **Prerequisites:**
    *   Ensure you have Python 3.8+ installed.
    *   Install Ollama by following the instructions at [https://ollama.com/](https://ollama.com/). Ensure Ollama is running before starting the application.

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/oddshellnick/ai_freelancer_earning_and_job_trends.git
    ```

3.  **Install the required Python packages using pip:**

    ```bash
    pip install -r requirements.txt
    ```


### Example Queries

*   **Query 1: Get average earnings by job category.**
    ```
    Type your query: What is the average earnings for each job category?
    ```

*   **Query 2: Find platforms with high job success rates.**
    ```
    Type your query: Which platforms have a job success rate greater than 90%?
    ```

*   **Query 3: Check distribution of hourly rates.**
    ```
    Type your query: Show me the percentage distribution of hourly rates.
    ```

*   **Query 4: Multi-turn conversation.**
    ```
    Type your query: What are the unique experience levels?
    Type your query: And what about the average hourly rate for each of those levels?
    ```
    *(The agent will use the previous context to understand "those levels".)*

* To quit the application, simply press Enter at the `Type your query:` prompt without typing any text.


## Classes and Functions

### AI Interaction Logic (`ai_handler.py`)
This module is responsible for setting up the AI model, creating the Langgraph agent, and handling user queries, including conversation history management.

*   `_setup_ollama(...)`:
    *   Checks if the Ollama command-line tool is installed on the system. Exits the script with a warning if not found.
    *   Attempts to pull the specified Ollama language model. Exits with a warning on failure (e.g., network issues, invalid model name).
    *   Returns an initialized `ChatOllama` instance, ready to communicate with the local LLM.
*   **Class `AI_Handler`**: Manages the AI agent's lifecycle and interaction.
    *   `__init__(...)`:
        *   Initializes the `AI_Handler` with the specified `model_name` for the Ollama LLM and `messages_cache_size` to control the length of conversational history.
        *   Instantiates `DataHandler` to access its data analysis tools.
        *   Calls `_build_ollama_graph` to set up and compile the Langgraph agent.
        *   Initializes an empty `messages_cache` to store conversation history.
    *   `_build_ollama_graph(...)`:
        *   Sets up the Ollama model using the `_setup_ollama` helper function.
        *   Constructs a ReAct agent using `langgraph.prebuilt.create_react_agent`.
        *   Provides the agent with a list of callable tools from the `DataHandler` instance (e.g., `get_all_data_types`, `get_average_values_by_data`, etc.).
        *   Configures the agent with a prompt instructing it to provide helpful responses and adapt to the user's language.
    *   `_add_message_to_cache(...)`:
        *   Adds a new message with its associated role (`user`, `ai`, `tool`, etc.) to the internal `messages_cache`.
        *   Implements a simple cache management strategy: if the cache exceeds `messages_cache_size`, the oldest message is removed.
    *   `invoke_agent(...)`:
        *   Records the user's `query` by adding it to the `messages_cache` with the `user` role.
        *   Invokes the compiled Langgraph agent (`self.graph.invoke`) with the current `messages_cache` as input.
        *   Extracts the content of the agent's final response message.
        *   Adds the agent's response to the `messages_cache` with the `ai` role.
        *   Returns the agent's response string.
        *   Includes comprehensive error handling, catching any exceptions during invocation and returning a generic error message along with the detailed traceback for debugging.

### Data Processing and Analysis (`data_handler.py`)
This module handles loading, processing, and providing structured access to the freelancer earnings dataset. It leverages the `pandas` library for efficient data manipulation and analysis.

*   **Internal Helper Functions** (prefixed with `_`): These private functions facilitate core data operations and validations.
    *   `_read_pandas_data(...)`:
        *   Converts pandas DataFrame or Series objects (or NumPy arrays) into their string representation.
        *   Ensures that data extracted from `DataHandler` methods is always returned as a readable string, suitable for AI agent consumption.
    *   `_validate_previous_stack_from_datahandler()`:
        *   Checks if the immediate caller of the decorated function is a method of a `DataHandler` instance.
        *   Used by the `tool_handler` decorator to conditionally process return values.
    *   `tool_handler(...)`:
        *   A decorator that wraps `DataHandler` methods.
        *   If the decorated method is called internally by a `DataHandler` instance, its raw return value (e.g., a pandas object) is returned directly.
        *   If called externally (e.g., by the AI agent), the return value is first processed by `_read_pandas_data` to convert it into a string representation.
    *   `_compare_value(...)`:
        *   Filters a pandas DataFrame based on a single comparison condition applied to a specific column (e.g., `data[data["Earnings_USD"] > 1000]`).
        *   Supports standard comparison operators (`gt`, `ge`, `lt`, `le`, `eq`, `ne`).
    *   `_validate_types_len(...)`:
        *   Helper to ensure all provided sequences (e.g., lists of columns, operators, values) have identical lengths, crucial for multi-condition filtering.
    *   `_validate_all_types_are_sequence(...)`:
        *   Checks if all given arguments are instances of `collections.abc.Sequence`.
    *   `_validate_comparison_types(...)`:
        *   Validates consistency in comparison arguments: either all arguments are sequences (for multiple conditions) or all are single values (for a single condition).
    *   `_compare_values(...)`:
        *   A general-purpose function for filtering a DataFrame based on one or multiple comparison conditions.
        *   Uses `_compare_value` internally and handles validation of input types and lengths for complex queries.
*   **Class `DataHandler`**: Provides methods that serve as tools for the AI agent to query and analyze the dataset.
    *   `__init__(...)`:
        *   Initializes the `DataHandler` by loading the `freelancer_earnings_bd.csv` dataset into a pandas DataFrame, making it ready for analysis.
    *   `get_all_data_types(...)`:
        *   Retrieves all unique values present in one or more specified categorical columns (e.g., unique `Job_Category` values).
    *   `get_average_value(...)`:
        *   Calculates the average (mean) value of a specified numerical column across the entire dataset.
    *   `get_average_values_by_data(...)`:
        *   Calculates the average value of a specified numerical column, grouped by one or more categorical columns.
    *   `get_data_count(...)`:
        *   Counts the number of records for each distinct group within one or more specified categorical columns.
    *   `get_data_count_percentage(...)`:
        *   Calculates the percentage of total records that fall into each group of one or more specified categorical columns. Returns percentages as formatted strings (e.g., '25.00%').
    *   `get_data_count_with_value_percentage(...)`:
        *   Calculates the percentage of records, grouped by specified categorical columns, where a numerical column meets a given comparison condition.
    *   `get_max_values_by_data(...)`:
        *   Calculates the maximum value of a specified numerical column, grouped by one or more categorical columns.
    *   `get_min_values_by_data(...)`:
        *   Calculates the minimum value of a specified numerical column, grouped by one or more categorical columns.
    *   `get_sum_value(...)`:
        *   Calculates the total sum of values in a specified numerical column across the entire dataset.
    *   `get_sum_values_by_data(...)`:
        *   Calculates the sum of values in a specified numerical column, grouped by one or more categorical columns.
    *   `get_value_above_or_below(...)`:
        *   Calculates and returns a string summarizing the percentage of records that are above and below a specified value for a given numerical column.
    *   `get_values_percentage(...)`:
        *   Calculates the percentage distribution of values within equal-width bins for a specified numerical column. Returns a DataFrame with value ranges and their percentages.
    *   `get_data_count_with_value(...)`:
        *   Filters rows where specified numerical columns satisfy corresponding comparison conditions. If grouping columns are provided, it then counts the number of matching records per group combination.

### Main Application (`main.py`)
This script serves as the primary entry point for the application, initializing the AI agent and managing the interactive command-line user session.

*   Initializes an `AI_Handler` instance, configuring it to use the `"qwen3:8b"` Ollama model and maintain a message cache of up to `50` messages.
*   Enters an infinite loop that facilitates continuous user interaction:
    *   Prompts the user to `Type your query:`.
    *   If the user inputs an empty string, the loop breaks, and the application gracefully exits.
    *   For any non-empty input, it invokes the AI agent with the user's query using `agent.invoke_agent(input_)`.
    *   Prints the comprehensive response received from the AI agent to the console, allowing users to see the analysis results or further questions from the agent.


## **Note:**

Please be aware that the `test_conditions` folder contains files related to the test assignment. You are welcome to attempt to solve it independently, if you wish.