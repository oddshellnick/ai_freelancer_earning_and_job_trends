# ai_freelancer_earning_and_job_trends: An AI-powered tool for analyzing freelancer earnings and job trends from a dataset.

This project utilizes a local AI model (via Ollama) integrated with Langchain and Langgraph to interact with a dataset of freelancer earnings. It allows users to ask natural language questions about the data and receive insights related to job categories, platforms, experience levels, client regions, payment methods, earnings, hourly rates, and job success rates. The system is designed to understand queries and use a variety of data analysis tools to provide accurate answers.


## Technologies

| Name           | Badge                                                                                                                                                      | Description                                                                                                                                                |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python         | [![Python](https://img.shields.io/badge/Python%2DPython?style=flat&logo=python&color=%231f4361)](https://www.python.org/)                                  | Core programming language for the entire project.                                                                                                          |
| Ollama         | [![Ollama](https://img.shields.io/badge/Ollama%2DOllama?style=flat&logo=ollama&color=%23dc6416)](https://ollama.com/)                                      | Used to run local large language models (e.g., `qwen3:8b` specified in `main.py`) that power the AI agent.                                                 |
| Langchain      | [![LangChain](https://img.shields.io/badge/LangChain%2DLangChain?style=flat&logo=langchain&color=%231c3c3c)](https://www.langchain.com/)                   | Framework used for its `ChatOllama` wrapper (from `langchain_ollama`) to interface with the Ollama-served LLM.                                             |
| LangGraph      | [![LangGraph](https://img.shields.io/badge/LangGraph%2DLangGraph?style=flat&logo=langchain&color=%23053d5b)](https://python.langchain.com/docs/langgraph/) | Utilized via `create_react_agent` to build the AI agent as a stateful graph, enabling ReAct prompting and dynamic tool use based on `DataHandler` methods. |
| Pandas         | [![Pandas](https://img.shields.io/badge/Pandas%2DPandas?style=flat&logo=pandas&color=%23130654)](https://pandas.pydata.org/)                               | Essential library for data manipulation and analysis; used in `data_handler.py` to load, process, and query the `freelancer_earnings_bd.csv` dataset.      |


## Key Features

*   **Natural Language Interaction:** Ask questions about freelancer data in plain English (or other languages, as the agent aims to respond in the user's language).
*   **Local LLM Powered:** Leverages local large language models through Ollama, ensuring data privacy and offline capabilities (once the model is downloaded).
*   **Advanced Agent Framework:** Built with Langchain and Langgraph for robust agent creation, tool usage, and ReAct-based reasoning.
*   **Conversation History:** The AI agent maintains a cache of recent messages (configurable size) to enable more coherent multi-turn conversations.
*   **Comprehensive Data Analysis:** The `DataHandler` class provides a rich set of tools for:
    *   Retrieving unique values from categorical columns.
    *   Calculating aggregate statistics (average, sum, min, max).
    *   Grouping data by various dimensions (Job Category, Platform, Experience Level, etc.).
    *   Computing counts and percentages of data segments.
    *   Filtering data based on single or multiple complex conditions.
*   **Modular Design:**
    *   `ai_handler.py`: Manages AI agent setup and interaction logic.
    *   `data_handler.py`: Encapsulates all data loading, processing, and analysis operations.
    *   `main.py`: Simple entry point to run the interactive agent.
*   **Automated Ollama Setup:** The script checks for Ollama installation and attempts to pull the required model automatically.


## Installation

1.  **Prerequisites:**
    *   Ensure you have Python 3.8+ installed.
    *   Install Ollama by following the instructions at [https://ollama.com/](https://ollama.com/). Ensure Ollama is running.

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/oddshellnick/ai_freelancer_earning_and_job_trends
    ```

3.  **Install the required Python packages using pip:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here are some examples of how to use `ai_freelancer_earning_and_job_trends`:
After running the script, it will initialize the AI Handler (which may take a moment, especially on first run if the model needs downloading). Once ready, you'll be prompted to enter your query:

```
Type your query:
```

*   You can then ask questions about the dataset. For example:

    ```
    Type your query: What is the average hourly rate for 'Graphic Design' on 'Upwork'?
    ```

    The agent will process your query using the dataset and provide a response.

*   To quit the application, simply press Enter at the `Type your query:` prompt without typing any text.


## Classes and Functions

### AI Interaction Logic (`ai_handler.py`)
This module is responsible for setting up the AI model, creating the agent, and handling user queries.

*   `_setup_ollama(model_name: str) -> ChatOllama`:
    *   Checks if Ollama CLI is installed. Exits if not.
    *   Attempts to pull the specified Ollama model. Exits on failure.
    *   Returns an initialized `ChatOllama` instance.
*   **Class `AI_Handler`**: Manages the AI agent.
    *   `__init__(...)`:
        *   Initializes with the specified `model_name` for the Ollama LLM.
        *   Sets the `messages_cache_size` to control conversation history length.
        *   Creates a `DataHandler` instance to access data analysis tools.
        *   Builds the Langgraph agent using `_build_ollama_graph`.
    *   `_build_ollama_graph(...)`:
        *   Sets up the Ollama model using `_setup_ollama`.
        *   Creates a ReAct agent using `langgraph.prebuilt.create_react_agent`, providing it with tools from the `DataHandler` instance.
        *   The agent is prompted to be helpful and respond in the user's language.
    *   `_add_message_to_cache(...)`:
        *   Adds a given message and its role to the internal message cache.
        *   Manages cache size, removing the oldest message if the `messages_cache_size` limit is exceeded.
    *   `invoke_agent(...)`:
        *   Adds the user's `query` to the message cache.
        *   Invokes the compiled Langgraph agent with the current `messages_cache`.
        *   Adds the agent's response to the message cache.
        *   Returns the content of the agent's final message as a string. Handles exceptions during invocation by returning a generic error message with traceback details.


### Data Processing and Analysis (`data_handler.py`)
This module loads and provides tools to analyze the freelancer earnings dataset. It uses `pandas` for data manipulation.

*   **Internal Helper Functions** (prefixed with `_`): These functions are used internally by the `DataHandler` class to perform common data validation and comparison tasks.
    *   `_compare_value(...)`: Filters a pandas DataFrame based on a single comparison condition applied to a specific column.
    *   `_validate_types_len(...)`: Checks if all provided sequences have the same length.
    *   `_validate_all_types_are_sequence(...)`: Checks if all provided arguments are instances of Sequence.
    *   `_validate_comparison_types(...)`: Checks if the provided arguments are consistent in type (all sequences or all single values).
    *   `_compare_values(...)`: Filters a pandas DataFrame based on one or multiple comparison conditions applied to corresponding columns. Handles both single conditions and lists of conditions.
*   **Class `DataHandler`**: Provides methods as tools for the AI agent to interact with the freelancer dataset.
    *   `__init__(...)`: Initializes the `DataHandler` by loading the `freelancer_earnings_bd.csv` dataset into a pandas DataFrame.
    *   `get_all_data_types(...)`: Retrieves all unique values present in a specified categorical column.
    *   `get_average_value(...)`: Calculates the average value of a specified numeric column across the entire dataset.
    *   `get_average_values_by_data(...)`: Calculates the average value of a specified numeric column, grouped by one or more categorical columns.
    *   `get_data_count(...)`: Counts the number of records for each distinct group within one or more specified columns.
    *   `get_data_count_percentage(...)`: Calculates the percentage of total records for each group in one or more specified columns.
    *   `get_data_count_with_value_percentage(...)`: Calculates the percentage of records, grouped by specified columns, where a numeric column meets a given comparison condition.
    *   `get_max_values_by_data(...)`: Calculates the maximum value of a specified numeric column, grouped by one or more categorical columns.
    *   `get_min_values_by_data(...)`: Calculates the minimum value of a specified numeric column, grouped by one or more categorical columns.
    *   `get_sum_value(...)`: Calculates the total sum of values in a specified numeric column across the entire dataset.
    *   `get_sum_values_by_data(...)`: Calculates the sum of values in a specified numeric column, grouped by one or more categorical columns.
    *   `get_value_above_or_below(...)`: Calculates and returns a string indicating the percentage of records that are above and below a specified value for a given numeric column.
    *   `get_values_percentage(...)`: Calculates the percentage distribution of values within equal-width bins for a specified numeric column, returning a DataFrame with value ranges and their percentages.
    *   `get_data_count_with_value(...)`: Filters rows where specified numeric columns satisfy corresponding comparison conditions, and then counts the number of matching records per group combination if grouping columns are provided.


### Main Application (`main.py`)
This script serves as the entry point for the application, initializing the AI agent and handling the interactive user session.

*   Initializes an `AI_Handler` instance, configuring it to use the `"qwen3:8b"` Ollama model and maintain a message cache of up to 20 messages.
*   Enters an infinite loop that:
    *   Prompts the user to `Type your query:`.
    *   If the user enters an empty string, the loop breaks, and the application exits.
    *   Otherwise, it invokes the AI agent with the user's input query using `agent.invoke_agent(...)`.
    *   Prints the response received from the AI agent to the console.