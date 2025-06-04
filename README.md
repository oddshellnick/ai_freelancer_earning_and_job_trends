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

* You can then ask questions about the dataset. For example:

    ```
    Type your query: What is the average hourly rate for 'Graphic Design' on 'Upwork'?
    ```
    
    The agent will process your query using the dataset and provide a response.

* To quit the application, simply press Enter at the `Type your query:` prompt without typing any text.


## Classes and Functions

### AI Interaction Logic (`ai_handler.py`)
This module is responsible for setting up the AI model, creating the agent, and handling user queries.

*   `setup_ollama(model_name: str) -> ChatOllama`:
    *   Checks if Ollama CLI is installed. Exits if not.
    *   Attempts to pull the specified Ollama model. Exits on failure.
    *   Returns an initialized `ChatOllama` instance.
*   **Class `AI_Handler`**: Manages the AI agent.
    *   `__init__(self, model_name: str)`:
        *   Initializes with the specified `model_name`.
        *   Creates a `DataHandler` instance to access data tools.
        *   Builds the Langgraph agent using `build_ollama_graph`.
    *   `build_ollama_graph(self) -> CompiledGraph`:
        *   Sets up the Ollama model using `setup_ollama`.
        *   Creates a ReAct agent using `langgraph.prebuilt.create_react_agent`, providing it with tools from the `DataHandler` instance.
        *   The agent is prompted to be helpful and respond in the user's language.
    *   `invoke_agent(self, query: str) -> str`:
        *   Takes a user `query`.
        *   Invokes the compiled Langgraph agent with the query.
        *   Returns the agent's final message content as a string. Handles exceptions by returning an error message.


### Data Processing and Analysis (`data_handler.py`)
This module loads and provides tools to analyze the freelancer earnings dataset. It uses `pandas` for data manipulation.

*   **Internal Helper Functions** (prefixed with `_`):
    *   `_compare_value(...)`: Filters a DataFrame based on a single comparison.
    *   `_validate_types_len(...)`: Checks if multiple sequences have the same length.
    *   `_validate_all_types_are_sequences(...)`: Checks if all arguments are sequences.
    *   `_validate_comparison_types(...)`: Validates consistency in argument types for comparisons (all single or all sequence).
    *   `_compare_values(...)`: Filters a DataFrame based on one or more comparison conditions.
*   **Class `DataHandler`**: Provides methods as tools for the AI agent.
    *   `__init__(self)`: Loads data from `"freelancer_earnings_bd.csv"` into a pandas DataFrame.
    *   `get_all_data_types(...)`: Retrieves unique values from a specified column.
    *   `get_average_value(...)`: Calculates the average of a numeric column.
    *   `get_average_values_by_data(...)`: Calculates average values of a numeric column, grouped by one or more categorical columns.
    *   `get_data_count(...)`: Counts records per group for specified columns.
    *   `get_data_count_percentage(...)`: Calculates the percentage of records per group.
    *   `get_data_count_with_value_percentage(...)`: Calculates the percentage of records per group that meet a specific value condition.
    *   `get_max_values_by_data(...)`: Finds maximum values of a numeric column, grouped by categorical columns.
    *   `get_min_values_by_data(...)`: Finds minimum values of a numeric column, grouped by categorical columns.
    *   `get_sum_value(...)`: Calculates the sum of a numeric column.
    *   `get_sum_values_by_data(...)`: Calculates sums of a numeric column, grouped by categorical columns.
    *   `get_value_above_or_below(...)`: Calculates percentages of data above and below a certain value in a column.
    *   `get_values_percentage(...)`: Calculates the percentage distribution of values within automatically determined bins for a numeric column.
    *   `get_data_count_with_value(...)`: Counts records satisfying specific value conditions, optionally grouped.


### Main Application (`main.py`)
This script serves as the entry point for the application.
*   Initializes `AI_Handler` with a specified Ollama model name (e.g., "qwen3:8b").
*   Enters a loop that:
    *   Prompts the user for a query.
    *   If the query is empty, breaks the loop.
    *   Otherwise, invokes the AI agent with the query using `agent.invoke_agent(...)`.
    *   Prints the agent's response.