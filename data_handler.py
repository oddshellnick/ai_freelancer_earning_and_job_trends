import pandas
from collections import abc
from typing import (
	Any,
	Literal,
	Sequence,
	Union
)


_single_literal_data_type = Literal[
	"Job_Category",
	"Platform",
	"Experience_Level",
	"Client_Region",
	"Payment_Method"
]
_literal_data_type = Union[_single_literal_data_type, Sequence[_single_literal_data_type]]

_single_value_data_type = Literal["Job_Completed", "Earnings_USD", "Hourly_Rate", "Job_Success_Rate"]
_value_data_type = Union[_single_value_data_type, Sequence[_single_value_data_type]]

_single_comparison_operator_type = Literal["gt", "ge", "lt", "le", "eq", "ne"]
_comparison_operator_type = Union[
	_single_comparison_operator_type,
	Sequence[_single_comparison_operator_type]
]

_single_value_to_compare_type = Union[int, float]
_value_to_compare_type = Union[_single_value_to_compare_type, Sequence[_single_value_to_compare_type]]


def _compare_value(
		data: pandas.DataFrame,
		value_data_type: _single_value_data_type,
		comparison_operator: _single_comparison_operator_type,
		value_to_compare: _single_value_to_compare_type
) -> pandas.DataFrame:
	"""
	Filters a pandas DataFrame based on a single comparison condition applied to a specific column.

	Args:
		data (pandas.DataFrame): The input DataFrame to filter.
		value_data_type (_single_value_data_type): The name of the column to apply the comparison to.
		comparison_operator (_single_comparison_operator_type): The comparison operator as a string literal.
		value_to_compare (_single_value_to_compare_type): The value to compare against.

	Returns:
		pandas.DataFrame: A new DataFrame containing only the rows that satisfy the comparison condition.

	Raises:
		ValueError: If an invalid comparison operator is provided.
	"""
	if comparison_operator == "gt":
		return data[data[value_data_type] > value_to_compare]
	elif comparison_operator == "ge":
		return data[data[value_data_type] >= value_to_compare]
	elif comparison_operator == "lt":
		return data[data[value_data_type] < value_to_compare]
	elif comparison_operator == "le":
		return data[data[value_data_type] <= value_to_compare]
	elif comparison_operator == "eq":
		return data[data[value_data_type] == value_to_compare]
	elif comparison_operator == "ne":
		return data[data[value_data_type] != value_to_compare]
	else:
		raise ValueError("Invalid comparison operator.")


def _validate_types_len(*types: Sequence[Any]) -> bool:
	"""
	Checks if all provided sequences have the same length.

	Args:
		*types (Sequence[Any]): A variable number of sequences to compare the length of.

	Returns:
		bool: True if all sequences have the same length or if no sequences are provided, False otherwise.
	"""

	first_len = len(types[0])
	
	return all(len(type_) == first_len for type_ in types)


def _validate_all_types_are_sequence(*types: Sequence[Any]) -> bool:
	"""
	Checks if all provided arguments are instances of Sequence.

	Args:
		*types (Any): A variable number of arguments to check.

	Returns:
		bool: True if all arguments are sequences, False otherwise.
	"""

	return all(isinstance(type_, abc.Sequence) for type_ in types)


def _validate_comparison_types(*types: Sequence[Any]) -> bool:
	"""
	Checks if the provided arguments are consistent in type, meaning either all are Sequences
	or none are Sequences.

	This is used to ensure that multiple comparison conditions (data types, operators, values)
	are provided either all as single values or all as sequences of the same length.

	Args:
		*types (Any): A variable number of arguments to check.

	Returns:
		bool: True if all arguments are sequences or if none are sequences, False otherwise.
	"""

	return all(isinstance(type_, abc.Sequence) for type_ in types) or not any(isinstance(type_, abc.Sequence) for type_ in types)


def _compare_values(
		data: pandas.DataFrame,
		value_data_type: _value_data_type,
		comparison_operator: _comparison_operator_type,
		values_to_compare: _value_to_compare_type
) -> pandas.DataFrame:
	"""
	Filters a pandas DataFrame based on one or multiple comparison conditions applied to corresponding columns.
	Handles both single conditions and lists of conditions.

	Args:
		data (pandas.DataFrame): The input DataFrame to filter.
		value_data_type (_value_data_type): The name of the column (or a sequence of column names)
											to apply the comparison(s) to.
		comparison_operator (_comparison_operator_type): The comparison operator (or a sequence of operators)
													   as string literal(s).
		values_to_compare (_value_to_compare_type): The value (or a sequence of values) to compare against.

	Returns:
		pandas.DataFrame: A new DataFrame containing only the rows that satisfy all specified comparison conditions.

	Raises:
		ValueError: If the arguments are not consistently single values or sequences, or if sequences
					have mismatched lengths, or if an invalid comparison operator is provided.
	"""

	if not _validate_comparison_types(value_data_type, comparison_operator, values_to_compare):
		raise ValueError("Both arguments must be either a sequence or a single value.")
	
	if _validate_all_types_are_sequence(value_data_type, comparison_operator, values_to_compare):
		if not _validate_types_len(value_data_type, comparison_operator, values_to_compare):
			raise ValueError(
					"The length of the comparison operator sequence must match the length of the value data type sequence."
			)
	
		for value_type, operator_type, value_to_compare in zip(value_data_type, comparison_operator, values_to_compare):
			data = _compare_value(data, value_type, operator_type, value_to_compare)
	else:
		data = _compare_value(data, value_data_type, comparison_operator, values_to_compare)
	
	return data


class DataHandler:
	"""
	A class to handle operations and analysis on the freelancer earnings dataset.

	Loads data from "freelancer_earnings_bd.csv" and provides methods for aggregation,
	filtering, and calculating statistics and percentages.

	Attributes:
		data (pandas.DataFrame): The loaded pandas DataFrame containing the dataset.
	"""

	def __init__(self):
		"""
		Initializes the DataHandler by loading the dataset from "freelancer_earnings_bd.csv".
		"""

		self.data = pandas.read_csv("freelancer_earnings_bd.csv")
	
	def get_all_data_types(self, data_to_group: _literal_data_type):
		"""
		Retrieve all unique values in a specified column.

		Args:
			data_to_group (_literal_data_group_type):
				The column to extract unique values from.

		Returns:
			numpy.ndarray: An array of unique values in the specified column.
		"""
		
		return self.data[data_to_group].unique()
	
	def get_average_value(self, data_to_search: _single_literal_data_type):
		"""
		Calculate the average value of a specified column across all records.

		Args:
			data_to_search (str): The column to calculate the average for.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.

		Returns:
			float: The mean value of the specified column.
		"""
		
		return self.data[data_to_search].mean()
	
	def get_average_values_by_data(
			self,
			data_to_group: _literal_data_type,
			data_to_search: _single_value_data_type
	):
		"""
		Calculate the average value of a specified column grouped by one or more columns.

		Args:
			data_to_group (Union[str, List[str]]): The column(s) to group the data by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.
			data_to_search (str): The column to calculate the average for.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.

		Returns:
			pandas.Series: A Series with group labels (or MultiIndex for multiple groups) as the index and the mean values of the specified column.
		"""
		
		return self.data.groupby(data_to_group)[data_to_search].mean()
	
	def get_data_count(self, data_to_group: _literal_data_type):
		"""
		Count the number of records for each group in one or more specified columns.

		Args:
			data_to_group (Union[str, List[str]]): The column(s) to group the data by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.

		Returns:
			pandas.Series: A Series with group labels (or MultiIndex for multiple groups) as the index and the count of records in each group.
		"""
		
		return self.data.groupby(data_to_group).size()
	
	def get_data_count_percentage(self, data_to_group: _literal_data_type):
		"""
		Calculate the percentage of records for each group in one or more specified columns.

		Args:
			data_to_group (Union[str, List[str]]): The column(s) to group the data by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.

		Returns:
			pandas.Series: A Series with group labels (or MultiIndex for multiple groups) as the index and percentage values as strings (e.g., '25.00%').
		"""
		
		all_data_count = len(self.data)
		
		return self.get_data_count(data_to_group).apply(lambda x: f"{(x / all_data_count) * 100:.2f}%")
	
	def get_data_count_with_value_percentage(
			self,
			data_to_group: _literal_data_type,
			data_to_search: _value_data_type,
			value: _value_to_compare_type,
			method: _comparison_operator_type
	):
		"""
		Calculate the percentage of records grouped by one or more columns where a specified column meets a condition.

		Args:
			data_to_group (Union[str, List[str]]): The column(s) to group the data by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.
			data_to_search (str): The column to apply the comparison condition to.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.
			value (Union[int, float]): The value to compare against.
			method (str): The comparison method:
				- 'gt': greater than
				- 'ge': greater than or equal to
				- 'lt': less than
				- 'le': less than or equal to
				- 'eq': equal to
				- 'ne': not equal to

		Returns:
			pandas.Series: A Series with group labels (or MultiIndex for multiple groups) as the index and percentage values as strings (e.g., '25.00%').
		"""
		
		all_data_count = len(self.data)
		
		return self.get_data_count_with_value(data_to_group, data_to_search, value, method).apply(lambda x: f"{(x / all_data_count) * 100:.2f}%")
	
	def get_max_values_by_data(
			self,
			data_to_group: _literal_data_type,
			data_to_search: _single_value_data_type
	):
		"""
		Calculate the max value of a specified column grouped by one or more columns.

		Args:
			data_to_group (Union[str, List[str]]): The column(s) to group the data by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.
			data_to_search (str): The column to calculate the max for.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.

		Returns:
			pandas.Series: A Series with group labels (or MultiIndex for multiple groups) as the index and the max values of the specified column.
		"""
		
		return self.data.groupby(data_to_group)[data_to_search].max()
	
	def get_min_values_by_data(
			self,
			data_to_group: _literal_data_type,
			data_to_search: _single_value_data_type
	):
		"""
		Calculate the min value of a specified column grouped by one or more columns.

		Args:
			data_to_group (Union[str, List[str]]): The column(s) to group the data by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.
			data_to_search (str): The column to calculate the min for.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.

		Returns:
			pandas.Series: A Series with group labels (or MultiIndex for multiple groups) as the index and the min values of the specified column.
		"""
		
		return self.data.groupby(data_to_group)[data_to_search].min()
	
	def get_sum_value(self, data_to_search: _single_value_data_type):
		"""
		Calculate the total sum of values in a specified column across all records.

		Args:
			data_to_search (str): The column to calculate the sum for.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.

		Returns:
			float: The total sum of values in the specified column.
		"""
		
		return self.data[data_to_search].sum()
	
	def get_sum_values_by_data(
			self,
			data_to_group: _literal_data_type,
			data_to_search: _single_value_data_type
	):
		"""
		Calculate the sum of values in a specified column grouped by one or more columns.

		Args:
			data_to_group (Union[str, List[str]]): The column(s) to group the data by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.
			data_to_search (str): The column to calculate the sum for.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.

		Returns:
			pandas.Series: A Series with group labels (or MultiIndex for multiple groups) as the index and the sum of values in the specified column.
		"""
		
		return self.data.groupby(data_to_group)[data_to_search].sum()
	
	def get_value_above_or_below(
			self,
			data_to_search: _single_literal_data_type,
			value: _single_value_to_compare_type
	):
		"""
		Calculate the percentage of records above and below a specified value for a given column.

		Args:
			data_to_search (str): The column to compare against the value.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.
			value (Union[int, float]): The value to compare against.

		Returns:
			str: A string summarizing the percentage of records above and below the value
				(e.g., 'Earnings_USD: above 1000 - 40.00%, below 1000 - 60.00%').
		"""
		
		all_data_count = len(self.data)
		data_above_value = len(self.data[self.data[data_to_search] > value])
		data_below_value = len(self.data[self.data[data_to_search] < value])
		
		return f"{data_to_search}: above {value} - {data_above_value} ({data_above_value / all_data_count * 100:.2f}%), below {value} - {data_below_value} ({data_below_value / all_data_count * 100:.2f}%)"
	
	def get_values_percentage(self, data_to_search: _single_literal_data_type):
		"""
		Calculate the percentage of records in equal-width bins for a specified column.

		Args:
			data_to_search (str): The column to bin and calculate percentages for.
				Must be one of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.

		Returns:
			pandas.DataFrame: A DataFrame with two columns:
				- 'Value': String describing the bin range (e.g., '0.00 to 100.00').
				- 'Percentage': The percentage of records in each bin.
		"""
		
		all_data_count = len(self.data)
		steps_count = min(10, all_data_count) if all_data_count > 0 else 1
		
		bins = pandas.cut(
				self.data[data_to_search],
				bins=steps_count,
				include_lowest=True,
				right=False
		)
		percentages = bins.value_counts(normalize=True) * 100
		
		return pandas.DataFrame(
				{
					"Value": percentages.index.map(lambda x: f"{x.left:.2f} to {x.right:.2f}"),
					"Percentage": percentages.values
				}
		).sort_values(by="Value")
	
	def get_data_count_with_value(
			self,
			data_to_group: _literal_data_type,
			data_to_search: _value_data_type,
			values_to_compare: _value_to_compare_type,
			comparison_operator: _comparison_operator_type
	):
		"""
		Filters rows where each specified column satisfies its corresponding comparison condition.
		If grouping columns are provided, counts the number of matching records per group combination.

		Args:
			data_to_group (Optional[Union[str, List[str]]]): The column(s) to group the results by.
				Must be one or more of: 'Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method'.
				If None, returns the total count of matching rows. Defaults to None.
			data_to_search (Union[str, List[str]]): The column(s) to apply the comparison conditions to.
				Must be one or more of: 'Job_Completed', 'Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate'.
			values_to_compare (Union[int, float, List[Union[int, float]]]): The value(s) to compare against.
			comparison_operator (Union[str, List[str]]): The comparison operator(s) to apply:
				- 'gt': greater than
				- 'ge': greater than or equal to
				- 'lt': less than
				- 'le': less than or equal to
				- 'eq': equal to
				- 'ne': not equal to

		Returns:
			pandas.Series: a Series with a MultiIndex of group labels and counts of matching rows.
		"""
		
		return _compare_values(self.data, data_to_search, comparison_operator, values_to_compare).groupby(data_to_group).size()
