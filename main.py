from ai_handler import AI_Handler


agent = AI_Handler(model_name="qwen3:8b", messages_cache_size=20)

while True:
	input_ = input("Type your query: ")

	if input_ == "":
		break

	print(agent.invoke_agent(input_))
