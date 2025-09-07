from vllm import LLM
llm = LLM("/app/models/models/llama-2/7B/", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")