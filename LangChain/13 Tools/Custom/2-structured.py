from langchain_core.tools import StructuredTool


def add_numbers(a: int, b: int) -> int:
    return a + b


add_tool = StructuredTool.from_function(
    func=add_numbers,
    name="add_numbers",
    description="Add two numbers"
)

result = add_tool.invoke({
    "a": 10,
    "b": 20
})

print(result)