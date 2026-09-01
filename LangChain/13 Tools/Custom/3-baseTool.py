from langchain_core.tools import BaseTool


class AddTool(BaseTool):

    name: str = "add_numbers"

    description: str = "Add two numbers"

    def _run(self, a: int, b: int):
        return a + b


add_tool = AddTool()

result = add_tool.invoke({
    "a": 10,
    "b": 20
})

print(result)