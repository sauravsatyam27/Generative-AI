from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('suly.pdf')

pdf = loader.load()

# print(pdf)
print(len(pdf))
# print(pdf[0].page_content)
print(pdf[0].metadata)



