import langchain
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("D:\Random_pdf.pdf",extract_images=True)
pages = loader.load_and_split()
print(pages[3])
