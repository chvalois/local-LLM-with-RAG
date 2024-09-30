import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import pdfplumber
# from unstructured.documents.elements import CompositeElement, Table
#UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY")

load_dotenv()

documents = []

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)


def element_to_document(element):
    # if isinstance(element, CompositeElement):
    #     content = element.text  # Assuming CompositeElement has a text attribute
    # elif isinstance(element, Table):
    #     content = str(element.to_dict())  # Assuming Table has a to_list method to convert to a list
    # else:
    #     content = str(element)  # Fallback to string representation for unknown types

    content = str(element)
    return Document(page_content=content)


# Extracts the elements from the PDF

def extract_names(text):
    # Regex pattern to match "Mr [First Name] [Surname]" or "Mme [First Name] [Surname]"
    name_pattern = r'\b(M.|Mme)\s+([A-Z][a-zA-Zéà]*)\s+([A-Z][a-zA-Zéà]*)\b'
    
    # Find all matches in the text
    matches = re.findall(name_pattern, text)
    
    # Extract full names
    names = ", ".join([" ".join(match) for match in matches])
    
    return names

def extract_columns_from_page(page):
    # Define the left and right column bounding boxes
    # (left, top, right, bottom) coordinates for each column
    left_bbox = (0, 0, page.width / 2, page.height)
    right_bbox = (page.width / 2, 0, page.width, page.height)

    # Extract text from each column
    left_text = page.within_bbox(left_bbox).extract_text()
    right_text = page.within_bbox(right_bbox).extract_text()

    # Combine the text from both columns, reading left column first
    combined_text = (left_text or "") + "\n" + (right_text or "")
    
    return combined_text

def extract_elements_from_pdf(filepaths):
    """ extract elements from pdf with partition_pdf"""
    # for filepath in filepaths:
    #     pdf_elements = partition_pdf(
    #         filename=filepath,
    #         strategy="auto",
    #         #strategy="hi_res", 
    #         #model="yolox",
    #         #infer_table_structure=True,
    #         extract_images_in_pdf=True,                            # mandatory to set as ``True``
    #     )
    # documents.append(pdf_elements)

    pdf_files = [os.path.join("sources", filepaths, file) for file in os.listdir(os.path.join("sources", filepaths)) if file.endswith('.pdf')]
    print(pdf_files)

    for pdf_path in pdf_files:

        pdf_type = pdf_path.split("_")[-2]
        pdf_lang = pdf_path.split("_")[-1].split('.')[0]
        print(pdf_path)
        print("Type de document PDF : ", pdf_type)
        print("Langue du document PDF : ", pdf_lang)

        # Extract text using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                extracted_text = ""
                page_text = ""

                if pdf_type == "pdf2pp":
                    # Si 2 colonnes
                    page_text = extract_columns_from_page(page)
                elif pdf_type == "pdf1pp":
                    # Si texte normal
                    page_text = page.extract_text(x_tolerance=1)

                extracted_text += page_text + "\n"
                
                # Split text using LangChain's RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(extracted_text)
                
                # Wrap each chunk in a Document object with metadata
                for chunk in chunks:
                    # Extract names from the text
                    names = extract_names(chunk)
                    doc = Document(
                        page_content=chunk,
                        metadata={"page": i + 1, "source": pdf_path, "names": names}
                    )
                    documents.append(doc)

                    print(doc)
                    print('\n\n')


    return documents



def process_pdf_documents(filepaths):
    """Process the PDF documents and extract elements."""
    print("Processing PDF documents...")
    print(filepaths)
    # List all PDF files
    pdf_files = [os.path.join(filepaths, file) for file in os.listdir(filepaths) if file.endswith('.pdf')]
    
    extract_elements_from_pdf(pdf_files)

    flattened_elements_list = [element for sublist in documents for element in sublist]
    docs = [element_to_document(element) for element in flattened_elements_list]
    _ = [print(doc.page_content) for doc in docs]
    return docs



