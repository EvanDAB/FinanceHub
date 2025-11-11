import PyPDF2
from typing import Optional

def read_pdf(file_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> str:
    """
    Read a PDF file and extract its text content.
    
    Args:
        file_path (str): Path to the PDF file
        start_page (int, optional): First page to read (0-based index). If None, starts from first page
        end_page (int, optional): Last page to read (0-based index). If None, reads until last page
    
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Open the PDF file
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get total number of pages
            total_pages = len(pdf_reader.pages)
            
            # Validate and adjust page range
            start = start_page if start_page is not None else 0
            end = min(end_page if end_page is not None else total_pages, total_pages)
            
            # Extract text from specified pages
            text_content = []
            for page_num in range(start, end):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract text from page
                text_content.append(page.extract_text())
            
            # Join all text with newlines
            return '\n'.join(text_content)
            
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return ""
    except Exception as e:
        print(f"An error occurred while reading the PDF: {str(e)}")
        return ""

# Example usage
if __name__ == "__main__":
    # Replace with your PDF file path
    pdf_path = "path/to/your/file.pdf"
    
    # Read entire PDF
    full_text = read_pdf(pdf_path)
    print("Full PDF content:")
    print(full_text)
    
    # Read specific page range (e.g., pages 0-2)
    partial_text = read_pdf(pdf_path, start_page=0, end_page=2)
    print("\nFirst two pages:")
    print(partial_text)