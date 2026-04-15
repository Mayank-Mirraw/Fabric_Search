from PyPDF2 import PdfReader, PdfWriter
import os

def split_pdf(input_path, pages_per_part=30):
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    reader = PdfReader(input_path)
    
    # FIX: Use reader.pages to get the total count
    total_pages = len(reader.pages) 
    
    print(f"Processing {input_path} ({total_pages} pages total)...")
    
    for i in range(0, total_pages, pages_per_part):
        writer = PdfWriter()
        
        for page_num in range(i, min(i + pages_per_part, total_pages)):
            writer.add_page(reader.pages[page_num])
        
        part_num = i // pages_per_part + 1
        
        # Saving in the same directory as the original file
        output_dir = os.path.dirname(input_path)
        output_filename = os.path.join(output_dir, f"Vendor_Catalog_Part_{part_num}.pdf")
        
        with open(output_filename, "wb") as output_pdf:
            writer.write(output_pdf)
        
        print(f"Successfully created: {output_filename}")

if __name__ == "__main__":
    # Ensure this path is correct for your Linux machine
    file_to_split = r"/home/mirraw/Downloads/Satkaar fashion.pdf" 
    split_pdf(file_to_split, pages_per_part=30)
    