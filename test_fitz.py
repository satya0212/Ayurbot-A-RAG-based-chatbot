import os
import fitz
doc = fitz.open(r"c:\Users\user\OneDrive\Desktop\ayurbot\data\Scientific_Basis_for_Ayurvedic_Therapies (1).pdf")
print("pages:", len(doc))
total = sum(len(page.get_text()) for page in doc)
print("total chars:", total)
