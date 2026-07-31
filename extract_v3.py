import fitz
doc = fitz.open("/Users/avalok/work/Q-TAIL-MVP/v3.pdf")
text = ""
for i in range(min(4, len(doc))): # Read first 4 pages for methods/abstract
    text += doc[i].get_text()
print(text[:2000])
