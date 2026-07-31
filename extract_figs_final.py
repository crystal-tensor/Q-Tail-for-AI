import fitz

pdf_path = "/Users/avalok/work/Q-TAIL-MVP/Enhanced_Paper_Quantum_RCS_Long_Tail_Embedded_Learning_v3.pdf"
doc = fitz.open(pdf_path)

# Figure 2 on Page 2 (Risk)
page2 = doc[1] 
# Right column: x approx 315 to 560. y from 290 to 540.
rect2 = fitz.Rect(310.0, 290.0, 565.0, 545.0)
pix2 = page2.get_pixmap(dpi=300, clip=rect2)
pix2.save("results/fig_risk_wasserstein.png")
print("Saved results/fig_risk_wasserstein.png")

# Figure 3 on Page 3 (Exploration)
page3 = doc[2]
# Left column: x approx 45 to 305. y from 270 to 445.
rect3 = fitz.Rect(45.0, 270.0, 310.0, 445.0)
pix3 = page3.get_pixmap(dpi=300, clip=rect3)
pix3.save("results/fig_exploration_discovery.png")
print("Saved results/fig_exploration_discovery.png")

