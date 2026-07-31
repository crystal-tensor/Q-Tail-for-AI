import re

for filename in ['index.html', 'qtail-mvp-presentation.html']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # The top span might be in the Navbar component, which previously had Q-TAIL-MVP or QTail-For-AI
    # Looking at lines 162-165:
    # <div className="flex items-center gap-2">
    #     <IconAtom className="text-qcyan" />
    # </div>
    # It seems the span might have already been removed from the Navbar in a previous edit!
    
    # Wait, the user's prompt shows the element they selected:
    # <span class="font-bold text-lg tracking-wider text-white trae-browser-inspect-draggable">From QRCS to Future Scales</span>
    # Let's search for "tracking-wider text-white"
    
    # Let's check where it is
    pass
