import seaborn

def rgb_to_hex(r, g, b):
    return ('#{:X}{:X}{:X}').format(r, g, b)

seaborn.set_theme()
colors = seaborn.color_palette()
colors = [rgb_to_hex(int(r*255),int(g*255),int(b*255)) for r,g,b in colors]