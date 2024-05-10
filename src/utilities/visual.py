import os
from graphviz import Digraph
from src.utilities.colors import colors


def draw_net(stitchNet, name="_results_with_finetune/stitchnet/net"):
    name = os.path.splitext(name)[0]
    hexColor = {}
    rep = stitchNet.get_id()
    # order based on the nets input when fragments are generated
    nIds=["alexnet","densenet121","mobilenet_v3_small","resnet50","vgg16"]
    for i,nId in enumerate(nIds):
        hexColor[i] = colors[i%len(colors)] #rgb_to_hex(*[int(c*255) for c in colors[i%len(colors)]])
        
    dot = Digraph(comment='StitchNet')
    dot.graph_attr['rankdir'] = 'BT'  
    # dot.graph_attr['minlen'] = '0.1'  
    for r in rep:
        dot.node(str(r), f'F{r[1]} of {nIds[r[0]]}', fontcolor="white", color=hexColor[r[0]], style="filled", shape="rectangle")
    
    for i,f1 in enumerate(rep[:-1]):
        f2 = rep[i+1]
        dot.edge(str(f1), str(f2), label=str(i))
        # dot.edge(str(f1), str(f2), minlen="1")
            
    dot.format = 'svg'
    print('saving to', f'{name}.{dot.format}')
    dot.render(f'{name}', view=False)
    return dot


def draw_stitchNet(nets, stitchNet, name="_results_with_finetune/stitchnet/net"):
    hexColor = {}
    for i,net in enumerate(nets):
        hexColor[net.get_id()] = colors[i%len(colors)] #rgb_to_hex(*[int(c*255) for c in colors[i%len(colors)]])
        
    dot = Digraph(comment='StitchNet')
    dot.graph_attr['rankdir'] = 'BT'  
    # dot.graph_attr['minlen'] = '0.1'  
    rep = stitchNet.get_id()
    for r in rep:
        dot.node(str(r), f'F{r[1]} of N{r[0]}', fontcolor="white", color=hexColor[r[0]], style="filled", shape="rectangle")
    
    for i,f1 in enumerate(rep[:-1]):
        f2 = rep[i+1]
        dot.edge(str(f1), str(f2), label=str(i))
        # dot.edge(str(f1), str(f2), minlen="1")
            
    dot.format = 'svg'
    dot.render(f'{name}', view=False)
    return dot, f'{name}.{dot.format}'


def draw_stitchNet_fromTuples(fragmentTuples, numNets=10, name="_results_with_finetune/stitchnet/net"):
    hexColor = {}
    for i in range(numNets):
        hexColor[i] = colors[i%len(colors)] #rgb_to_hex(*[int(c*255) for c in colors[i%len(colors)]])
        
    dot = Digraph(comment='StitchNet')
    dot.graph_attr['rankdir'] = 'BT'  
    # dot.graph_attr['minlen'] = '0.1'  
    rep = fragmentTuples
    for r in rep:
        dot.node(str(r), f'F{r[1]} of N{r[0]}', fontcolor="white", color=hexColor[r[0]], style="filled", shape="rectangle")
    
    for i,f1 in enumerate(rep[:-1]):
        f2 = rep[i+1]
        dot.edge(str(f1), str(f2), label=str(i))
        # dot.edge(str(f1), str(f2), minlen="1")
            
    dot.format = 'svg'
    dot.render(f'{name}', view=False)
    return dot, f'{name}.{dot.format}'