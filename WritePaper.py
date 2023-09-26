from pylatex import Document, Section, Subsection, Command, Subsubsection, NewLine

class DocGraph():
    def __init__(self,name):
        self.description = ""
        self.name = name
        self.nodes = []
        
    def createGraph(self):
        from pyvis.network import Network

        net = Network(directed=True)

        # add the first node
        net.add_node("Input", label="Input", shape="box", level=0)

        mappings = [
            "Input"
        ]
        level = 0

        for node in self.nodes:
            type = node.split(" ")[0]
            targets = node.split(" ")[1:]
            level+=1
            if targets.__len__() > 2 and targets[-2] == "SetHeight":
                level = int(targets[-1])
                
            if type == "Chunk":
                name = f"Chunk-{mappings.__len__()}"
                mappings.append(f"{name}")
                net.add_node(f"{name}", label="Chunk", shape="box", level=level)
                net.add_edge(mappings[int(targets[0])], f"{name}")
                


            elif type == "Shift":
                name = f"Shift-{mappings.__len__()}"
                mappings.append(f"{name}")
                for i in range(int(targets[1])):
                    net.add_node(f"{name}-{i}", label=f"Shift: {targets[2].split(',')[i]}", shape="box", level=level)
                    net.add_edge(f"{mappings[int(targets[0])]}", f"{name}-{i}")
            
            elif type == "Cat":
                name = f"Cat-{mappings.__len__()}"
                mappings.append(f"{name}")
                net.add_node(f"{name}", label=f"Concatonate", shape="box", level=level)
                for i in range(int(targets[1])):
                    net.add_edge(f"{mappings[int(targets[0])]}-{i}", f"{name}")

            

            elif type == "Mult":
                name = f"{type}-{mappings.__len__()}"
                mappings.append(f"{name}")
                net.add_node(f"{name}", label=f"{type}", shape="box", level=level)
                net.add_edge(f"{mappings[int(targets[0])]}", f"{name}")
                net.add_edge(f"{mappings[int(targets[1])]}", f"{name}")
                
            elif type == "Sum":
                name = f"Sum-{mappings.__len__()}"
                mappings.append(f"{name}")
                net.add_node(f"{name}", label=f"Sum", shape="box", level=level)
                for i in range(int(targets[1])):
                    net.add_edge(f"{mappings[int(targets[0])]}-{i}", f"{name}")

            else:
                name = f"{type}-{mappings.__len__()}"
                mappings.append(f"{name}")
                net.add_node(f"{name}", label=f"{type}", shape="box", level=level)
                net.add_edge(f"{mappings[int(targets[0])]}", f"{name}")
                

        net.generate_html("graph.html")

def getDocumentation(filename):
    with open(filename, "r") as f:
        v = f.read()
    
    # split newline
    v = v.split("\n")
    docs = {
    }

    currentdoc = None
    graphing = False
    for line in v:
        if line.startswith("#####"):
            graphing = False
            currentdoc = None
        elif line.startswith("####"):
            if graphing:
                docs[currentdoc].nodes.append(line[5:])
        elif line.startswith("###"):
            currentdoc = line[4:]
            docs[currentdoc] = DocGraph(currentdoc)
        elif line.startswith("## Graphing"):
            graphing = True
        elif line.startswith("#"):
            if currentdoc is not None:
                docs[currentdoc].description += line[1:] + "\n"
        
        
        

    return docs

def write_paper():
    name = "SuperFFN Network"
    # do mkdir ./paper/
    import os
    if not os.path.exists("./paper/"):
        os.mkdir("./paper/")
    doc = Document("./paper/"+name.replace(" ", "_").lower() + "_paper", page_numbers=False)
    doc.preamble.append(Command('title', name))
    # get git username and email
    import subprocess
    username = subprocess.check_output(["git", "config", "user.name"]).decode().strip()

    doc.preamble.append(Command('author', username))
    doc.preamble.append(Command('date',  Command('today')))
    doc.append(Command('maketitle'))
    doc.append(Section('Introduction'))
    
    # convert readme to latex
    with open("./README.md", "r") as f:
        readme = f.read()
    
    isopenbrackets = False
    for line in readme.split("\n"):
        if line.startswith("##"):
            doc.append(Subsubsection(line[2:]))
        elif line.startswith("#"):
            doc.append(Subsection(line[1:]))
        elif line.startswith("###"):
            doc.append(Subsubsection(line[4:]))
        
        elif line.startswith("```"):
            
            if isopenbrackets:
                doc.append(Command("end", "lstlisting"))
                isopenbrackets = False

            else:
                doc.append(Command("begin", "lstlisting"))
                isopenbrackets = True
            
            doc.append(line[3:])
        else:
            if line == "":
                pass
            else:
                doc.append(line)
                doc.append(NewLine())

            
    docs = getDocumentation("./src/RWKVTools/modules/FFN.py")

    doc.append(Section("Documentation"))
    for key in docs:
        doc.append(Subsection(key))
        doc.append(docs[key].description)
        doc.append(Subsubsection("Graphing"))
        for node in docs[key].nodes:
            doc.append(node)
            doc.append(NewLine())

        docs[key].createGraph()
        
            
            


    # make pdf
    doc.generate_pdf(compiler='pdflatex', clean_tex=False)

if __name__ == "__main__":
    write_paper()
