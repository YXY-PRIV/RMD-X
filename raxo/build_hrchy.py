"""
INTEGRATED WITH WORDNET
TO DO: LLM chatgpt
"""

import argparse
import nltk
from nltk.corpus import wordnet as wn
import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from llmcontrollers import HrchyPrompter, LLMBot

parser = argparse.ArgumentParser(description='Construct hierarchy for a set of classes')
parser.add_argument('--categories', required=True, help='Path to the .json with the categories names')
parser.add_argument('--method', choices=['wordnet','llm'], required=True, help='wordnet or llm')
parser.add_argument('--out', required=True, help='Path to the folder where all the information will be saved')
parser.add_argument('--n_hyponyms', required=True, help='Number of hyponims')



def plot_hierarchy_tree(hierarchy, out_file_name):

    # Create a directed graph
    G = nx.DiGraph()

    # Add the nodes and edges based on the hierarchy
    G.add_node(hierarchy["Supercategory"])  # Add Supercategory node
    G.add_node(hierarchy["Cat"])            # Add Cat node

    # Add edges from Supercategory to Cat
    G.add_edge(hierarchy["Supercategory"], hierarchy["Cat"])

    # Add Subcategories and edges
    for subcategory in hierarchy["Subcategories"]:
        G.add_node(subcategory)  # Add each Subcategory node
        G.add_edge(hierarchy["Cat"], subcategory)  # Edge from Cat to Subcategories

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # This is more specific for tree layouts

    # Color list for nodes
    node_colors = []
    for node in G.nodes():
        if node == hierarchy["Cat"]:  # Color the "Cat" node differently
            node_colors.append("lightgreen")  # Color for the "Cat" node
        else:
            node_colors.append("lightblue")  # Default color for other nodes


    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, font_size=12, font_weight="bold", edge_color="gray")
    plt.title("Hierarchy Tree")
    #plt.show()
    plt.savefig(out_file_name)




def build_tree_wordnet(categories, n, out_folder, save_trees_plot=False):
    """
    Build a 3-level hierarchy where the top-level (supercategory) is a single hypernym, 
    the middle level is the given concept, and the bottom level (subcategories) consists of n hyponyms.
    Note: Supercategories are not necessary.
    """

    nltk.download('wordnet')
    hierarchies = []

    for c in categories:
        synsets = wn.synsets(c)
    
        if not synsets:
            print(f"No synsets found for the word '{c}'")
        else:
            # Take the first synset as the primary sense
            synset = synsets[0]
            
            # Retrieve the single hypernym (supercategory)
            hypernyms = synset.hypernyms()
            if not hypernyms:
                supercategory = "None"
                print(f"No hypernyms found for '{c}' category")
            else:
                supercategory = hypernyms[0].name().split('.')[0]
            
            # Retrieve up to n hyponyms (subcategories)
            hyponyms = [hyponym.name().split('.')[0] for hyponym in synset.hyponyms()[:n]]
            if len(hyponyms) < n:
                print(f"Not enough hyponyms found for '{c}' category")
            
            # Build the hierarchy
            hierarchy = {
                "Supercategory": supercategory,
                "Cat": c,
                "Subcategories": hyponyms,
            }
            hierarchies.append(hierarchy)

            if save_trees_plot:
                plot_hierarchy_tree(hierarchy, os.path.join(out_folder, f"tree_{c}"))

    
    return hierarchies


def build_tree_llm(categories, n, out_folder, save_trees_plot=False):
    bot = LLMBot('gpt-4o')
    query_times=1
    hierarchies = []
    h_prompter = HrchyPrompter(num_sub=n)

    for c in categories:
        cpt = h_prompter.query_hyponims(category=c)
        child_answers = [bot.infer(cpt, temperature=0.7) for i in range(query_times)]
        
        print(f"Question: {cpt}")
        for i in range(query_times):
            print(f"Answer {1+i}: {child_answers[i]}")

        # Clean elements
        clean_childs = {
            name.strip()
            for dirty_child in child_answers
            for name in dirty_child.split('&')
            if 3 <= len(name.strip()) <= 100
        }
        print(clean_childs)
        hierarchy = {
            "Supercategory": "None",
            "Cat": c,
            "Subcategories": clean_childs,
        } 
        hierarchies.append(hierarchy)
        if save_trees_plot:
            plot_hierarchy_tree(hierarchy, os.path.join(out_folder, f"tree_{c}"))

    return hierarchies


def build_json(tree_hrchy):
    # The output format should be something like:
    # {
    #     "category_list_real":[
    #         "Pressure_vessel", "Pressure_vessel", "Pressure_vessel", "Pressure_vessel", "Razor_blade","Razor_blade", "Razor_blade".....]
    #     ],
    #     "category_list_find":[
    #         "Pressure_vessel1", "Pressure_vessel2 ", "Pressure_vessel3", "Pressure_vessel4", "Pressure_vessel5", "Razor_blade1", "Razor_blade2", "Razor_blade3",......]        
    #     ]
    # }

    category_list_real = []
    category_list_find = []

    for element_cat in tree_hrchy:
        # Always insert the propio element_cat in the searching list
        category_list_real.append(element_cat['Cat'])
        category_list_find.append(element_cat['Cat'])
        for hiponym in element_cat["Subcategories"]:
            category_list_real.append(element_cat['Cat'])
            category_list_find.append(hiponym)
    
    final_json = {"category_list_real": category_list_real, "category_list_find": category_list_find}
    print(final_json)
    return final_json




def main(args):
    os.makedirs(args.out, exist_ok=True)
    categories = json.load(open(args.categories))
    # Extract categories sorted by id
    sorted_categories = sorted(categories["categories"], key=lambda x: x["id"])
    # Map to a new list with names only
    categories = [category["name"] for category in sorted_categories]
    print(categories)

    if args.method=="wordnet":
        tree_hrchy = build_tree_wordnet(categories, int(args.n_hyponyms), args.out, save_trees_plot=False)

    elif args.method=="llm":
        tree_hrchy = build_tree_llm(categories, int(args.n_hyponyms), args.out, save_trees_plot=False)

    # Convert tree_hrchy to a format accepted by google_image_retrievalv2.py
    json_hrchy = build_json(tree_hrchy)
    # Save json
    name = os.path.join(args.out, "categories.json")
    with open(name, 'w') as f:
        json.dump(json_hrchy, f)


   
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

   
        
