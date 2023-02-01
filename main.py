from macq.generate.pddl import StateEnumerator, VanillaSampling
from pddl_vis.visualizers import VISUALIZERS
from pddl_vis.dataset import visualize_traces
import networkx as nx
from PIL import Image
from io import BytesIO
import numpy as np
import copy



def colour_state(state, graph, colour='#bababa'):
    graph = graph.copy()
    graph.nodes[state]['style'] = 'filled'
    graph.nodes[state]['fillcolor'] = colour
    return graph



def get_graph_img(graph, file_name=None):
    dot_graph = nx.nx_pydot.to_pydot(graph)
    dot_graph.dpi = 500
    img = Image.open(BytesIO(dot_graph.create_png())).convert('RGB')
    if file_name is not None:
        img.save(file_name)
    return img




if __name__ == '__main__':

    # Domain and problem files
    # In data/pddl
    # Domains are: blocks, elevator, grid, hanoi, slide_tile
    # Problems are in the problem folder for each domain

    # Problem files, referenced as domain# (ex. blocks1, grid3, etc) - increase in size with number
    # blocks 1, 2, 3
    # elevator 1, 2
    # grid 1, 2, 3, 4
    # hanoi 1, 2, 3
    # slide_tile 1, 2, 3

    # Removed problems that are too large to load
    domains_and_problems = {'blocks' : ['blocks1', 'blocks2'],
                            'elevator' : ['elevator1', 'elevator2'], 
                            'grid' : ['grid1', 'grid2'],
                            'hanoi' : ['hanoi1', 'hanoi2'],
                            'slide_tile' : ['slide_tile1']}

    for domain in domains_and_problems.keys():
        for problem in domains_and_problems[domain]:

            domain_file = 'data/pddl/' + domain + '/' + domain + '.pddl'
            problem_file = 'data/pddl/' + domain + '/problems/' + problem + '.pddl'
            trace_len = 10
            n_traces = 1
            action_labels = True      # Set to false to generate graph without action labels


            # Loads domain and problem as a macq object - StateEnumerator is a Generator you can get the state space from
            generator = StateEnumerator(
                dom=domain_file, 
                prob=problem_file
            )

            states = generator.graph.nodes()
            print(f"Loaded in {domain} domain and {problem} problem, has {len(states)} states\n")

            # VanillaSampling is another object that inherits from Generator - but need this to generate traces
            # There's no Generator object in macq you can get the state space from and generate traces
            trace_generator = VanillaSampling(
                dom=domain_file, 
                prob=problem_file,
                plan_len=trace_len,
                num_traces=n_traces
            )

            print(f"\nFinished generating {n_traces} traces of length {trace_len}")

            macq_states = list(map(generator.tarski_state_to_macq, states))         # States are tarski objects, make them macq objects
            state_hashes = list(map(hash, map(str, macq_states)))                   # Hashing the states since we have two generator objects - use hashes to figure out what states are the same


            # Get the state (as a number) for each state in the traces
            trace_states = []
            for trace in trace_generator.traces:
                trace_states.append([])
                for step in trace_generator.traces[0]:
                    trace_states[-1].append(state_hashes.index(hash(str(step.state))))

            # Trace states should be (n_traces, trace_len)


            # Relabel nodes in state graph so they can be displayed nicely
            state_mapping = dict(zip(states, range(len(states))))
            state_graph = nx.relabel_nodes(generator.graph, state_mapping)


            # Colour edges based on the action, and relabel (action labels are too long otherwise)
            actions = []
            colours = ['#72EC91', '#C07DFF', '#4D7DEE', '#FF5A5F', '#C81D25']       # Can create colour palettes here: https://coolors.co/ - kind of fun
            for _, _, act in state_graph.edges(data=True):
                action = str(act['label']).split('(')[0]
                if action in actions:
                    act['color'] = colours[actions.index(action)]
                else:
                    actions.append(action)
                    if len(actions) > len(colours):
                        print('Need more colours')
                        exit()
                    act['color'] = colours[len(actions) - 1]
                if action_labels:
                    act['label'] = '  ' + action + '  '
                else:
                    act['label'] = ' '

            
            # Save the state space graph
            get_graph_img(state_graph, f'results/state_spaces/{domain}_{problem}_state_space.jpg')


            # Get visualizations of the traces
            visualizer = VISUALIZERS[domain](generator)     # List of visualizers, can be found in __init__.py in pddl_vis/visualizers


            # Get gifs of vis and highlighted graph for the first trace
            state_imgs = list(map(visualizer.visualize_state, trace_generator.traces[0]))
            graph_imgs = [get_graph_img(colour_state(state, state_graph)) for state in trace_states[0]]

            state_imgs[0].save(f'results/vis_gifs/{domain}_{problem}_vis.gif', save_all=True, append_images=state_imgs[1:], duration=1000, loop=0)
            graph_imgs[0].save(f'results/graph_gifs/{domain}_{problem}_graph.gif', save_all=True, append_images=graph_imgs[1:], duration=1000, loop=0)