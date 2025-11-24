# external imports
import abc
import asyncio
import logging
from typing import Any, List

# internal imports
from tinyflow.actor import Actor


class WorkFlow:
    """Pipeline abstract class.

    This entity is an abstract for every pipeline definition.
    """

    def __init__(self, name=""):
        """creates a Workflow object."""
        self.name = name

    def __display__(self):
        logging.info(f"{self.__repr__} args")
        logging.info(self.__dict__)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """This function defines the pipeline logic. It must be implemented by any pipeline."""
        raise NotImplementedError()

    def show(self, *args, save_img=False, **kwargs):
        """Plots the workflow diagram.

        Does a reverse breath first transversal to plot the workflow.
        """

        output_node = self.forward(*args, **kwargs)

        if not isinstance(output_node, tuple):
            node_dict = {output_node: {}}
            frontier = [output_node]
        else:
            node_dict = {}
            frontier = []

            for node in output_node:
                node_dict[node] = {}
                frontier.append(node)

        edges = []

        while len(frontier) > 0:

            new_frontier = []

            for leaf in frontier:

                for node in leaf.parents:

                    if node not in node_dict:

                        new_frontier.append(node)
                        node_dict[node] = node.children.keys()

                        for child in node.children.keys():
                            edges.append((node.name, child.name))

            frontier = new_frontier

        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure(figsize=(18, 18))

        G = nx.DiGraph(edges)
        pos = nx.fruchterman_reingold_layout(G)  # Seed layout for reproducibility
        nx.draw(
            G,
            pos,
            node_color="b",
            node_size=400,
            with_labels=True,
            font_size=12,
        )

        if save_img:
            plt.savefig("workflow.pdf")

        else:
            plt.show()

    async def __forward_and_run(self, *args, **kwargs) -> List[Any]:
        """Auxiliary function to define and run the pipeline.

        Returns:
            List[Any]: One result for each output actor of the pipeline.
        """

        output = self.forward(*args, **kwargs)
        results = await WorkFlow.run(output)

        return results

    def __call__(self, *args, **kwargs) -> List[Any]:
        """Function for defining, runing and waiting for the pipeline upon execution.

        Returns:
            List[Any]: One result for each output actor of the pipeline.
        """
        return asyncio.run(self.__forward_and_run(*args, **kwargs))

    
    async def run(*kwargs: List[Actor]) -> List[Any]:
        """Function to run the pipeline.

        Returns:
            List[Any]: One result for each output actor of the pipeline.
        """

        logging.debug("--" * 40)
        logging.debug("Starting breath transversal.")

        tasks = []
        initial_frontier = []
        visited = set()

        for output in kwargs:
            if isinstance(output, list):
                initial_frontier.extend(output)

            if isinstance(output, tuple):
                initial_frontier.extend(output)

            else:
                initial_frontier.append(output)

        frontier = list(initial_frontier)

        while len(frontier) > 0:

            logging.debug(frontier)

            next_frontier = []

            for node in frontier:

                if node.runnable() or node.source():

                    if node.source():

                        tasks.append(asyncio.create_task(node.feed()))

                    else:
                        op, parents = node.execution_context()

                        for p in parents:
                            if p not in visited:
                                visited.add(p)
                                next_frontier.append(p)

                        # if len(parents) == 1:
                        #    parents = parents[0]

                        tasks.append(
                            asyncio.create_task(op.execute(*parents, output=node))
                        )

            frontier = next_frontier

        logging.debug("Terminated breath transversal.")
        logging.debug("--" * 40)
        logging.info("Sucessfully running every process.")

        for output in initial_frontier:
            tasks.append(asyncio.create_task(output.tolist()))

        noutputs = len(initial_frontier)

        results = await asyncio.gather(*tasks)

        return results[-noutputs:]
