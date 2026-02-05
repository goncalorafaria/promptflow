from promptflow.process import ProcessUnion, Bridge
from promptflow.workflow import WorkFlow


### I would like to build a function that given some unionprocess turns it into a workflow. 

def convert_to_workflow(process: ProcessUnion) -> WorkFlow:
    """
    Given a ProcessUnion, turns it into a Workflow.
    """
    
    class InplaceWorkflow(WorkFlow):
        def forward(self, *args, **kwargs):
            return process(*args, **kwargs)
    
    return InplaceWorkflow()

def convert_from_bridge(bridge: Bridge) -> WorkFlow:
    """
    Given a Bridge, turns it into a Workflow.
    """
    class InplaceWorkflow(WorkFlow):
            def forward(self):
                return bridge
        
    return InplaceWorkflow()