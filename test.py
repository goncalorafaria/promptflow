from promptflow import *
from promptflow.functools import *

example_list=["hello ghwi ogwhowhgw o0ghjwohgwog  gohw9ohgowhg ohgiouwejgj oghjwoue90 hello"]

class WordCount(WorkFlow):

    def forward(self, input_text):
        
        a = Iterable(iterable=example_list, keyvalue=False)
     
        b = (  FlatMap(
                func=lambda line: line.split()) 
        | Map(
            func=lambda word: (word, 1),
        ) 
        | Aggregate(key_factory = fst ) 
        | Map(func=lambda words: ( sum(map(snd, words)) ))
        )


        return b(a)
 
 
b = (  FlatMap(
        func=lambda line: line.split()) 
| Map(
    func=lambda word: (word, 1),
) 
| Aggregate(key_factory = fst, ) 
| Map(func=lambda words: ( sum(map(snd, words)) ))
)

resultsb = Iterable(iterable=example_list, keyvalue=False) 
wf1 = convert_to_workflow(b)

import pdb; pdb.set_trace()    
    
wf = WordCount()
resultsa = wf(example_list)
resultsb = Iterable(iterable=example_list, keyvalue=False) 
print(resultsa)

import pdb; pdb.set_trace()

