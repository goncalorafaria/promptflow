from promptflow import *
from promptflow.functools import *

example_list=["hello ghwi ogwhowhgw o0ghjwohgwog gohw9ohgowhg ohgiouwejgj oghjwoue90 hello"]

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
 
 
word_count_pipeline = (  
    FlatMap(
        func=lambda line: line.split()) 
    | Map(
        func=lambda word: (word, 1),
    ) 
    | Aggregate(
        key_factory = fst, ) 
    | Map(
        func=lambda words: ( sum(map(snd, words)) ))
)

final_process = word_count_pipeline(example_list)

print(final_process)

print(final_process.run())
import pdb; pdb.set_trace()    
    
wf = WordCount()
resultsa = wf(example_list)
resultsb = Iterable(iterable=example_list, keyvalue=False) 
print(resultsa)

import pdb; pdb.set_trace()

