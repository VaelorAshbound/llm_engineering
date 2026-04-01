#!/usr/bin/env python
# coding: utf-8

# # RAG Day 4
#
# ## Evaluation!
#
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Keep in mind how you would evaluate RAG for your business</h2>
#             <span style="color:#181;">This is such an important part of building an accurate and reliable RAG pipeline. And it's applicable to many aspects of solving business problems with LLMs. People are often focused on RAG architecture and RAG frameworks for their business. But even more important: evaluations!</span>
#         </td>
#     </tr>
# </table>

# %%


from collections import Counter

from evaluation import test
from evaluation.eval import evaluate_answer, evaluate_retrieval

# %%


tests = test.load_tests()


# %%


len(tests)


# %%


example = tests[0]
print(example.question)
print(example.category)
print(example.reference_answer)
print(example.keywords)


# %%


count = Counter([t.category for t in tests])
_ = count


# %%


# %%


evaluate_retrieval(example)


# %%


eval, answer, chunks = evaluate_answer(example)


# %%


_ = eval


# %%


print(eval.feedback)
print(eval.accuracy)
print(eval.completeness)
print(eval.relevance)


# %%
