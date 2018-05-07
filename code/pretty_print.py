"""This file contains functions to pretty-print a SQuAD example"""

from colorama import Fore, Back, Style
from vocab import _PAD

def yellowtext(s):
    return Fore.YELLOW + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def greentext(s):
    return Fore.GREEN + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redtext(s):
    return Fore.RED + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redback(s):
    return Back.RED + s + Back.RESET

def magentaback(s):
    return Back.MAGENTA + s + Back.RESET



def print_example(word2id, context_tokens, qn_tokens, true_ans_start, true_ans_end, pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em):
    """
    Pretty-print the results for one example.
    """
    
    ss=" ".join(context_tokens)
    curr_context_len = len(context_tokens)

    # Highlight out-of-vocabulary tokens in context_tokens
    context_tokens = [w if w in word2id else "_%s_" % w for w in context_tokens]

    # Highlight the true answer green.
    truncated = False
    for loc in range(true_ans_start, true_ans_end+1):
        if loc in range(curr_context_len):
            context_tokens[loc] = greentext(context_tokens[loc])
        else:
            truncated = True

    assert pred_ans_start in range(curr_context_len)
    assert pred_ans_end in range(curr_context_len)

    context_tokens[pred_ans_start] = magentaback(context_tokens[pred_ans_start])
    context_tokens[pred_ans_end] = redback(context_tokens[pred_ans_end])

    print ("CONTEXT: (%s is true answer, %s is predicted start, %s is predicted end, _underscores_ are unknown tokens). Length: %i" % (greentext("green text"), magentaback("magenta background"), redback("red background"), len(context_tokens)))
    print (" ".join(context_tokens))

    question = " ".join(qn_tokens)

    print (yellowtext("{:>20}: {}".format("QUESTION", question)))
    if truncated:
        print (redtext("{:>20}: {}".format("TRUE ANSWER", true_answer)))
        print (redtext("{:>22}(True answer was truncated from context)".format("")))
    else:
        print (yellowtext("{:>20}: {}".format("TRUE ANSWER", true_answer)))
    print (yellowtext("{:>20}: {}".format("PREDICTED ANSWER", pred_answer)))
    print (yellowtext("{:>20}: {:4.3f}".format("F1 SCORE ANSWER", f1)))
    print (yellowtext("{:>20}: {}".format("EM SCORE", em)))
    print ("")
    