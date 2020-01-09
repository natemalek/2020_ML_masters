### This file operates on tables (lists of lists) to evaluate the performance (precision, recall, f-score)
### of a classification system against a gold standard.
# Given a gold file and a set of system files, outputs a latex table evaluating all systems against the gold.
# To pass each system, include the system name (a shorthand) and the system file path.
# Any number of systems can be passed, and the resulting table contains the evaluation statistics for each of the passed systems.

# In addition to the output of a latex table, an evaluation_outcome.txt file is written which also contains the evaluation
# statistics.

### Command call: 'python evaluation.py gold_filename system1_name system1_filename system2_name system2_filename system3_name ...'
import sys
import pandas as pd
from preprocess import import_conll, extract_tokens, compare_tokens
from collections import defaultdict

def evaluate_predictions(gold_labels, system_labels):
    '''
    Computes precision, recall and f-scores for each label in a gold and a system output set
    
    :param gold_labels: a list of gold labels
    :param system_labels: a list of system labels
    
    :returns a dictionary of dictionaries of the form:
        {"label1":{"precision":0.XXX, "recall":0.XXX, "f-score":0.XXX}
         "label2":{"precision":0.XXX, "recall":0.XXX, "f-score":0.XXX}
         ...
        }
    '''
    assert len(gold_labels) == len(system_labels), \
    "Error: uncomparable data sets: mismatching length"
    
    # First, iterate once through gold and system labels, counting True Positives,
    # False Negatives, and False Positives for each label
    
    # This dict looks like: {'label1':{'TP':XXX,'FP':XXX,'FN':XXX},'label2':{...}...}
    count_dicts = defaultdict(lambda: {"TP":0,
                                       "FP":0,
                                       "FN":0})
    for i in range(len(gold_labels)):
        gold_label = gold_labels[i]
        system_label = system_labels[i]
        if gold_label == system_label:
            # True Positive
            count_dicts[gold_label]["TP"] += 1
        elif gold_label != system_label:
            # False Positive
            count_dicts[system_label]["FP"] += 1
            # False Negative
            count_dicts[gold_label]["FN"] += 1
    
    # Now, iterate through count_dicts computing precision, recall and f-score for each label
    # precision = TP/(TP+FP), recall = TP/(TP+FN), f-score = 2*(precision*recall)/(precision+recall)
    
    # This dict looks like: {'label1':{'precision':0.XXX,'recall':0.XXX,'f-score':0.XXX},'label2':{...}...}
    evaluation_dict = defaultdict(lambda: {"precision":0,
                                           "recall":0,
                                           "f-score":0})
    for label, count_dict in count_dicts.items():
        TP = count_dict["TP"]
        FP = count_dict["FP"]
        FN = count_dict["FN"]
        if (TP+FP) == 0:
            precision = 0
        else:
            precision = TP/(TP+FP)
        if (TP+FN) == 0:
            recall = 0
        else:
            recall = TP/(TP+FN)
        if (precision + recall) == 0:
            f_score = 0
        else:
            f_score = (2*(precision*recall)) / (precision+recall)
        evaluation_dict[label]["precision"] = str('%.3f' % precision)
        evaluation_dict[label]["recall"] = str('%.3f' % recall)
        evaluation_dict[label]["f-score"] = str('%.3f' % f_score)
    
    return evaluation_dict

def main():
    # Parse info from command line
    gold_file = sys.argv[1]
    system_list = []
    for i in range(2, len(sys.argv), 2):
        try: 
            system_name = sys.argv[i]
            system_file = sys.argv[i+1]
            system_list.append([system_name, system_file])
        except:
            print("Argument Error: Need at least one system and two"
                  + " arguments for each system: name, filepath")
            return

    custom_indices = False
    columns = input("Do all labels appear in the second column (index 1)? y/n: ")
    if columns == "n":
        custom_indices = True
        gold_label_col = int(input("Enter the index of the labels in the gold file: "))
        for system in system_list:
            label_col = int(input(f"Enter the index of the labels in the {system[0]}: "))
            system.append(label_col)
            
    # import conll data
    if custom_indices:
        gold_data = import_conll(gold_file, label_col=gold_label_col)
    else:
        gold_data = import_conll(gold_file)
    gold_labels = gold_data["labels"]

    table_list = []
    outstring = "" # this is for evaluation_outcome.txt file
    
    for system in system_list:
        system_name = system[0]
        system_file = system[1]
        
        if custom_indices:
            system_data = import_conll(system_file, label_col=system[2])
        else:
            system_data = import_conll(system_file)

        #assert compare_tokens(extract_tokens(gold_data), extract_tokens(system_data)), \
        #"Uncomparable files: tokens must align"

        # Extract label lists and run evaluation
        system_labels = system_data["labels"]
        evaluation_dict = evaluate_predictions(gold_labels, system_labels)
        df = pd.DataFrame(evaluation_dict).T
        table_list.append(df)
        
        # construct outstring for writing to evaluation_outcome.txt
        for label, label_dict in evaluation_dict.items():
            for metric, value in label_dict.items():
                outstring += f'{system_name} {label} {metric} {value}\n'

    concat_table = pd.concat(table_list, keys=[system[0] for system in system_list])
    concat_table.to_latex('latex_table.tex')
   
    with open('evaluation_outcome.txt', 'w') as outfile:
        outfile.write(outstring)
        
if __name__ == '__main__':
    main()
