### This file contains functions for the preprocessing of conll files for later evaluation.
# It's primary functionality is to determine if the tokens in the given gold and system files
# are aligned; it writes the results of checking this to a file called 'preprocessing.log'.
# It also contains functionality for cleaning conll files (the convert_conll_file() function), 
# designed with the Spacy and Stanford files in mind: any conll file which contains a) empty lines, 
# b) lines that begin with the string "-DOCSTART-", and c) lines that begin with a quotation mark 
# ('\"') will have these lines removed. In addition, the convert_conll_file() function takes as
# an argument a NERC-tag translation file (a tsv with old and new tags in columns 0 and 1); files
# will have tags converted as specified.


### Command line call: python preprocessing.py gold_filename gold_label_index system_filename system_label_index
### OR: python preprocessing.py
import sys

def import_conll(conll_filename, delimeter='\t', labels=True, token_col=0, label_col=1):
    '''
    Takes a file and extracts relevant information, outputting it in a dict of lists.
    Removes empty lines and document-initial content.
    
    :param conll_filname: a string representing the location of a conll file.
    :param delimeter: the delimeter used in conll_filename
    :param labels: boolean of whether or not the file contains labels
    :param token_col: an integer representing the index of the column containing tokens
    :param label_col: an integer representing the index of the column containing labels
    
    :returns return_data: a dictionary of lists {"tokens":[], "labels":[]}
    '''
    
    return_data = {"tokens":[], "labels":[]}
    with open(conll_filename, 'r') as infile:
        file_data = infile.read().split('\n')
        
    # Set of problematic characters found during parsing/inspection
    invalid_starts = {'-DOCSTART-', '\"'}
        
    for line in file_data:
        # add non-empty and non invalid lines to return_data as lists
        # this deals with formatting differences involving empty lines
        add = True
        for entry in invalid_starts:
            if line.startswith(entry):
                add = False
                break
        if add and line != '':
            
            return_data["tokens"].append(line.split(delimeter)[token_col])
            if labels:
                return_data["labels"].append(line.split(delimeter)[label_col])
    
    return return_data

def extract_tokens(conll_object):
    '''
    Takes a conll array and returns a the list of tokens.
    
    :param1 conll_object: a list of dicts of data from a conll file, as created by import_conll.
        Must be a dictionary with a list as the value of the key "tokens".
   
    :returns tokens: a list of tokens
    '''
    assert "tokens" in conll_object.keys() and type(conll_object["tokens"]) == list, \
            "Argument must be a dictionary with a list as the value of the key 'tokens'"
    return conll_object["tokens"]

def compare_tokens(tokens1, tokens2):
    '''
    Compares two sets of tokens to verify they are identical, returning a boolean.
    
    :param1: tokens1: a list of tokens
    :param2: tokens2: a list of tokens
    
    :return1: matching: a boolean value 'are the two sets identical'
    
    '''
    assert type(tokens1) == type(tokens2) == list, "Arguments must both be lists"
    return tokens1 == tokens2

def extract_label_set(conll_object):
    '''
    Takes a conll object and outputs a set of NERC labels.
    
    :param conll_object: a conll dict of lists, as created by import_conll()
    
    :returns a set of ne labels utilized in conll_object
    '''

    return set(conll_object["labels"])

def convert_conll_file(conll_object, translation_table, output_file_name, 
                       token_translation_dict={"-LRB-":"(", "-RRB-":")"}):
    '''
    Writes given .conll object, as produced by import_conll(), to file with converted
    NER annotations.
    
    :param1: conll_object: a dict of lists of data as produced by import_conll()
    :param2: translation_table: the location of a .tsv file containing the translation table
    :param3: output_file_name
    :param4: token_translation_dict: a dictionary of token translations to perform. Default value
        includes 'problematic' tokens already encountered.
    
    '''
    new_conll = {"tokens":[], "labels":[]}
    translation_dict = dict()
    
    with open(translation_table, 'r') as infile:
        for line in infile:
            label1, label2 = line.strip('\n').split('\t')
            translation_dict[label1] = label2
    
    outfile = open(output_file_name,'w')
    # convert annotation
    for label in conll_object["labels"]:
        if label in translation_dict:
            new_label = translation_dict[label]
            new_conll["labels"].append(new_label)
        else:
            new_conll["labels"].append(label)
        
    # convert tokens
    for token in conll_object["tokens"]:
        if token in token_translation_dict:
            new_token = token_translation_dict[token]
            new_conll["tokens"].append(new_token)
        else:
            new_conll["tokens"].append(token)
    
    # write new_conll to output_file
    for token, label in zip(new_conll["tokens"], new_conll["labels"]):
        outfile.write(f"{token}\t{label}\n")
    outfile.close()
    
    return

def main():
    # First, import: collect filenames from command line arg and run import_conll
    if len(sys.argv) != 5:
        cont = input("Invalid number of arguments, would you like to enter manually? y/n: ")
        if cont == "y":
            gold_file = input("Enter the path to file1 (the gold file): ")
            gold_label_index = int(input("Enter the column index of the labels in file1: "))
            system_out_file = input("Enter the path to file2 (the system file): ")
            system_label_index = int(input("Enter the column index of the labels in file2: "))
        else:
            print("Invalid input, program will terminate.")
            return
    else:
        gold_file = sys.argv[1]
        gold_label_index = int(sys.argv[2])
        system_out_file = sys.argv[3]
        system_label_index = int(sys.argv[4])
        
    gold_data = import_conll(gold_file, label_col=gold_label_index)
    system_data = import_conll(system_out_file, label_col=system_label_index)

    # Next, extract tokens
    gold_tokens = extract_tokens(gold_data)
    system_tokens = extract_tokens(system_data)

    # Compare tokens
    comparison = compare_tokens(gold_tokens, system_tokens)
    # Write comparison to preprocessing.log
    if comparison:
        outstring = f"{gold_file} {system_out_file}: ALIGN"
    else:
        outstring = f"{gold_file} {system_out_file}: MISSMATCH"
    with open("preprocessing.log","w") as outfile:
        outfile.write(outstring)
    
    print(outstring)
    # Give user option to convert labels:
    convert_gold = input(f"Would you like to convert the labels and/or tokens in {gold_file}? y/n: ")
    new_gold_filename = input("Enter a desired path for converted file: ")
    if convert_gold == "y":
        convert_conll_file(gold_data, "translation_table.tsv", new_gold_filename)
        with open("preprocessing.log", "a") as outfile:
            outfile.write(f"\n{new_gold_filename} contains CONVERSIONS")
    convert_system = input(f"Would you like to convert the labels and/or tokens in {system_out_file}? y/n: ")
    new_system_filename = input("Enter a desired path for converted file: ")
    if convert_system == "y":
        convert_conll_file(system_data, "translation_table.tsv", new_system_filename)
        with open("preprocessing.log", "a") as outfile:
            outfile.write(f"\n{new_system_filename} contains CONVERSIONS")
            
if __name__ == '__main__':
    main()



