import pandas as pd

def read_treekey(file):
    '''
    file is a csv containing columns:

    Junction, Node.From, Width.From
    '''
    treekey = []
    df = pd.read_csv(file, usecols=['Junction','Width.From','Node.From', 'Turn.Type'])
    for index, row in df.iterrows():
        edge = f"{int(row[0])} {int(row[2])} {float(row[1])}"
        if edge not in treekey:
            treekey.append(edge)
    return treekey

def read_predictions(file):
    '''
    file is a csv containing columns:

    Junction, Node.From, sharpIsLeft
    '''
    predictions = {}
    df = pd.read_csv(file, usecols=['Junction','Node.From','sharpIsLeft','Sharp.Turn.Prob', 'U.Turn.Prob', 'Turn.Probability'])
    for index, row in df.iterrows():
        approach = (str(row[1]),str(row[0])) # from, to
        if approach not in predictions.keys():

            if row[2]: #if sharp is left
                predictions[approach] = (row[4], row[3], row[5]) #ULR
            else:
                predictions[approach] = (row[4], row[5], row[3]) #ULR
    return predictions