import pandas as pd

#change numpy array into excel
def np_excel(sheet, path):
    df = pd.DataFrame(sheet)
    df.to_excel(path)

def load_excel_np(path):
    df = pd.read_excel(path)
    df = df.values
    return df