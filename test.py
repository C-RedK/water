import pandas as pd
import numpy as np

a = [[1,2,3,4,5,6],[4,5,6,7,8,9],[7,8,9,10,11,12]]
anp = np.array(a)
anp = anp.transpose()

df = pd.DataFrame(anp)
df.to_excel("haha.xlsx")
print(df)