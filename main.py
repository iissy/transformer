import torch
import pandas as pd
import numpy as np
from position import PositionEncoding
import plotly.express as px

pe = PositionEncoding(120, 0)
z = pe.forward(torch.zeros(1, 100, 120))
df = pd.DataFrame(z[0, :, [0 ,20 ,60 ,110]].data.numpy() ,columns = ["dim " +c for c in ['0' ,'20' ,'60' ,'110']])
df.insert(0 ,"x" ,np.arange(100))
px.line(df, x = "x" ,y = ["dim " +c for c in ['0' ,'20' ,'60' ,'110']]).show()
px.imshow(np.squeeze(z.data.numpy()) ,color_continuous_scale="blues",width=1000,height=800).show()