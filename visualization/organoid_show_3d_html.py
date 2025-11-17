### The major of this script was written by Modi Safra


import numpy as np
from skimage import io
import pandas as pd
import sys
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

file = r'C:\Users\Moshe\OneDrive\coding\SpatialGenomicsLab\SegmentedData\Organoids\filtered_organoids\001STXBP1_A2_left\\FOV1FOV_1_round001_ch00.tif.cells.tif'
typeS = r'C:\Users\Moshe\OneDrive\coding\SpatialGenomicsLab\SegmentedData\Organoids\filtered_organoids\\001STXBP1_A2_left\filtered_organoids_001STXBP1_A2_left_profile_table.csv'

try:
    file = sys.argv[1]
    typeS = sys.argv[2]
except:
    pass

cells = io.imread(file)
cells = cells[::2, ::2, ::2]
types = pd.read_csv(typeS)
types['t'] = types.cell_id.str.replace('_', '               ').str[:10].str.replace(' ', '').astype(int)
t = types[types.duplicated('t') == False]
x = pd.DataFrame(np.argwhere(cells))
x.columns = ['z', 'x', 'y']
x['cell'] = cells[cells != 0]
x = x[x.cell != 0]
x.x *= 16 * 0.17 / 3.3
x.y *= 16 * 0.17 / 3.3
x.z *= 16 * 0.4 / 3.3

cl = ['red', 'blue', 'white', 'green', 'orange', 'pink', 'purple', 'silver', 'lightgreen']
q = ''
if not 'color' in t.columns:
    j = 0
    t['color'] = ''
    for i in np.unique(t.cell_type):
        t['color'] = np.where(t.cell_type == i, cl[j], t.color)
        q += '<span style="color:' + cl[j] + '">' + i + '   </span>'
        j += 1

a = []
j = 0
d = []
for i in np.unique(t.t):
    z = x[x.cell == i].iloc[:, :3]
    z.z += [0.1] + [0] * (z.shape[0] - 1)
    if z.shape[0] > 1:
        h = ConvexHull(z)
        z = z.iloc[np.unique(h.simplices.reshape(h.simplices.size))]
        d.append(go.Mesh3d(x=z.x, y=z.y, z=z.z, alphahull=0, color=str(t[t.t == i].color.iloc[0])
                           , hoverinfo='skip'))
        a.append(j)
    j += 1
f = go.Figure(data=d)

f.write_html(file + '.html', div_id='1', include_plotlyjs=False)

l = ''.join(open(file + '.html').readlines()).split('<div id="1"')

z = "var v={'cell-type':" + str(list(t.color)) + ',' + 'None:' + str([0] * len(d)) + ','
s0 = '<option value="cell-type">cell-type</option>'
s = '<option value="None">None</option>'
for i in t.columns[4:-3]:
    y = np.array(list(t[i].iloc[a]))
    y = (y / np.max(y) * 255).astype(int)
    z += "'" + i + "':" + '[' + ','.join(y.astype(str)) + ']' + ','
    s += '<option value="' + i + '">' + i + '</option>'
z = z[:-1] + '};\n'

l[0] += """ 
<select id="r" onchange="getComboA(this);" style="color:red;display:block">""" + s0 + s + """'</select>
<select id="g" onchange="getComboA(this);" style="color:green;display:none">""" + s + """'</select>
<div id='types' style='display:block;'>
""" + q + """
</div>

<script>

"""
l = l[0] + z + """
var R,G,B;
var cells=""" + str(list(x.cell)) + """;\n
 function getComboA(selectObject) {
     R = document.getElementById('r').value;  
     G = document.getElementById('g').value;  
     if(R=='cell-type'){
       Plotly.restyle('1',{'color':v['cell-type']} ,""" + str(list(range(len(d)))) + """ );
       document.getElementById('types').style.display='block';
       document.getElementById('g').style.display='none';
     }
     else{
         document.getElementById('types').style.display='none';
         document.getElementById('g').style.display='block';
        c=[];
        for(i in v[R]){
          // c.push('rgb('+v[R][i]+','+v[G][i]+','+v[B][i]+')');
          b=255-Math.max(v[R][i],v[G][i]);
           c.push('rgb('+(v[R][i]+b)+','+(v[G][i]+b)+','+b+')');
         }
        Plotly.restyle('1',{'color':c} ,""" + str(list(range(len(d)))) + """ );
    }

}
</script>
<div id="1" """ + l[1]

l = l.replace('<head>', "<head><script src='https://cdn.plot.ly/plotly-3.0.1.min.js'></script>").replace(
    "<body>", "<body align='center'>")

w = open(file + '.html', 'w')
w.write(l)
w.close()

