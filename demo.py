import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import plotly.graph_objs as go
import networkx as nx
import webbrowser

# 1. Carga del dataset Cora
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# 2. Definición del modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. Preparación de dispositivo, modelo y optimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 4. Entrenamiento del modelo
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 5. Evaluación y precisión primeros 100 nodos
model.eval()
with torch.no_grad():
    logits = model(data)
    pred = logits.argmax(dim=1)
correct_100 = int((pred[:100] == data.y[:100]).sum())
acc_100 = correct_100 / 100
acc_100_str = f"{acc_100*100:.2f}%"
print(f"Precisión (primeros 100 nodos): {acc_100_str}")

# 6. Visualización solo de los primeros 100 nodos
def plot_interactive_first_100(data, labels, title, filename):
    nodes_to_keep = set(range(100))
    edge_index = data.edge_index.cpu().numpy()
    mask = [
        i for i in range(edge_index.shape[1])
        if edge_index[0, i] in nodes_to_keep and edge_index[1, i] in nodes_to_keep
    ]
    filtered_edges = edge_index[:, mask]
    G = nx.Graph()
    G.add_edges_from(filtered_edges.T)
    for node in nodes_to_keep:
        G.add_node(node)
    pos = nx.spring_layout(G, seed=42, k=0.5)

    node_x, node_y, node_color = [], [], []
    for node in sorted(G.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(int(labels[node].cpu().numpy()))
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Rainbow',
            color=node_color,
            size=16,
            colorbar=dict(
                thickness=15,
                title=dict(text='Clase de nodo', side='right'),
                xanchor='left'
            ),
            line_width=2
        ),
        text=[f'Node {n}, Clase {c}' for n, c in zip(sorted(G.nodes()), node_color)]
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(text=title, font=dict(size=16)), # <--- AQUÍ CAMBIA
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text=f"Precisión: {acc_100_str}" if "Predicción" in title else "",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.5, y=-0.1, align="center"
                       ) ],
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False))
                   )
    fig.write_html(filename)
    print(f"Gráfico interactivo guardado en {filename}")

plot_interactive_first_100(data, pred, "Clasificación de los primeros 100 nodos (Predicción GCN)", "prediccion_gcn_100.html")
plot_interactive_first_100(data, data.y, "Clasificación real de los primeros 100 nodos", "real_gcn_100.html")

# 7. Leer el HTML base y reemplazar el marcador de precisión
with open("resultado_gcn_interactivo.html", "r", encoding="utf-8") as f:
    html_base = f.read()
html_final = html_base.replace("{accuracy}", acc_100_str)
with open("resultado_gcn_interactivo_final.html", "w", encoding="utf-8") as f:
    f.write(html_final)
print("Archivo resultado_gcn_interactivo_final.html creado.")

# 8. Abre el HTML resultado en el navegador
ruta_html = os.path.abspath("resultado_gcn_interactivo_final.html")
webbrowser.open(f"file://{ruta_html}")