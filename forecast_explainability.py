import ast
import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import random

from safetensors.torch import load_file
from net import gtnet
import torch.nn as nn
from config import NET_ROOT

pyplot.rcParams['savefig.dpi'] = 1200
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#added with explainability
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#explainability
def visualise_saliency_maps(saliency_maps):

    for i in range(len(saliency_maps)):               
        #normalise
        saliency_maps[i] = (saliency_maps[i] - saliency_maps[i].min()) / (saliency_maps[i].max() - saliency_maps[i].min())

        saliency_maps[i] = saliency_maps[i].detach().cpu().numpy()
        #reshape
        saliency_maps[i] = saliency_maps[i].reshape(saliency_maps[i].shape[1],saliency_maps[i].shape[2])

    print('Saliency Maps')
    print(saliency_maps[0])
    print(saliency_maps[0].shape)


    # Define the number of rows and columns in the grid
    num_rows = 6
    num_columns = 6


    # Create a figure and a grid of subplots
    fig, axes = pyplot.subplots(num_rows, num_columns, figsize=(12, 12))

    # Flatten the axes array for easier iteration
    axes = axes.ravel()

    images=[]
    # Iterate through the saliency maps and plot them in subplots
    for i in range(min(len(saliency_maps), num_rows * num_columns)):
        im=axes[i].imshow(saliency_maps[i], cmap='Reds', interpolation='nearest')
        images.append(im)
        axes[i].set_title('t+'+str(i+1)+' forecast')  # Set a title for each submap
        axes[i].axis('off')


    # Hide any remaining empty subplots
    for i in range(len(saliency_maps), num_rows * num_columns):
        fig.delaxes(axes[i])

    # Add a dummy image for colorbar creation
    cax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # Adjust the position and size of the colorbar
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Intensity')  # Label for the colorbar
    # Move the colorbar label to the right
    cbar.ax.xaxis.set_label_coords(0.5, -0.3)

    # Adjust spacing between subplots
    pyplot.tight_layout()

    # Show the entire figure
    pyplot.show()









#explainability
def visualise_saliency_map(saliency_map,node):

    saliency_map = saliency_map.reshape(saliency_map.shape[1],saliency_map.shape[2])

    # #normalise each row separately
    # for i in range(len(saliency_map)):
    #     saliency_map[i] = (saliency_map[i] - saliency_map[i].min()) / (saliency_map[i].max() - saliency_map[i].min())
    
    #normalise
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    saliency_map = saliency_map.detach().cpu().numpy()


    print('Saliency Map')
    print(saliency_map)
    print(saliency_map.shape)


    # Visualize the saliency map using a heatmap
    pyplot.figure(figsize=(8, 8))
    pyplot.imshow(saliency_map, cmap='YlGnBu', interpolation='nearest',aspect='auto')
    pyplot.colorbar()
    pyplot.title(consistent_name(node),fontsize=20)
    pyplot.xlabel('Past 10 timesteps',fontsize=20)
    pyplot.ylabel('Future 36 timesteps',fontsize=20)
    x_ticks = ['N-'+str(i) for i in range(10, 0,-1)]  # X-axis tick labels
    y_ticks = ['N+'+str(i) for i in range(1, 37)]  # Y-axis tick labels
    pyplot.xticks(range(10), x_ticks)  # Set x-axis tick locations and labels
    pyplot.yticks(range(36), y_ticks)  # Set y-axis tick locations and labels

    print(consistent_name(node)+'_SM.pdf')
    pyplot.savefig(consistent_name(node)+'_SM.pdf', format='pdf', bbox_inches="tight")
    pyplot.close()
    # pyplot.show()


def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def consistent_name(name):

    name=name.replace('-ALL','').replace('Mentions-','').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
    
    #special case
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

#returns the closest curve cc to a given curve c in a list of forecasted curves, where cc is strictly larger than c
def getClosestCurveLarger(c,forecast,confidence, attack, solutions,col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in solutions and not col[j]==attack: #exclude irrelevant curves
            continue 
        if torch.mean(f) <= torch.mean(c):
            continue #must be larger
        if torch.mean(f)-torch.mean(c)<d:
            d=torch.mean(f)-torch.mean(c)
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf

#returns closest curve cc to given curve c in a list of forecasted curves, where cc is strictly smaller than c
def getClosestCurveSmaller(c,forecast,confidence,attack, solutions, col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in solutions and not col[j]==attack: #exclude irrelevant curves
            continue 
        if torch.mean(f) >= torch.mean(c):
            continue #must be smaller
        if torch.abs(torch.mean(f)-torch.mean(c))<d:
            d=torch.abs(torch.mean(f)-torch.mean(c))
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf


def zero_negative_curves(data, forecast, attack, solutions):
    a = data[:, index[attack]]
    f= forecast[:,index[attack]]
    for i in range(a.shape[0]):
        if a[i]<0:
            a[i]=0
    for i in range(f.shape[0]):
        if f[i]<0:
            f[i]=0

    for s in solutions:
        a = data[:, index[s]]
        f= forecast[:,index[s]]
        for i in range(a.shape[0]):
            if a[i]<0:
                a[i]=0
        for i in range(f.shape[0]):
            if f[i]<0:
                f[i]=0
    return data, forecast
           

# def no_curve_in_between(x, y, curves, attack, solutions, point):
#     for j in range(curves.shape[1]):
#         if not col[j] == attack and not col[j] in solutions:
#             continue
#         c = curves[point, j]
#         if torch.equal(x, c) or torch.equal(y, c):
#             continue
#         if (c > x) & (c < y):
#             return False
#         if (c < x) & (c > y):
#             return False
#     return True
        

#plots forecast of attack and relevant solutions trends. If alarming is set to True, plots the solutions trend forecasted to be less than the attack trend.
def plot_forecast(data,forecast,confidence,attack,solutions,index,col,alarming=True):

    data,forecast= zero_negative_curves(data, forecast, attack, solutions)
    
    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
          "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
          "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
          "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
          "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
          "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]


    # rest_colours=colours[1:]
    # random.shuffle(rest_colours)
    # colours=[colours[0]]+rest_colours

    
    pyplot.style.use("seaborn-dark") 
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])


    #Plot the forecast of attack
    counter=0
    d=torch.cat((data[:,index[attack]],forecast[0:1,index[attack]]),dim=0)#connect the past to future in the plot
    f=forecast[:,index[attack]]
    c=confidence[:,index[attack]]
    a=consistent_name(attack)
    ax.plot(range(len(d)),d,'-', color=colours[counter],label=a,linewidth = 2)
    ax.plot(range(len(d)-1, (len(d)+len(f))-1),f,'-', color=colours[counter],linewidth=2)
    ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),f - c, f + c, color=colours[counter], alpha=0.6)
    f_attack=f.clone()
    counter+=1

    #remove technologies that we are not worried about in the future
    if alarming:
        for s in list(solutions):
            f=forecast[:,index[s]]
            if torch.mean(f)>= torch.mean(f_attack): #solution with higher trend in the future
                solutions.remove(s)

    #Plot the forecast of the solutions
    for s in solutions:
        d=torch.cat((data[:,index[s]],forecast[0:1,index[s]]),dim=0)#connect the past to future in the plot
        f=forecast[:,index[s]]
        c=confidence[:,index[s]]
        s=consistent_name(s)
        ax.plot(range(len(d)),d,'-', color=colours[counter],label=s,linewidth = 1)
        ax.plot(range(len(d)-1, (len(d)+len(f))-1),f,'-', color=colours[counter],linewidth=1)
        ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),f - c, f + c, color=colours[counter], alpha=0.6)
        if torch.mean(f_attack) > torch.mean(f):
            cc,cc_conf=getClosestCurveLarger(f,forecast,confidence,attack, solutions, col)#prevents overlap in the area above the curve
            ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),cc-cc_conf, f+c, color=colours[counter], alpha=0.3)
        else:
            cc,cc_conf=getClosestCurveSmaller(f,forecast,confidence,attack,solutions,col)#prevents overlap in the area under the curve
            ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),cc+cc_conf, f-c,  color=colours[counter], alpha=0.3)
            #ax.fill_between(range(len(d)-1, (len(d)+len(f))-1),cc+cc_conf, f-c, where=[no_curve_in_between(x_val,y_val,forecast,attack,solutions, point) for point,(x_val, y_val) in enumerate(zip(cc+cc_conf, f-c))], color=colours[counter], alpha=0.3)

        counter+=1  
    
    x=['2012', '2013','2014', '2015', '2016', '2017', '2018','2019', '2020', '2021','2022','2023','2024','2025','2026']
    ax.set_xticks([6,18,30,42,54,66,78,90,102,114,126,138,150,162,174], x) # positions of years on x axis

    #ax.axvspan(138, 173, color="skyblue", alpha=0.6,  label="Forecast Period")
    ax.set_ylabel("Trend",fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left",prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90,fontsize=13)
    pyplot.title(a, y=1.03,fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7) 

    #save and show the forecast
    images_dir = f'{NET_ROOT}/model/Bayesian/forecast/plots/'
    pyplot.savefig(images_dir+a.replace('/','_')+'.png', bbox_inches="tight")
    pyplot.savefig(images_dir+a.replace('/','_')+".pdf", bbox_inches = "tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()



def save_data(data, forecast, confidence, variance, col):
    # write the data and forecast
    for i in range(data.shape[1]):
        d= data[:,i]
        f= forecast[:,i]
        c=confidence[:,i]
        v=variance[:,i]
        name=col[i]
        file_dir = f'{NET_ROOT}/model/Bayesian/forecast/data/'
        with open(file_dir+name.replace('/','_')+'.txt', 'w') as ff:
            ff.write('Data: '+str(d.tolist())+'\n')
            ff.write('Forecast: '+str(f.tolist())+'\n')
            ff.write('95% Confidence: '+str(c.tolist())+'\n')
            ff.write('Variance: '+str(v.tolist())+'\n')
    ff.close()

#saves the forecasted trend's gap between attack and its relevant solutions to a csv file. The gap is for 3 years resulting in 3 values per solution.
def save_gap(forecast, attack, solutions,index):
    # write the data and forecast
    with open(f'{NET_ROOT}/model/Bayesian/forecast/gap/'+consistent_name(attack).replace('/','_')+'_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the list as a row
        writer.writerow(['Solution','2023','2024','2025'])
        table=[]
        a=forecast[:,index[attack]].tolist()
        a_reduced= [sum(a[i:i+12]) / 12 for i in range(0, len(a), 12)]#mean of every 12 months
        for s in solutions:
            row=[consistent_name(s)]
            f=forecast[:,index[s]].tolist()
            f_reduced= [sum(f[i:i+12]) / 12 for i in range(0, len(f), 12)]#mean of every 12 months
            
            gap=[x - y for x, y in zip(a_reduced, f_reduced)]#calculate the gap
            row.extend(gap)#3 years gap
            table.append(row)
        sorted_table = sorted(table, key=lambda row: sum(row[-3:]))
        for row in sorted_table:
            writer.writerow(row)
    





#given data file, returns the list of column names and dictionary of the format (column name,column index)
def create_columns(file_name):

    col_name=[]
    col_index={}

    # Read the CSV file of the dataset
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Read the first row
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name= col_name[1:]
        
        for i,c in enumerate(col_name):
            col_index[c]=i
        
        return col_name,col_index


#builds the attacks to solutions graph
def build_graph(file_name):
    # Initialize an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    # Read the graph CSV file
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the key node from the first column
            key_node = row[0]
            # Extract the adjacent nodes from the remaining columns
            adjacent_nodes =  [node for node in row[1:] if node]#does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'attacks...')
    return graph




#This script forecasts the future of the attacks-solutions graph, up to 3 years in advance

#added with explainability
fixed_seed = 123
set_random_seed(fixed_seed)

data_file = f'{NET_ROOT}/data/Smoothed_CyberTrend_Forecasting_All.txt'
model_file = f'{NET_ROOT}/model/Bayesian/model.safetensors'
nodes_file = f'{NET_ROOT}/data/Smoothed_CyberTrend_Forecasting_All.csv'
graph_file = f'{NET_ROOT}/data/graph.csv'
hp_path = f"{NET_ROOT}/model/Bayesian/hp.txt"

#read the data
fin = open(data_file)
rawdat = np.loadtxt(fin, delimiter='\t')
n, m = rawdat.shape

#load column names and dictionary of (column name, index)
col,index=create_columns(nodes_file)


#build the graph in the format {attack:list of solutions}
graph=build_graph(graph_file)

#for normalisation
scale = np.ones(m)
dat = np.zeros(rawdat.shape)

#normalise
for i in range(m):
    scale[i] = np.max(np.abs(rawdat[:, i]))
    dat[:, i] = rawdat[:, i] / np.max(np.abs(rawdat[:, i]))

scale = torch.from_numpy(scale).to(device=device)

print('data shape:',dat.shape)

#preparing last part of the data to be used for the forecast
P=10
X= torch.from_numpy(dat[-P:, :]) #look back 10 months
X = torch.unsqueeze(X,dim=0)
X = torch.unsqueeze(X,dim=1)
X = X.transpose(2,3)
X = X.to(dtype=torch.float, device=device)

X.requires_grad = True #explainability
dat = torch.from_numpy(dat).to(device=device)

with open(hp_path, "r") as f:
    best_hp = ast.literal_eval(f.read())

(gcn_depth, _lr, conv, res, skip, end,
 k, dropout, dilation_ex, node_dim,
 prop_alpha, tanh_alpha, layer, _) = best_hp

#load the model
# ✅ 동일 아키텍처로 모델 인스턴스 생성 (훈련 시 하이퍼파라미터와 일치해야 함)
model = gtnet(
    gcn_true=True,
    buildA_true=True,
    gcn_depth=gcn_depth,
    num_nodes=m,
    device=device,
    predefined_A=None,
    static_feat=None,
    dropout=dropout,
    subgraph_size=k,
    node_dim=node_dim,
    dilation_exponential=dilation_ex,
    conv_channels=conv,
    residual_channels=res,
    skip_channels=skip,
    end_channels=end,
    seq_length=P,          # 학습의 seq_in_len과 동일해야 함(기본 10)
    in_dim=1,
    out_dim=36,            # 학습의 seq_out_len과 동일
    layers=layer,
    propalpha=prop_alpha,
    tanhalpha=tanh_alpha,
    layer_norm_affline=False
).to(device)

state = load_file(model_file, device=device)
model.load_state_dict(state)
model.eval()

# Bayesian estimation
num_runs = 10

# Create a list to store the outputs
outputs = []

# Use model to predict next time step
for _ in range(num_runs):
    with torch.no_grad():
        output = model(X)  
        y_pred = output[-1, :, :,-1].clone()#10x142
    outputs.append(y_pred)



# Stack the outputs along a new dimension
outputs = torch.stack(outputs)


Y=torch.mean(outputs,dim=0)
variance = torch.var(outputs, dim=0)#variance
variance.to(device=device)

std_dev = torch.std(outputs, dim=0)#standard deviation
# Calculate 95% confidence interval
z=1.96
confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))
confidence.to(device=device)

dat*=scale
Y*=scale
variance*=scale
confidence*=scale

print('output shape:',Y.shape)




#Explainability
# Let's say you have a particular layer's attention scores, e.g., the first layer
# attention_scores_layer1 = attention_scores[0]

# # Get a specific sample or sequence for visualization
# sample_index = 0  # Change this to the desired sample/sequence index

# # Plot the attention scores
# pyplot.figure(figsize=(12, 4))
# pyplot.imshow(attention_scores_layer1[sample_index].cpu().numpy(), cmap='viridis', aspect='auto')
# pyplot.title(f'Attention Scores for Layer 1 - Sample {sample_index}')
# pyplot.xlabel('Nodes')
# pyplot.ylabel('Nodes')
# pyplot.colorbar()
# pyplot.show()




#from util import DataLoaderS
#Data = DataLoaderS('data/data.txt')
# node_idx=79
# print('output:',(Y/scale)[0,node_idx].item())
# model.explain_by_adjacency(79, X[0,0,:,9])

time_saliency_maps=[]
saliency_map_36=None
node=3
for t in range(0,36):
    saliency_map = model.compute_saliency(X,t,node,True) #Explainability, set second argument to True if you want timestep importance
    time_saliency_maps.append(saliency_map)
    #2d
    if saliency_map_36==None:
        saliency_map_36=saliency_map
    else:
        saliency_map_36 = torch.cat((saliency_map_36, saliency_map), dim=1)#2d


#visualise_saliency_maps(time_saliency_maps)
visualise_saliency_map(saliency_map_36,col[node])
sys.exit()


#----------------------------------------------------------------------------------------------------#

#Plotting:



#save the data to desk
dat=torch.from_numpy(dat)
save_data(dat,Y,confidence,variance,col)

#combine data
all=torch.cat((dat,Y), dim=0)


#scale down full data (global normalisation)
incident_max=-999999999
mention_max=-999999999

for i in range(all.shape[0]):
    for j in range(all.shape[1]):
        if 'WAR' in col[j] or 'Holiday' in col[j] or j in range(16,32):
            continue
        if 'Mention' in col[j]:
            if all[i,j]>mention_max:
                mention_max=all[i,j]
        else:
            if all[i,j]>incident_max:
                incident_max=all[i,j]

# incident_max=1 #1672
# mention_max=1 #2028
# incident_max=1
# mention_max=1
#print(mention_max, incident_max)
all_n=torch.zeros(all.shape[0],all.shape[1])
confidence_n=torch.zeros(confidence.shape[0],confidence.shape[1])
u=0
for i in range(all.shape[0]):
    for j in range(all.shape[1]):
            if 'Mention' in col[j]:
                all_n[i,j]=all[i,j]/mention_max
            else:
                all_n[i,j]=all[i,j]/incident_max
            
            if i>=all.shape[0]-36:
                confidence_n[u,j]=confidence[u,j]*(all_n[i,j]/all[i,j])
    if i>=all.shape[0]-36:
        u+=1

# max_col,max_indecies=torch.max(all,dim=0)
# min_col,min_indecies=torch.min(all,dim=0)
# full_dat_scaled=torch.ones(full_dat.shape[0],full_dat.shape[1])
# for j in range(full_dat.shape[1]):
#     full_dat_scaled[:,j] = full_dat[:,j]-min_col[j]
#     full_dat_scaled[:,j]=full_dat_scaled[:,j]/(max_col[j]-min_col[j])
# for j in range(confidence.shape[1]):
#     confidence[:,j] *= full_dat_scaled[-36:,j]/full_dat[-36:,j]
#     for i in range(confidence.shape[0]):
#         if confidence[i,j]>0.3:
#             confidence[i,j]=0.3 #limit confidence for visualisation

#smoothing
smoothed_dat=torch.stack(exponential_smoothing(all_n, 0.1))
smoothed_confidence=torch.stack(exponential_smoothing(confidence_n, 0.1))





#plot all forecasted nodes in the graph using each attack to its solutions list
for attack, solutions in graph.items():
    plot_forecast(smoothed_dat[:-36,],smoothed_dat[-36:,], smoothed_confidence,attack,solutions,index,col)
    save_gap(smoothed_dat[-36:,],attack,solutions,index)
    









