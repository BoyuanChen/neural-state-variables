import os
import shutil
import json
import random
import numpy as np
from PIL import Image
import plotly.graph_objects as go


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def create_random_data_splits(seed, data_filepath, object_name, ratio=0.8):
    random.seed(seed)
    seq_dict = {}

    obj_filepath = os.path.join(data_filepath, object_name)
    num_vids = int(len(os.listdir(obj_filepath)) - 1)
    vid_id_lst = list(range(num_vids))
    random.shuffle(vid_id_lst)

    # test
    start = int(num_vids * (ratio + (1 - ratio) / 2))
    seq_dict['test'] = vid_id_lst[start:]
    # val
    start = int(num_vids * ratio)
    end = int(num_vids * (ratio + (1 - ratio) / 2))
    seq_dict['val'] = vid_id_lst[start:end]
    # train
    seq_dict['train'] = vid_id_lst[:int(num_vids * ratio)]

    with open(os.path.join(f'./datainfo/{object_name}/data_split_dict_{seed}.json'), 'w') as file:
        json.dump(seq_dict, file, indent=4)


def separate_eval_image(filepath):
    img = Image.open(filepath)
    img.crop((2, 2, 130, 130)).save('input_0.png')
    img.crop((2, 132, 130, 260)).save('input_1.png')
    img.crop((2, 262, 130, 390)).save('truth_0.png')
    img.crop((2, 392, 130, 520)).save('truth_1.png')
    img.crop((2, 522, 130, 650)).save('output_0.png')
    img.crop((2, 652, 130, 780)).save('output_1.png')


def collect_long_term_pred_sample(dataset, seed, test_vid_id, hybrid_step, start=0, step=12, stop=60, suf='png'):
    idv = id_value[dataset]
    fps = fps_value[dataset]
    save_path = f'long_term_pred_sample_{dataset}_seed_{seed}_vid_{test_vid_id}' 
    mkdir(save_path)
    
    data_filepath = f'/data/kuang/visphysics_data/{dataset}/{test_vid_id}'
    pred_filepaths = {'dim-8192':f'/data/kuang/logs/logs_{dataset}_encoder-decoder_{seed}/prediction_long_term/model_rollout/{test_vid_id}',
                      'dim-64':f'/data/kuang/logs/logs_{dataset}_encoder-decoder-64_{seed}/prediction_long_term/model_rollout/{test_vid_id}',
                      f'dim-{idv}':f'/data/kuang/logs/logs_{dataset}_refine-64_{seed}/prediction_long_term/model_rollout/{test_vid_id}',
                      f'hybrid-64-{idv}':f'/data/kuang/logs/logs_{dataset}_refine-64_{seed}/prediction_long_term/hybrid_rollout_{hybrid_step}/{test_vid_id}'}
    
    Image.open(os.path.join(data_filepath, str(start)+'.'+suf)).save(os.path.join(save_path, 'input_0.png'))
    Image.open(os.path.join(data_filepath, str(start+1)+'.'+suf)).save(os.path.join(save_path, 'input_1.png'))
    
    for k in range(start+step, stop+1, step):
        data = Image.open(os.path.join(data_filepath, str(k-1)+'.'+suf))
        data.save(os.path.join(save_path, 'truth_%.1fs.png'%((k-start)/fps)))
        for scheme in ['dim-8192', 'dim-64', f'dim-{idv}', f'hybrid-64-{idv}']:
            pred = Image.open(os.path.join(pred_filepaths[scheme], str(k-1)+'.'+suf))
            pred.save(os.path.join(save_path, scheme+'_%.1fs.png'%((k-start)/fps)))


"""
    physical system parameters
"""
id_value = {'circular_motion':2, 'reaction_diffusion':2, 'single_pendulum':2,
            'double_pendulum':4, 'swingstick_non_magnetic':4, 'elastic_pendulum':6,
            'air_dancer':8, 'lava_lamp':8, 'fire':24}
fps_value = {'circular_motion':60, 'reaction_diffusion':5, 'single_pendulum':60,
             'double_pendulum':60, 'swingstick_non_magnetic':60, 'elastic_pendulum':60,
             'air_dancer':60, 'lava_lamp':2.5, 'fire':24}


"""
    plot style
"""
cols = ['#df1e1e', '#f0975a', '#1b66cb', '#149650']

colorscale=[[0.0, "rgb(49,54,149)"],
            [0.1111111111111111, "rgb(69,117,180)"],
            [0.2222222222222222, "rgb(116,173,209)"],
            [0.3333333333333333, "rgb(171,217,233)"],
            [0.4444444444444444, "rgb(224,243,248)"],
            [0.5555555555555556, "rgb(254,224,144)"],
            [0.6666666666666666, "rgb(253,174,97)"],
            [0.7777777777777778, "rgb(244,109,67)"],
            [0.8888888888888888, "rgb(215,48,39)"],
            [1.0, "rgb(165,0,38)"]]

def light_color(col):
    if col[:3] == 'rgb':
        rgb = cols[n][4:-1]
    elif col[0] == '#':
        rgb = str(tuple(int(col.lstrip('#')[i:i+2],16) for i in (0,2,4)))[1:-1]
    return 'rgba('+rgb +',0.2)'

def update_figure(fig):
    fig.update_xaxes(showgrid=False, tickfont=dict(family="sans serif", size=18, color="black"))
    fig.update_yaxes(showgrid=False, tickfont=dict(family="sans serif", size=18, color="black"))
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black")
    fig.update_layout(legend=go.layout.Legend(traceorder="normal",
                                              font=dict(family="sans serif", size=18, color="black"),
                    ))
    fig.update_layout(font=dict(family="sans serif", size=24, color="black"),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                     )

def test():
    fig = go.Figure()
    x = np.arange(30)
    for k in range(4):
        fig.add_trace(go.Scatter(x=x, y=x*(k+1), mode='lines', line=dict(width=4, color=cols[k])))
    update_figure(fig)
    fig.write_image('test.png', scale=4)


"""
test code only
"""
if __name__ == '__main__':
    test()