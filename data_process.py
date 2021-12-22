import json
s = open('output.txt').read().split('\n')
data = {'dstar':{},'dstaropt':{}}
data['dstar']['bottomright'] = {'mazes':[],'shape':[],'time':[],'actions':[],'obstacles':[],'nodes':[]}
data['dstar']['topleft'] = {'mazes':[],'shape':[],'time':[],'actions':[],'obstacles':[],'nodes':[]}
data['dstar']['topright'] = {'mazes':[],'shape':[],'time':[],'actions':[],'obstacles':[],'nodes':[]}
data['dstaropt']['bottomright'] = {'mazes':[],'shape':[],'time':[],'actions':[],'obstacles':[],'nodes':[]}
data['dstaropt']['topleft'] = {'mazes':[],'shape':[],'time':[],'actions':[],'obstacles':[],'nodes':[]}
data['dstaropt']['topright'] = {'mazes':[],'shape':[],'time':[],'actions':[],'obstacles':[],'nodes':[]}

mazes = ['maze_10_bottomright.lay', 'maze_10_topleft.lay', 'maze_10_topright.lay', 'maze_20_bottomright.lay', 'maze_20_topleft.lay', 'maze_20_topright.lay', 'maze_30_bottomright.lay', 'maze_30_topleft.lay', 'maze_30_topright.lay', 'maze_40_bottomright.lay', 'maze_40_topleft.lay', 'maze_40_topright.lay', 'maze_50_bottomright.lay', 'maze_50_topleft.lay', 'maze_50_topright.lay', 'maze_60_bottomright.lay', 'maze_60_topleft.lay', 'maze_60_topright.lay', 'maze_70_bottomright.lay', 'maze_70_topleft.lay', 'maze_70_topright.lay', 'maze_80_bottomright.lay', 'maze_80_topleft.lay', 'maze_80_topright.lay', 'maze_90_bottomright.lay', 'maze_90_topleft.lay', 'maze_90_topright.lay', 'maze_100_bottomright.lay', 'maze_100_topleft.lay', 'maze_100_topright.lay']
i = 0
maze_id = 0
while (i<len(s)):
    line = s[i]
    if (line.startswith('===')):
        k = {}
        maze = mazes[maze_id]
        position = maze.split('.')[0].split('_')[-1]
        if 'opt' in line:
            k = data['dstaropt'][position]
        else:
            k = data['dstar'][position]
        i+=1
        line = s[i]
        print(line)
        print(maze_id)
        line = line.split(' ')
        shape = int(line[0])-1
        time = float(line[1].split(':')[-1])
        actions = int(line[2])
        obstacles = int(line[3])
        k['mazes'].append(maze)
        k['shape'].append(shape)
        k['time'].append(time)
        k['actions'].append(actions)
        k['obstacles'].append(obstacles)
        i+=2
        line = s[i]
        nodes = int(line.split(':')[-1])
        k['nodes'].append(nodes)
        maze_id+=1
        if(maze_id == 30):
            maze_id = 0
    i+=1

json.dump(data,open('data.json','w'))