# lifelong-planning-pacman 

search.py has dstar lite, dstar lite optimized, astar lifelong implemented.

<!-- Use --no-graphics flag to run without the graphics -->


# Command to run dstar:
python pacman.py -l tinyMaze -z .25 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic

# Command to run lifelong dstar optimized
python pacman.py -l tinyMaze -z 0.25 -p SearchAgent -a fn=dstaropt,heuristic=manhattanHeuristic

# Command to run lifelong astar
python pacman.py -l tinyMaze -z 0.25 -p SearchAgent -a fn=lastar,prob=AStarPositionSearchProblem,heuristic=manhattanHeuristic

# Mazes
These are mazes provided in the project modified to suit are requirement.

openMaze.lay\
tinyMaze.lay\
smallMaze.lay\
mediumMaze.lay\
bigMaze.lay

bigCornersModified.lay\
bigMazeReverseTopRight.lay\
bigSearchModified.lay\
contoursMaze.lay\
largeMaze.lay\
largeMazeTopRight.lay\
large_100_bottomright.lay\
large_100_topleft.lay\
mediumDottedMaze.lay\
smallSafeSearch.lay\
smallSearchModified.lay\
testMaze.lay\
testSearch.lay\
trickyClassicModified.lay\
trickySearchModified.lay\

# Generated mazes
maze_x0_position.lay\
x0 --> size of maze x0*x0\
position --> position of pacman

List of generated mazes and expanded nodes of D* lite and D* lite optimizes.

| Maze                     | D* lite     | D* lite optimized     |
| :---                     |    :----:   |        ---: |
| maze_10_bottomright.lay  | 780         | 627         |
| maze_10_topleft.lay      | 137         | 120         |
| maze_10_topright.lay     | 723         | 633         |
| maze_20_bottomright.lay  | 967         | 874         |
| maze_20_topleft.lay      | 1240        | 1127        |
| maze_20_topright.lay     | 3727        | 3512        |
| maze_30_bottomright.lay  | 4536        | 3946        |
| maze_30_topleft.lay      | 2673        | 2379        |
| maze_30_topright.lay     | 7978        | 7286        |
| maze_40_bottomright.lay  | 4042        | 3702        |
| maze_40_topleft.lay      | 13688       | 12155       |
| maze_40_topright.lay     | 18294       | 17023       |
| maze_50_bottomright.lay  | 6291        | 5510        |
| maze_50_topleft.lay      | 5566        | 5132        |
| maze_50_topright.lay     | 19863       | 18573       |
| maze_60_bottomright.lay  | 8958        | 8316        |
| maze_60_topleft.lay      | 18726       | 17366       |
| maze_60_topright.lay     | 22291       | 21856       |
| maze_70_bottomright.lay  | 23288       | 21028       |
| maze_70_topleft.lay      | 36464       | 33665       |
| maze_70_topright.lay     | 30130       | 29153       |
| maze_80_bottomright.lay  | 41828       | 38250       |
| maze_80_topleft.lay      | 57129       | 52052       |
| maze_80_topright.lay     | 82863       | 77330       |
| maze_90_bottomright.lay  | 11617       | 10394       |
| maze_90_topleft.lay      | 105106      | 96849       |
| maze_90_topright.lay     | 101643      | 94971       |
| maze_100_bottomright.lay | 73186       | 66171       |
| maze_100_topleft.lay     | 55074       | 51427       |
| maze_100_topright.lay    | 155629      | 143945      |