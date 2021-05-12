import numpy as np 
import cvxpy as cp
import copy
action= ["UP", "LEFT", "DOWN", "RIGHT", "STAY", "SHOOT", "HIT", "CRAFT", "GATHER", "NONE"] 
health=[0,25,50,75,100]
pos=["C","N","E","W","S"]
arrows=[0,1,2,3]
mat=[0,1,2]
mmstate=["D","R"] #can be D or R
stepcost= -10

possibility={
    "C" : [["UP",[(0.85,"N"),(0.15,"E")]],["DOWN",[(0.85,"S"),(0.15,"E")]],["STAY",[(0.85,"C"),(0.15,"E")]],["RIGHT",[(1,"E")]],["LEFT",[(0.85,"W"),(0.15,"E")]],["SHOOT",[(0.5,25,1),(0.5,0,0)]],["HIT",[(0.1,50),(0.9,0)]]],
    "N" : [["CRAFT",[(0.5,1),(0.35,2),(0.15,3)]],["STAY",[(0.85,"N"),(0.15,"E")]],["DOWN",[(0.85,"C"),(0.15,"E")]]],
    "S" : [["GATHER",[(0.75,1),(0.25,0)]],["UP",[(0.85,"C"),(0.15,"E")]],["STAY",[(0.85,"S"),(0.15,"E")]]],
    "W" : [["SHOOT",[(0.25,25,1),(0.75,0,0)]],["RIGHT",[(1,"C")]],["STAY",[(1,"W")]]],
    "E" : [["SHOOT",[(0.9,25,1),(0.1,0,0)]],["HIT",[(0.2,50),(0.8,0)]],["LEFT",[(1,"C")]],["STAY",[(1,"E")]]],
    "D" : [[0.2,"R"],[0.8,"D"]],
    "R" : [[0.5,"D"],[0.5,"R"]] #this combination of RnD is shoot spot where you have to change ur reward function.
}


def getindex(arr):
    #print(arr, "is array passed")
    match=0
    ind=-1
    for j in juststates:
        match=0
        for i in range(len(arr)):
            if j[i]==arr[i]:
                match+=1
        if match==5:
            #print(j)
            ind=juststates.index(j)
            return ind   
    print(arr,"is not found")
    return 0


def compute(state,stateindex):  
    state_array=[]  
    state_nextstates=[]
    if state[4]==0:
        #state_action="NONE"
        rew=0 #or stepcost #didnt append these yet
        state_array=[["NONE",0]]
        
    else:
        for act in possibility[state[0]]:
            state_action=""
            rew=0    #all movement action
            state_nextstates=[]
            if act[0]=="UP" or act[0]=="DOWN" or act[0]=="RIGHT" or act[0]=="LEFT" or act[0]=="STAY" :
                state_action=act[0]   
                if state[3] == "D":         #move and D
                        for mm in possibility[state[3]]: 
                            for indi in act[1]:
                                nextstate=[indi[1],state[1],state[2],mm[1],state[4]]
                                probab=indi[0]*mm[0]
                                state_nextstates.append([nextstate,probab])
                                rew+=probab*(stepcost)

                elif state[3]=="R":
                    for mm in possibility[state[3]]: 
                        for indi in act[1]:
                            probab=indi[0]*mm[0]
                            if mm[1]=="D" and (state[0]=="E" or state[0]=="C"):
                                nextstate=[state[0],state[1],0,mm[1],min(state[4]+25,100)]
                                rew+=probab*(stepcost - 40)    #indiana got hit 
                            else:
                                nextstate=[indi[1],state[1],state[2],mm[1],state[4]]
                                rew+=probab*(stepcost)
                            state_nextstates.append([nextstate,probab])
                state_array.append([state_action,rew,state_nextstates])            
            #gather actions
            elif act[0]=="GATHER":
                state_action=act[0]   
                if state[3] == "D":         #move and D
                        for mm in possibility[state[3]]: 
                            for indi in act[1]:
                                nextstate=[state[0],min(state[1]+indi[1],2),state[2],mm[1],state[4]]
                                probab=indi[0]*mm[0]
                                state_nextstates.append([nextstate,probab])
                                rew+=probab*(stepcost)

                elif state[3]=="R":
                    for mm in possibility[state[3]]: 
                        for indi in act[1]:
                            probab=indi[0]*mm[0]
                            if mm[1]=="D" and (state[0]=="E" or state[0]=="C"):
                                nextstate=[state[0],state[1],0,mm[1],min(state[4]+25,100)]
                                rew+=probab*(stepcost - 40)    #indiana got hit 
                            else:
                                nextstate=[state[0],min(state[1]+indi[1],2),state[2],mm[1],state[4]]
                                rew+=probab*(stepcost)
                            state_nextstates.append([nextstate,probab])
                state_array.append([state_action,rew,state_nextstates])
            #craft actions
            elif act[0] == "CRAFT" and state[1]!=0:
                state_action=act[0]   
                if state[3] == "D":         #move and D
                        for mm in possibility[state[3]]: 
                            for indi in act[1]:
                                nextstate=[state[0],state[1]-1,min(state[2]+indi[1],3),mm[1],state[4]]
                                probab=indi[0]*mm[0]
                                state_nextstates.append([nextstate,probab])
                                rew+=probab*(stepcost)

                elif state[3]=="R":
                    for mm in possibility[state[3]]: 
                        for indi in act[1]:
                            probab=indi[0]*mm[0]
                            if mm[1]=="D" and (state[0]=="E" or state[0]=="C"):
                                nextstate=[state[0],state[1],0,mm[1],min(state[4]+25,100)]
                                rew+=probab*(stepcost - 40)    #indiana got hit 
                            else:
                                nextstate=[state[0],state[1]-1,min(state[2]+indi[1],3),mm[1],state[4]]
                                rew+=probab*(stepcost)
                            state_nextstates.append([nextstate,probab])
                state_array.append([state_action,rew,state_nextstates])
            #hit actions    
            elif act[0] == "HIT":
                state_action=act[0]   
                if state[3] == "D":         #move and D
                        for mm in possibility[state[3]]: 
                            for indi in act[1]:
                                nextstate=[state[0],state[1],state[2],mm[1],max(state[4]-indi[1],0)]
                                probab=round(indi[0]* mm[0],3)
                                #print(indi[0],mm[0],round(probab,3))
                                state_nextstates.append([nextstate,probab])
                                rew+=probab*(stepcost)

                elif state[3]=="R":
                    for mm in possibility[state[3]]: 
                        for indi in act[1]:
                            probab=indi[0]*mm[0]
                            #print(probab)
                            if mm[1]=="D" and (state[0]=="E" or state[0]=="C"):
                                nextstate=[state[0],state[1],0,mm[1],min(state[4]+25,100)]
                                rew+=probab*(stepcost - 40)    #indiana got hit 
                            else:
                                nextstate=[state[0],state[1],state[2],mm[1],max(state[4]-indi[1] ,0)]
                                rew+=probab*(stepcost)
                            state_nextstates.append([nextstate,probab])
                state_array.append([state_action,rew,state_nextstates])

            elif act[0] == "SHOOT" and state[2] != 0:
                state_action=act[0]   
                if state[3] == "D":         #move and D
                        for mm in possibility[state[3]]: 
                            for indi in act[1]:
                                nextstate=[state[0],state[1],state[2]-1,mm[1],max(state[4]-indi[1],0)]
                                probab=indi[0]*mm[0]
                                state_nextstates.append([nextstate,probab])
                                rew+=probab*(stepcost)

                elif state[3]=="R":
                    for mm in possibility[state[3]]: 
                        for indi in act[1]:
                            probab=indi[0]*mm[0]
                            if mm[1]=="D" and (state[0]=="E" or state[0]=="C"):
                                nextstate=[state[0],state[1],0,mm[1],min(state[4]+25,100)]
                                rew+=probab*(stepcost - 40)    #indiana got hit 
                            else:
                                nextstate=[state[0],state[1],state[2]-1,mm[1],max(state[4]-indi[1],0)]
                                rew+=probab*(stepcost)
                            state_nextstates.append([nextstate,probab])

            #put this action in this state account
                state_array.append([state_action,rew,state_nextstates]) #this is allvals[indexofstate]  
    allvals[stateindex]=state_array



def compute_x(index):
    x_state=[]
    for stateform in allvals[index]:
            x_state.append(stateform[0])
    x_arr[index]=x_state                  



def compute_r(index):
    r_state=[]
    for stateform in allvals[index]:
            r_state.append(stateform[1])
    r_arr[index]=r_state  



def compute_alpha(index):
    if index==startindex:
        alpha_arr[index]=1
    else:
        alpha_arr[index]=0    



def compute_A(col):
    global A
    A=[[0 for i in range(col)]for j in range(600)]
    column=0
    ind=0
    for states in allvals:
        for acts in states:   #each act corresponds to each column
            A[ind][column]+=1  
            #change values of consequences
            if acts[0]!="NONE":
                for cons in acts[2]:
                    A[getindex(cons[0])][column]-=cons[1]
            column+=1  
        ind+=1





x_arr=[[]for i in range(600)]
r_arr=[0 for i in range(600)]
alpha_arr=[[]for i in range(600)]
allvals=[[] for i in range(600)]  #[0]will later have next [action ,reward, [consquence,probab]] such n arrays of state i in an array of array format
juststates=[]
count=0
final=0
for a1 in pos:
    for a2 in mat:
        for a3 in arrows:
            for a4 in mmstate:
                for a5 in health:
                    juststates.append([a1,a2,a3,a4,a5]) #all 600 states
                    if a5==0:
                        final+=1
                    compute([a1,a2,a3,a4,a5],count)                  
                    count+=1

count=0
startindex=getindex(["C",2,3,'R',100])
for a1 in pos:
    for a2 in mat:
        for a3 in arrows:
            for a4 in mmstate:
                for a5 in health:
                    compute_x(count)
                    compute_r(count)
                    compute_alpha(count) 
                    count+=1

#print(x_arr)
tot=0
for stateform in allvals:
    tot+=len(stateform)
    #print(tot)
compute_A(tot)    
  
#library usage
x = cp.Variable(shape=(tot,1), name="x") 
r=[]
for vals in r_arr:
    for val in vals:
        r.append(val)
r=np.array(r) 
B=copy.deepcopy(A)       
A=np.array(A)
constraints=[]
for i in range(600):
    constraints+= [ 
    cp.matmul(A[i], x) == alpha_arr[i], x>=0]

objective=cp.Maximize(cp.matmul(r,x))
problem = cp.Problem(objective, constraints)
solution = problem.solve()
print(solution)
arr1 = list(x.value)
l = [ float(val) for val in arr1]

state_action_pair=[]
left=0
right=0
loop=0
for vals in x_arr:
    right+=len(vals)
    tempar=l[left:right]
    actss=vals[tempar.index(max(tempar))]
    statess=juststates[loop]
    loop+=1
    left=right
    state_action_pair.append([statess,actss])
print(len(state_action_pair))    

output={
    "a":B,
    "r":r,
    "alpha": alpha_arr,
    "x": l,
    "policy":state_action_pair,
    "objective":solution
}
#todo in output r is np array make it list (initally it was list, get a deepcopy of it)
#todo state_action pair has 'S' for state it should be S and 
f = open("out2.txt", "w") 
f.write(str(output))
f.close()
# for i in range(len(r_arr)):
#     for j in r_arr[i]:
#         if j!=-10 and j!=0:
#             print(juststates[i])
        
#print (allvals[getindex(["C",0,1,"R",25])])