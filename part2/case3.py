# doubt :-
# Is there any sense of MM changing state after it's health becomes 0?
# handling ofcorner cases like MM can't attack after 0 health
import random
import numpy as np 
import json
import math

step_cost = 10
x = 64
gamma = 0.25
delta = 1e-3
mm_health = 100
final_reward = 50       #when IJ kills the MM(health of MM becomes 0)
bad_reward = 40         # when MM attacks IJ succesfully it loses -40 reward

time_instance = 10000    # max time to converge
positions = 5           # 5 different positions are possible('W'(0), 'N'(1), 'E'(2), 'S'(3), 'C'(4))..in this order
material = 3            # 0,1,2 material
arrows = 4              # 0,1,2,3 arrows can be possible
state = 2               # 2 states are possible 0:dormant 1:ready
health = 101            # health value from 0-100
action = 9              # 9 actions in order:- UP(0),DOWN(1),LEFT(2),RIGHT(3),STAY(4),SHOOT(5),HIT(6),CRAFT(7),GATHER(8) (SHOOT the arrow and hit with the blade)

choose_action = ['UP','DOWN','LEFT','RIGHT','STAY','SHOOT','HIT','CRAFT','GATHER','NONE']
choose_position = ['W','N','E','S','C']
choose_state = ['D', 'R']

# initializing all utilities to 0 initially
utility = np.zeros(shape=(time_instance, positions, material, arrows, state, health))

def setUtil(val, a, b, c, d, e, f):
    utility[a,b,c,d,e,f] = val
    print(a, b, c, d, e, f, val)

def max_over_all_actions(itr,pos,mat,arr,sta,hea):
    best_action = ''
    value = 0.0
    ma = -100000000.0     #setting ma value to big -ve
    for act in range(action):
        value = -100000000.0     #setting ma value to big -ve
        
        #absorbing state 
        if hea == 0:
            best_action = 'NONE'
            ma = 0.0
            break
        
        # west square (IJ will not be affected by MM's attack here)
        if pos == 0:
            # dormant state
            if sta == 0:
                # movement
                if act == 3:
                    value = (-step_cost + gamma*(1.0*(0.2*utility[itr-1,4,mat,arr,1,hea] + 0.8*utility[itr-1,4,mat,arr,0,hea]))) 
                                                  
                elif act == 4:
                    value = (-step_cost + gamma*(1.0*(0.2*utility[itr-1,0,mat,arr,1,hea] + 0.8*utility[itr-1,0,mat,arr,0,hea])))
                                                
                    
                    # arrow shoot
                elif act == 5 and arr>0 and hea>0:
                    value = (-step_cost + gamma*(0.25*(0.2*utility[itr-1,pos,mat,max(0,arr-1),1,max(0,hea-25)] + 0.8*utility[itr-1,pos,mat,max(0,arr-1),0,max(0,hea-25)]) + 0.75*(0.2*utility[itr-1,pos,mat,max(0,arr-1),1,hea] + 0.8*utility[itr-1,pos,mat,max(0,arr-1),0,hea])))
                    # adding final reward if the MM is killed
                    if hea == 25:
                        value += (0.25*final_reward + 0.75*0)
            
            # ready state (sta=1)
            else:
                # movement
                if act == 3:
                    value = (-step_cost + gamma*(1.0*(0.5*utility[itr-1,4,mat,arr,1,hea] + 0.5*utility[itr-1,4,mat,arr,0,hea]))) 
                                                  
                elif act == 4:
                    value = (-step_cost + gamma*(1.0*(0.5*utility[itr-1,0,mat,arr,1,hea] + 0.5*utility[itr-1,0,mat,arr,0,hea])))
                                                
                    
                    # arrow shoot
                elif act == 5 and arr>0 and hea>0:
                    value = (-step_cost + gamma*(0.25*(0.5*utility[itr-1,pos,mat,max(0,arr-1),1,max(0,hea-25)] + 0.5*utility[itr-1,pos,mat,max(0,arr-1),0,max(0,hea-25)]) + 0.75*(0.5*utility[itr-1,pos,mat,max(0,arr-1),1,hea] + 0.5*utility[itr-1,pos,mat,max(0,arr-1),0,hea])))
                    # adding final reward if the MM is killed
                    if hea == 25:
                        value += (0.25*final_reward + 0.75*0)
                    
                            
        
        # north square (IJ will not be affected by MM's attack here) 
        elif pos == 1:
            # dormant state
            if sta == 0:
                # movement
                if act == 1:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,4,mat,arr,1,hea] + 0.8*utility[itr-1,4,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 4:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,pos,mat,arr,1,hea] + 0.8*utility[itr-1,pos,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                    
                # material crafting
                elif act == 7 and mat>0:
                    value = (-step_cost + gamma*(0.5*(0.2*utility[itr-1,pos,max(0,mat-1),min(3,arr+1),1,hea] + 0.8*utility[itr-1,pos,max(0,mat-1),min(3,arr+1),0,hea]) + 0.35*(0.2*utility[itr-1,pos,max(0,mat-1),min(3,arr+2),1,hea] + 0.8*utility[itr-1,pos,max(0,mat-1),min(3,arr+2),0,hea]) + 0.15*(0.2*utility[itr-1,pos,max(0,mat-1),min(3,arr+3),1,hea] + 0.8*utility[itr-1,pos,max(0,mat-1),min(3,arr+3),0,hea])))
           
            # ready state (sta=1)
            else:
                # movement
                if act == 1:
                    value = (-step_cost + gamma*(0.85*(0.5*utility[itr-1,4,mat,arr,1,hea] + 0.5*utility[itr-1,4,mat,arr,0,hea]) + 0.15*(0.5*utility[itr-1,2,mat,arr,1,hea] + 0.5*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 4:
                    value = (-step_cost + gamma*(0.85*(0.5*utility[itr-1,pos,mat,arr,1,hea] + 0.5*utility[itr-1,pos,mat,arr,0,hea]) + 0.15*(0.5*utility[itr-1,2,mat,arr,1,hea] + 0.5*utility[itr-1,2,mat,arr,0,hea])))
                    
                    # material crafting
                elif act == 7 and mat>0:
                    value = (-step_cost + gamma*(0.5*(0.5*utility[itr-1,pos,max(0,mat-1),min(3,arr+1),1,hea] + 0.5*utility[itr-1,pos,max(0,mat-1),min(3,arr+1),0,hea]) + 0.35*(0.5*utility[itr-1,pos,max(0,mat-1),min(3,arr+2),1,hea] + 0.5*utility[itr-1,pos,max(0,mat-1),min(3,arr+2),0,hea]) + 0.15*(0.5*utility[itr-1,pos,max(0,mat-1),min(3,arr+3),1,hea] + 0.5*utility[itr-1,pos,max(0,mat-1),min(3,arr+3),0,hea])))
           
                
        # east square (IJ will be affected by MM's attack now) 
        elif pos == 2:
            # dormant state
            if sta == 0:
                # movement
                if act == 2:
                    value = (-step_cost + gamma*(1.0*(0.2*utility[itr-1,4,mat,arr,1,hea] + 0.8*utility[itr-1,4,mat,arr,0,hea])))
                elif act == 4:
                    value = (-step_cost + gamma*(1.0*(0.2*utility[itr-1,pos,mat,arr,1,hea] + 0.8*utility[itr-1,pos,mat,arr,0,hea])))
                    
                # arrow shoot
                elif act == 5 and hea>0 and arr>0:
                    value = (-step_cost + gamma*(0.9*(0.2*utility[itr-1,pos,mat,max(0,arr-1),1,max(0,hea-25)] + 0.8*utility[itr-1,pos,mat,max(0,arr-1),0,max(0,hea-25)]) + 0.1*(0.2*utility[itr-1,pos,mat,max(0,arr-1),1,hea] + 0.8*utility[itr-1,pos,mat,max(0,arr-1),0,hea])))
                    # adding final reward if the MM is killed
                    if hea == 25:
                        value += (0.9*final_reward + 0.1*0)
                # blade hit
                elif act == 6 and hea>0:
                    value = (-step_cost + gamma*(0.2*(0.2*utility[itr-1,pos,mat,arr,1,max(0,hea-50)] + 0.8*utility[itr-1,pos,mat,arr,0,max(0,hea-50)]) + 0.8*(0.2*utility[itr-1,pos,mat,arr,1,hea] + 0.8*utility[itr-1,pos,mat,arr,0,hea])))
                    # adding final reward if the MM is killed
                    if hea == 25 or hea == 50:
                        value += (0.2*final_reward + 0.8*0)
                        
            # ready state (sta=1)  
            else:
                # movement
                if act == 2:
                    value = (-step_cost -0.5*bad_reward  + gamma*(1.0*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,4,mat,arr,1,hea])))
                elif act == 4:
                    value = (-step_cost -0.5*bad_reward  + gamma*(1.0*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,arr,1,hea])))
                    
                # arrow shoot
                elif act == 5 and hea>0 and arr>0:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.9*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,max(0,arr-1),1,max(0,hea-25)]) + 0.1*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,max(0,arr-1),1,hea])))
                    # adding final reward if the MM is killed (with a probability that MM doesn't attack)
                    if hea == 25:
                        value += 0.5*(0.9*final_reward + 0.1*0)
                # blade hit
                elif act == 6 and hea>0:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.2*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,arr,1,max(0,hea-50)]) + 0.8*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,arr,1,hea])))
                    # adding final reward if the MM is killed (with a probability that MM doesn't attack)
                    if hea == 25 or hea == 50:
                        value += 0.5*(0.2*final_reward + 0.8*0)
                
                
         
        # south square (IJ will not be affected by MM's attack here)
        elif pos == 3:
            # dormant state
            if sta == 0:
                # movement
                if act == 0:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,4,mat,arr,1,hea] +0.8*utility[itr-1,4,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 4:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,pos,mat,arr,1,hea] + 0.8*utility[itr-1,pos,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                    
                # matrial gathering
                elif act == 8:
                    value = (-step_cost + gamma*(0.75*(0.2*utility[itr-1,pos,min(2,mat+1),arr,1,hea] + 0.8*utility[itr-1,pos,min(2,mat+1),arr,0,hea]) + 0.25*(0.2*utility[itr-1,pos,mat,arr,1,hea] + 0.8*utility[itr-1,pos,mat,arr,0,hea])))
                
            # ready state (sta=1)   
            else:
                # movement
                if act == 0:
                    value = (-step_cost + gamma*(0.85*(0.5*utility[itr-1,4,mat,arr,1,hea] + 0.5*utility[itr-1,4,mat,arr,0,hea]) + 0.15*(0.5*utility[itr-1,2,mat,arr,1,hea] + 0.5*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 4:
                    value = (-step_cost + gamma*(0.85*(0.5*utility[itr-1,pos,mat,arr,1,hea] + 0.5*utility[itr-1,pos,mat,arr,0,hea]) + 0.15*(0.5*utility[itr-1,2,mat,arr,1,hea] + 0.5*utility[itr-1,2,mat,arr,0,hea])))
                    
                # matrial gathering
                elif act == 8:
                    value = (-step_cost + gamma*(0.75*(0.5*utility[itr-1,pos,min(2,mat+1),arr,1,hea] + 0.5*utility[itr-1,pos,min(2,mat+1),arr,0,hea]) + 0.25*(0.5*utility[itr-1,pos,mat,arr,1,hea] + 0.5*utility[itr-1,pos,mat,arr,0,hea])))
                
        
        # Center square (IJ will not be affected by MM's attack here)
        elif pos == 4:
            # dormant state
            if sta == 0:
                # movement
                if act == 0:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,1,mat,arr,1,hea] + 0.8*utility[itr-1,1,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 1:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,3,mat,arr,1,hea] + 0.8*utility[itr-1,3,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 2:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,0,mat,arr,1,hea] + 0.8*utility[itr-1,0,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 3:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                elif act == 4:
                    value = (-step_cost + gamma*(0.85*(0.2*utility[itr-1,pos,mat,arr,1,hea] + 0.8*utility[itr-1,pos,mat,arr,0,hea]) + 0.15*(0.2*utility[itr-1,2,mat,arr,1,hea] + 0.8*utility[itr-1,2,mat,arr,0,hea])))
                
                # arrow shoot
                elif act == 5 and hea>0 and arr>0:
                    value = (-step_cost + gamma*(0.5*(0.2*utility[itr-1,pos,mat,max(0,arr-1),1,max(0,hea-25)] + 0.8*utility[itr-1,pos,mat,max(0,arr-1),0,max(0,hea-25)]) + 0.5*(0.2*utility[itr-1,pos,mat,max(0,arr-1),1,hea] + 0.8*utility[itr-1,pos,mat,max(0,arr-1),0,hea])))
                    # adding final reward if the MM is killed
                    if hea == 25:
                        value += (0.5*final_reward + 0.5*0)
                # blade hit
                elif act == 6 and hea>0:
                    value = (-step_cost + gamma*(0.1*(0.2*utility[itr-1,pos,mat,arr,1,max(0,hea-50)] + 0.8*utility[itr-1,pos,mat,arr,0,max(0,hea-50)]) + 0.9*(0.2*utility[itr-1,pos,mat,arr,1,hea] + 0.8*utility[itr-1,pos,mat,arr,0,hea])))
                    # adding final reward if the MM is killed
                    if hea == 25 or hea == 50:
                        value += (0.1*final_reward + 0.9*0)
            
            # ready state (sta=1) 
            else:
                # movement
                if act == 0:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.85*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,1,mat,arr,1,hea]) + 0.15*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,2,mat,arr,1,hea])))
                elif act == 1:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.85*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,3,mat,arr,1,hea]) + 0.15*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,2,mat,arr,1,hea])))
                elif act == 2:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.85*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,0,mat,arr,1,hea]) + 0.15*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,2,mat,arr,1,hea])))
                elif act == 3:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.85*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,2,mat,arr,1,hea]) + 0.15*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,2,mat,arr,1,hea])))
                elif act == 4:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.85*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,arr,1,hea]) + 0.15*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,2,mat,arr,1,hea])))
                
                # arrow shoot
                elif act == 5 and hea>0 and arr>0:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.5*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,max(0,arr-1),1,max(0,hea-25)]) + 0.5*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,max(0,arr-1),1,hea])))
                    # adding final reward if the MM is killed (with a probability that MM doesn't attack)
                    if hea == 25:
                        value += 0.5*(0.5*final_reward + 0.5*0)
                # blade hit
                elif act == 6 and hea>0:
                    value = (-step_cost -0.5*bad_reward + gamma*(0.1*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,arr,1,max(0,hea-50)]) + 0.9*(0.5*utility[itr-1,pos,mat,0,0,min(100,hea+25)] + 0.5*utility[itr-1,pos,mat,arr,1,hea])))
                    # adding final reward if the MM is killed (with a probability that MM doesn't attack)
                    if hea == 25 or hea == 50:
                        value += 0.5*(0.1*final_reward + 0.9*0)
                
                        
                            
        # here for actions having same utility choose random action wisely 
        if value != ma:
            if value > ma:
                ma = value
                best_action=choose_action[act]
        
    #writing in trace1.txt file    
    f = open("part_2_task_2.3_trace.txt", "a")
    f.write("({0}, {1}, {2}, {3}, {4}):{5} = [{6}]\n".format(choose_position[pos],mat,arr,choose_state[sta],hea,best_action,round(ma,3)))
    f.close()      
    return ma


def main(): 
    # wea re taking abs so max_diff can never be -1.0(everytime it will be more than -1.0) 
    max_diff = -1.0
    is_converged = 0
    for itr in range(1,time_instance):
        
        # writing to the file the generation number
        f = open("part_2_task_2.3_trace.txt", "a")
        f.write("iteration={0}\n".format(itr))
        f.close()
        
        temp_diff = 0.0
        for pos in range(positions):
            for mat in range(material):
                for arr in range(arrows):
                    for sta in range(state):
                        for hea in range(0,health,25):
                            temp=max_over_all_actions(itr,pos,mat,arr,sta,hea)
                            setUtil(temp, itr, pos, mat, arr, sta, hea)
                            temp_diff = max(temp_diff,abs(utility[itr,pos,mat,arr,sta,hea]-utility[itr-1,pos,mat,arr,sta,hea]))
        max_diff = temp_diff
                            
                            
        if max_diff <= delta:
            is_converged=1
            print('Converged Succesfully :-))')
            break
        
    print("Iteration Number : {0}\nMaximum Difference : {1}".format(itr,max_diff))
                        
# calling main function
main()
