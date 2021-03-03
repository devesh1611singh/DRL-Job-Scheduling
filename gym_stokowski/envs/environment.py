import numpy as np
import pandas as pd
from tabulate import tabulate
import copy
import gym
from prompt_toolkit import input
from pandas._libs import index
import openpyxl
import pickle

#import cProfile, pstats, io
#from pstats import SortKey

class Environment(gym.Env):

    #Calls the reset method
    def __init__(self):
 
        self.is_training = True
        self.problems_seen = pickle.load( open( "problems_seen.pkl", "rb" ) ) #This will be a dictionary to track the pandas of files seen, so we do not need to read them over and over during extensive training.
        self.problem_instance = 2
        self.reset()
        
    
    #Initializes the input variables,setup times and  associates the jobs and machines with the contingency table
    def reset(self):
        self.step_counter = 0
        #Stats
        self.stats = {"jobs_completed": 0, "makespan":0, "flow_time":0, "penalty_count":0, "average_tardiness": 0, "total_tardiness": 0, "total_major_setups":0, "total_minor_setups": 0, "rewards":[]}#We still add... "machine_start_times":{}, "machine_end_times":{}, "job_end_times":{}, "job_start_times":{}, "job_makespans":{}, "machine_makespans":{}
        #I did not know how to calculate "flow_time":0
        self.good = 0
        

        #Constants
        self.num_machines = 17
        self.num_jobs = 160
        self.num_families_s1 = 40
        self.num_families_s3 = 3
        self.num_stages = 4
        self.highest_prio = 20
        self.machines_to_stages={0:0,1:0,2:0,3:0,4:0,5:1,6:1,7:1,8:1,9:1,10:2,11:2,12:2,13:2,14:2,15:3,16:3}#Machine number to stage

        mst = dict()
        met = dict()
        mms = dict()
        jet = dict()
        jst = dict()
        jms = dict()

        for item in range(self.num_machines):
            mst[item] = -1
            met[item] = -1
            mms[item] = -1
        self.stats["machine_start_times"] = mst
        self.stats["machine_end_times"] = met
        self.stats["machine_makespans"] = mms

        for item in range(self.num_jobs):
            jst[item] = -1
            jet[item] = -1
            jms[item] = -1
        self.stats["job_start_times"] = jst
        self.stats["job_end_times"] = jet
        self.stats["job_makespans"] = jms

        self.stats["log"]=[]

        """
        Observation space: (93 columns)
        (Jobs by position)
        First, waiting for assignation
        Second, priority one hot encoded...
        Third, columns (one hot encoded family type 1 (, family type 2), 
        Then: processing time per stages
        Then: Up to what stage they have completed..., and whether they are done
        One what machine they currently are in...
        Other equality to family of last machine
        Difference to date...

        """
        #Indexes for accessing data on the state
        self.state_idx_wait_assign = 0
        self.state_idx_start_prio = 1
        self.state_idx_start_family_s1 = self.state_idx_start_prio+self.highest_prio
        self.state_idx_start_family_s3 = self.state_idx_start_family_s1+self.num_families_s1
        self.state_idx_pt_s1 = self.state_idx_start_family_s3 + self.num_families_s3
        self.state_idx_pt_s2 = self.state_idx_pt_s1 + 1
        self.state_idx_pt_s3 = self.state_idx_pt_s2 + 1
        self.state_idx_pt_s4 = self.state_idx_pt_s3 + 1
        self.state_idx_state_s1 = self.state_idx_pt_s4 + 1
        self.state_idx_state_s2 = self.state_idx_state_s1 + 1
        self.state_idx_state_s3 = self.state_idx_state_s2 + 1
        self.state_idx_state_s4 = self.state_idx_state_s3 + 1
        self.state_idx_is_done = self.state_idx_state_s4 + 1
        self.state_idx_start_machines = self.state_idx_is_done + 1
        self.state_idx_start_eq_last_fam_machines = self.state_idx_start_machines + self.num_machines
        self.state_idx_time_left = self.state_idx_start_eq_last_fam_machines + self.num_machines

        #Start time
        self.time = 0
        self.last_time = 0

        #Needed Gym variables
        self.action_space = gym.spaces.Discrete(1+(self.num_machines*self.num_jobs))
        #To move throught the actions we can use: machine_number*self_number_of_jobs+job id, this would give us the action of placing a job id on a machine number...
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,self.num_jobs, self.state_idx_time_left+1), dtype=np.float32)

        #Data load
        if self.is_training:
            machine_file = "machines.csv"
            problem_instance_file = "P1.xlsx"
            sheet_name = str(self.problem_instance)
        else:
            machine_file = "machines.csv"
            problem_instance_file = "P1.xlsx"
            sheet_name = str(self.problem_instance)

        machines = pd.read_csv(machine_file,sep=",")
        if not machine_file+"_"+problem_instance_file+"_"+sheet_name in self.problems_seen:
            readData = pd.read_excel(problem_instance_file, sheet_name=sheet_name)
            readData = readData[['Job ID','Priority'
                         ,'Family ID (SMD)','Processing time (SMD)','Processing time (AOI)','Processing time (SS)',
                         'Processing time (CC)','Family Type (CC)', 'Deadline in (min)']]
            readData.sort_values(by=['Job ID'], inplace=True)
            readData = readData.reset_index(drop=True)
            self.data = readData
            self.time_normalizer = self.data['Deadline in (min)'].max()
            self.deadlines={}#This is a dictionary that tracks for each job (by position) the deadline in minutes.
            self._init_state() #This initializes self.state
            self.problems_seen[machine_file+"_"+problem_instance_file+"_"+sheet_name]=(self.data, self.time_normalizer, self.deadlines, self.state)
            pickle.dump(self.problems_seen, open( "problems_seen.pkl", "wb" ) )
        else:
            self.data, self.time_normalizer, self.deadlines, self.state = self.problems_seen[machine_file+"_"+problem_instance_file+"_"+sheet_name]

        
        #Now we set the action mask
        self.unfinished_jobs=set([x for x in range(self.num_jobs)])
        self.triggers={}#Next actions to be done...When time increases
        self._init_action_mask() #This initializes self.action_mask. The semantic of the action mask is that there are 160 job-placing actions per machine.
        return np.expand_dims(self.state, axis=0)
        
    #Sets that we are training
    def set_train(self):
        self.is_training = True

    #Sets that we are testing
    def set_test(self):
        self.is_training = False
    
    @property
    def _n_actions(self):
        return int(1+(self.num_machines*self.num_jobs))

    def _get_obs(self):
        return np.expand_dims(self.state, axis=0)

    def _init_state(self):
        self.state = np.zeros((self.num_jobs, self.state_idx_time_left+1), dtype=np.float32)
        for index, row in self.data.iterrows():
            self.state[index][self.state_idx_wait_assign] = 1.
            self.state[index][self.state_idx_start_prio+int(row['Priority'])-1] = 1. #Poor man's one-hot-encoding
            self.state[index][self.state_idx_start_family_s1+int(row['Family ID (SMD)'])-1] = 1. #Poor man's one-hot-encoding
            if np.isnan(row['Family Type (CC)']):
                for item in range(self.num_families_s3):
                  self.state[index][self.state_idx_start_family_s3+item] = -1.
            else:
                self.state[index][self.state_idx_start_family_s3+int(row['Family Type (CC)'])-1] = 1. #Poor man's one-hot-encoding
            self.state[index][self.state_idx_pt_s1] = row['Processing time (SMD)']/self.time_normalizer if not np.isnan(row['Processing time (SMD)']) else -1.
            if self.state[index][self.state_idx_pt_s1]==-1.:
                self.state[index][self.state_idx_state_s1]=1.
            self.state[index][self.state_idx_pt_s2] = row['Processing time (AOI)']/self.time_normalizer if not np.isnan(row['Processing time (AOI)']) else -1.
            if self.state[index][self.state_idx_pt_s2]==-1.:
                self.state[index][self.state_idx_state_s2]=1.
            self.state[index][self.state_idx_pt_s3] = row['Processing time (SS)']/self.time_normalizer if not np.isnan(row['Processing time (SS)']) else -1.
            if self.state[index][self.state_idx_pt_s3]==-1.:
                self.state[index][self.state_idx_state_s3]=1.
            self.state[index][self.state_idx_pt_s4] = row['Processing time (CC)']/self.time_normalizer if not np.isnan(row['Processing time (CC)']) else -1.
            if self.state[index][self.state_idx_pt_s4]==-1.:
                self.state[index][self.state_idx_state_s4]=1.
            self.state[index][self.state_idx_time_left] = row['Deadline in (min)']/self.time_normalizer-(row['Deadline in (min)']-self.time)/self.time_normalizer
            self.deadlines[index]=row['Deadline in (min)']
              


    def _is_done(self):
        #for item in range(self.num_jobs):
        #    if self.state[item][self.state_idx_wait_assign]==1. or len(self.triggers)>0: #If we have an unassigned job or still things on the triggers list we are not done..
        #        return False
        return len(self.unfinished_jobs)==0

    def _get_next_stage(self, job):
        v1 = self.state[job][self.state_idx_state_s1]==1.
        v2 = self.state[job][self.state_idx_state_s2]==1.
        v3 = self.state[job][self.state_idx_state_s3]==1.
        v4 = self.state[job][self.state_idx_state_s4]==1.
        if v1 and v2 and v3 and v4:
            return 4
        if v1 and v2 and v3:
            return 3
        if v1 and v2:
            return 2
        if v1:
            return 1
        return 0
    
    def _get_available_machines(self):
        avail = []
        for machine in range(self.num_machines):
            if np.any(self.state[:, self.state_idx_start_machines+machine] == 1):
                avail.append(False)
            else:
                avail.append(True)
        return avail

    def _get_action_mask(self):
    #For all unassigned items, check if there is a machine on their stage where they could be assigned
        self.action_mask = np.zeros((1+(self.num_machines*self.num_jobs)), dtype=np.int32)
        if self._is_done():
            return self.action_mask
        if len(self.triggers)>0:
            self.action_mask[0]= 1
        avail = self._get_available_machines()
        for item in range(self.num_jobs):
            if self.state[item][self.state_idx_wait_assign]!=0.:#Unassigned
                stage = self._get_next_stage(item)
                #if not stage==4 and not self.state[item][self.state_idx_is_done]==1.: #Done!
                #    self.state[item][self.state_idx_wait_assign]=0.
                #    self.state[item][self.state_idx_is_done]=1.
                if stage!=4:
                    for machine in range(self.num_machines):
                        if self.machines_to_stages[machine]==stage and avail[machine]:
                            self.action_mask[1+(160*machine)+item]=1
        return self.action_mask

    def _init_action_mask(self):
        #self._get_action_mask()
        #return 0
        self.action_mask = np.zeros((1+(self.num_machines*self.num_jobs)), dtype=np.int32)
        for job in range(self.num_jobs):
            for item in range(self.num_machines):
                if self.state[job][self.state_idx_pt_s1]!=-1.:
                    if self.machines_to_stages[item]==0: 
                        self.action_mask[(160*item)+job+1]=1
                elif self.state[job][self.state_idx_pt_s2]!=-1.:
                    if self.machines_to_stages[item]==1:
                        self.action_mask[(160*item)+job+1]=1
                elif self.state[job][self.state_idx_pt_s3]!=-1.:
                    if self.machines_to_stages[item]==2:
                        self.action_mask[(160*item)+job+1]=1
                elif self.state[job][self.state_idx_pt_s4]!=-1.:
                    if self.machines_to_stages[item]==3:
                        self.action_mask[(160*item)+job+1]=1

    def _calculate_reward(self, gains):
        if len(gains)==0 and (self.time-self.last_time)==0:
            return 0
        bad_now=0
        for x in gains:
          if x>0:
            self.good+= 1
          else:
            bad_now+=1
        return -1*bad_now#(sum([x for x in gains])/((self.time_normalizer)))#+((self.time-self.last_time)/self.time_normalizer)

    def _set_last_family(self, job, machine):
        s1 = False
        s3 = False
        family = 0
        if self.machines_to_stages[machine] == 0:
            s1 = True
            for item in range(self.num_families_s1):
                if self.state[job][self.state_idx_start_family_s1+item] == 1.:
                    family = item
                    break
        elif self.machines_to_stages[machine] == 2:
            s3 = True
            for item in range(self.num_families_s3):
                if self.state[job][self.state_idx_start_family_s3+item] == 1.:
                    family = item
                    break
        if s1 or s3:
            for j in range(self.num_jobs):
                if s1:
                    self.state[j][self.state_idx_start_eq_last_fam_machines+machine] = 1. if self.state[j][self.state_idx_start_family_s1+family] == 1. else 0.
                else:
                    self.state[j][self.state_idx_start_eq_last_fam_machines+machine] = 1. if self.state[j][self.state_idx_start_family_s3+family] == 1. else 0.

             

    def _act(self,action):
        #First we update the state according to the action
        selected_job = action
        selected_machine = 0
        setup_time=0
        while selected_job>=160:
            selected_job-=160
            selected_machine+=1
        #We mark the selected job as no longer waiting to be assigned
        family_change = False
        self.state[selected_job][self.state_idx_wait_assign]=0.
        if self.state[selected_job][self.state_idx_start_eq_last_fam_machines+selected_machine]!=1.:
            family_change = True
        if self.stats["job_start_times"][selected_job]==-1:
            self.stats["job_start_times"][selected_job] = self.time
        if self.stats["machine_start_times"][selected_machine]==-1:
            self.stats["machine_start_times"][selected_machine] = self.time
            family_change = False
        if self.machines_to_stages[selected_machine]==3 or self.machines_to_stages[selected_machine]==0:
            self.stats["log"].append("At time: "+str(self.time)+", Job: "+str(selected_job)+" moves to Machine: "+str(selected_machine)+", Family change was: "+str(family_change))
        else:
            self.stats["log"].append("At time: "+str(self.time)+", Job: "+str(selected_job)+" moves to Machine: "+str(selected_machine))
        if self.machines_to_stages[selected_machine]==0:
            pt = self.state_idx_pt_s1
            if family_change:
                setup_time+=65
                self.stats["total_minor_setups"]+=1
            else:
                setup_time+=20
            self.state[selected_job][self.state_idx_state_s1]=1.
        elif self.machines_to_stages[selected_machine]==1:
            pt = self.state_idx_pt_s2
            setup_time+=15
            self.state[selected_job][self.state_idx_state_s2]=1.
        elif self.machines_to_stages[selected_machine]==2:
            pt = self.state_idx_pt_s3
            setup_time+=25
            self.state[selected_job][self.state_idx_state_s3]=1.
        else:
            self.state[selected_job][self.state_idx_state_s4]=1.
            pt = self.state_idx_pt_s4
            if family_change:
                setup_time+=120
                self.stats["total_major_setups"]+=1
            else:
                setup_time+=15
        self.state[selected_job][self.state_idx_start_machines+selected_machine] = 1.
        leave_machine_time = self.time + (self.state[selected_job][pt]*self.time_normalizer)+setup_time
        new_set = self.triggers[leave_machine_time] if leave_machine_time in self.triggers else set()
        new_set.add((selected_job, selected_machine))
        self.triggers[leave_machine_time]=new_set
        if family_change:
            self._set_last_family(selected_job, selected_machine)
        return selected_job, selected_machine

    def _update(self, special_request):
        self.last_time=self.time
        gains=set()
        while True:
            self.action_mask = self._get_action_mask()
            am = set(self.action_mask[1:])#Excluding special wait action
            if 1 in am and not special_request:#We can do one action
                break
            #Since we cannot do any action and are not done, we move the clock by the triggers
            if len(self.triggers)>0:
                self.time = sorted(self.triggers.keys())[0]
                triggers = self.triggers.pop(self.time)
                for item in triggers:#Item is a tuple(selected_job, selected_machine)
                    self.stats["log"].append("At time: "+str(self.time)+", Job: "+str(item[0])+" leaves Machine: "+str(item[1]))
                    self.state[item[0]][self.state_idx_start_machines+item[1]]=0.#We free the machine
                    self.stats["machine_end_times"][item[1]] = self.time
                    if self._get_next_stage(item[0])==4 and self.state[item[0]][self.state_idx_is_done]!=1.: #We can take this as done...
                       self.state[item[0]][self.state_idx_is_done]=1.
                       self.state[item[0]][self.state_idx_wait_assign]=0.
                       gains.add(self.deadlines[item[0]]-self.time)
                       self.stats["job_end_times"][item[0]] = self.time
                       self.unfinished_jobs.remove(item[0])
                       self.stats["log"].append("At time: "+str(self.time)+", Job: "+str(item[0])+" is done. "+str(len(self.unfinished_jobs))+"/"+str(self.num_jobs)+" jobs pending.")                    
                       if self.deadlines[item[0]]-self.time<0:
                           self.stats["penalty_count"]+=1 
                           self.stats["total_tardiness"]+=(-1*self.deadlines[item[0]]-self.time) 
                           self.stats["average_tardiness"] = self.stats["total_tardiness"]/self.stats["penalty_count"]   
                    else:
                       self.state[item[0]][self.state_idx_wait_assign]=1.
                for item in range(self.num_jobs):
                    if not self.state[item][self.state_idx_is_done]==1.:
                        self.state[item][self.state_idx_time_left] = self.deadlines[item]/self.time_normalizer-(self.deadlines[item]-self.time)/self.time_normalizer
                    else:
                        self.state[item][self.state_idx_time_left] = 0.
                if special_request:
                  break
            else:
                break
        return self._calculate_reward(gains)
    
    def close(self,):
      return None    
    
    def step(self, action):
        self.step_counter+=1
        self.stats["makespan"]=self.time
        self.stats["jobs_completed"] = int(self.num_jobs-len(self.unfinished_jobs))
        if self.step_counter>1280: #Timeout
            self.stats["rewards"].append(-1.)
            self.action_mask = np.zeros((1+(self.num_machines*self.num_jobs)), dtype=np.int32)
            return np.expand_dims(self.state, axis=0), self.time_normalizer*(self.good/(1+self.time)), True, self.stats        
        if self._is_done():#Invalid action
            #print("Invalid action")
            self.stats["rewards"].append(0.)
            self.action_mask = np.zeros((1+(self.num_machines*self.num_jobs)), dtype=np.int32)
            return np.expand_dims(self.state, axis=0), 0., True, self.stats
        if action == 0:
            prev_time = self.time
            reward = int(self._update(True))
            if prev_time==self.time:
        #        pointless_wait = True
        #        #print("No point in waiting here.")
                self.stats["rewards"].append(0.)
                self.action_mask[0]=0
                return np.expand_dims(self.state, axis=0), -1., False, self.stats
        elif action<0 or action>len(self.action_mask) or self.action_mask[action]!=1:
                #selected_job = action-1
                #selected_machine = 0
                #while selected_job>=160:
                #    selected_job-=160
                #    selected_machine+=1
                #print(str(selected_job)+","+str(selected_machine))
                #print(self._get_available_machines()[selected_machine])
                self.stats["rewards"].append(0.)
                return np.expand_dims(self.state, axis=0), -1., False, self.stats
        
        #We act and update
        if action!=0:
            selected_job, selected_machine  = self._act(action-1)
            #print(str(selected_job)+", "+str(selected_machine))
            reward = self._update(False) #+ ((self.machines_to_stages[selected_machine]+1)/4)
        
        #We check if done. If that is the case we update stats
        done = False
        if self._is_done():
            #print("Seems we're really done :)")
            self.action_mask = self.action_mask = np.zeros((1+(self.num_machines*self.num_jobs)), dtype=np.int32)
            done = True
            cand_flow = 0
            for item in range(self.num_jobs):
                if self.stats["job_end_times"][item]!=-1 and self.stats["job_start_times"][item]!=-1:
                    if self.stats["job_end_times"][item]-self.stats["job_start_times"][item]> cand_flow:
                        cand_flow = self.stats["job_end_times"][item]-self.stats["job_start_times"][item]
                    self.stats["job_makespans"][item]=self.stats["job_end_times"][item]-self.stats["job_start_times"][item]
            for item in range(self.num_machines):
                if self.stats["machine_end_times"][item]!=-1 and self.stats["machine_start_times"][item]!=-1:
                    self.stats["machine_makespans"][item]=self.stats["machine_end_times"][item]-self.stats["machine_start_times"][item]
            self.stats["flow_time"]=cand_flow
            self.stats["log"].append("At time: "+str(self.time)+", All jobs are done :)")
            reward = reward + (self.time_normalizer*(self.good/(1+self.time)))+ 200 #Bonus                    
        self.stats["makespan"]=self.time
        self.stats["rewards"].append(reward)
        self.stats["jobs_completed"] = int(self.num_jobs-len(self.unfinished_jobs))
        self.action_mask = self._get_action_mask()
        return np.expand_dims(self.state, axis=0), float(reward), done, self.stats
"""
class OurSolver():

    def __init__(self, problem_instance):
        self.solutions_seen = pickle.load( open( "solutions_seen.pkl", "rb" ) ) #This will be a dictionary to track the pandas of files seen, so we do not need to read them over and over during extensive training.
        self.solutions_file = "Sols.xlsx"
        if not  self.solutions_file+"_"+problem_instance in self.solutions_seen:
            readData = pd.read_excel("Sols.xlsx", sheet_name=problem_instance)
            readData = readData[['Order ID','SMD Line'
                         ,'AOI Device','SS Machine','CC Machine','Start Time ',
                         'Start Time AOI','Working Time AOI', 'Start Time Painting']]
            readData.sort_values(by=['Order ID'], inplace=True)
            readData = readData.reset_index(drop=True)
            self.data = readData
            self.solutions_seen[self.solutions_file+"_"+problem_instance] = self.data
            pickle.dump(self.solutions_seen, open( "solutions_seen.pkl", "wb" ) )
        else:
            self.data = self.solutions_seen[self.solutions_file+"_"+problem_instance]

    def get_sol(self):
        actions = []
        action_dict = dict()
        for index, row in self.data.iterrows():
            if not np.isnan(row['SMD Line']):
                selected_machine=row['SMD Line']-1
                action = 1+(160*selected_machine)+index
                time = row['Start Time ']
                if np.isnan(row['Start Time ']):
                    print("Dataset problem")
                    return 0
                if not time in action_dict:
                    action_dict[int(time)]=set()
                action_dict[int(time)].add(int(action))
            if not np.isnan(row['AOI Device']):
                selected_machine=4+row['AOI Device']
                action = 1+(160*selected_machine)+index
                time = row['Start Time AOI']
                if np.isnan(row['Start Time AOI']):
                    print("Dataset problem")
                    return 0
                if not time in action_dict:
                    action_dict[int(time)]=set()
                action_dict[int(time)].add(int(action))
            if not np.isnan(row['SS Machine']):
                selected_machine=9+row['SS Machine']
                action = 1+(160*selected_machine)+index
                time = row['Working Time AOI']
                if np.isnan(row['Working Time AOI']):
                    print("Dataset problem")
                    return 0
                if not time in action_dict:
                    action_dict[int(time)]=set()
                action_dict[int(time)].add(int(action))
            if not np.isnan(row['CC Machine']):
                selected_machine=14+row['CC Machine']
                action = 1+(160*selected_machine)+index
                time = row['Start Time Painting']
                if np.isnan(row['Start Time Painting']):
                    print("Dataset problem")
                    return 0
                if not time in action_dict:
                    action_dict[int(time)]=set()
                action_dict[int(time)].add(int(action))
        return action_dict

#Playground code: 
import copy
profile = False
full_sol_check = True
if profile:
    pr = cProfile.Profile()
    pr.enable()

sol = OurSolver("Problem Instance (2)")
prob = Environment(2)
actions = sol.get_sol()
can_do= dict()
for item in sorted(actions.keys()):
    if item <= prob.time:
        can_do[item]=actions.pop(item)
    else:
        break
counter=0
while len(can_do)>0 or len(prob.triggers)>0 or len(actions)>0:
    for item in sorted(actions.keys()):
        if item <= prob.time:
            can_do[item]=actions.pop(item)
        else:
            break
    counter+=1
    if not full_sol_check and counter>20:
        break
    print("Time: "+str(prob.time))
    action = None
    for t in sorted(can_do.keys()):
        for a in sorted(can_do[t]):
            if prob.action_mask[a]==1:
                action = a
                break
        if action!=None:
            can_do[t].remove(action)
            if len(can_do[t])==0:
                del can_do[t]
            break
    if action == None:
        s,r,d,stats = prob.step(0)
        print("Time (after wait): "+str(prob.time))
    else:
        #print(action)
        s,r,d,stats = prob.step(action)
    if d:
        print("Done")
        if len(can_do)>0:
            print("Error")
            exit()

while not prob._is_done() and full_sol_check:
    print("Loop- Time (after wait): "+str(prob.time))
    s,r,d,stats = prob.step(0)
    print(d)
print("Done with loop")
if "log" in stats: 
    for item in stats["log"]:
        print(item)
print("Next events:")
for item in sorted(prob.triggers.keys()):
    print(str(item)+":"+str(prob.triggers[item]))
print(stats)

if profile: 
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

#a.step(0)
#a.step(161)
#a.step(322)
#a.step(483)
#s,r,d,stats = a.step(644)
#a.step(645)#5th job on machine 4
#for item in stats["log"]:
#    print(item)


"""