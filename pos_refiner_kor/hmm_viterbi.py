
# coding: utf-8

# In[1]:




# In[1]:

import math #log 계산을 위해 import한다.


# In[2]:

global m_p #pos = {} : 품사와 그 품사의 개수를 저장한다.
global pos #m_p = {} : 형태소/품사 묶음과 그 묶음의 개수를 저장한다.
global p_p #p_p = {} : 품사/품사 묶음과 그 묶음의 개수를 저장한다.(품사/품사는 연속적으로 쓰인 두 품사)
m_p = {}
pos = {}
p_p = {}


# In[3]:

def tokenizer_for_m_p(line):#입력받은 세종 corpus를 tokenize하고 필요에 맞게 pre-processing한다. 그리고 저장한다.
    
    if '++' not in line and line[0] != '+':
        component = line.split('+')         
            
    else:#예외 처리하는 부분 '++'일 경우 바로 tokenizing하면 문제가 생겨서 아래의 과정을 거친다.
        component = []
        component_component = ""
        for i in range(len(line)):                        
            
            if  i != 0 and line[i] == '+' and line[i-1] != '+':
                component.append(component_component)
                component_component = ""
            else:
                component_component += line[i]
                
                if i == len(line)-1:#마지막 부분을 빼놓지 않기 위해..(마지막 요소)
                    component.append(component_component)
        
    
    for i in component:#형태소/품사 저장
        if i in m_p:
            m_p[i][0] += 1
        else:
            m_p[i] = [1]
    
    
    #tokenizer_for_pos and tokenizer_for_p_p       #'//'인 경우의 예외처리, /가 시작인 경우도..
    p_p_component = []
    
    for i in component:#품사 저장
        
        if '//' not in i and i[0] != '/':
            i = i.split('/')[1] #            pos_component = i.split('/')[1]
        else:
            i = i[2:]
            
        if i in pos:
            pos[i][0] += 1
        else:
            pos[i] = [1]
        
        p_p_component.append(i)
    
    for i in range(len(p_p_component)-1): #품사/품사 저장
        p_p_new_component = p_p_component[i]+','+p_p_component[i+1]
        
        if p_p_new_component in p_p:
            p_p[p_p_new_component][0] += 1
        else:
            p_p[p_p_new_component] = [1]
    
    
    


# In[4]:

with open("train.txt") as f: #세종 corpus를 읽어 들이고 한 줄마다 tokenizer로 보내서 전처리 후 저장하는 과정을 거친다.
    for i in f.readlines(): 
        line = i.splitlines()[0]
        
        if line is '':
            continue
        
        line = line.split('\t')[1]
        tokenizer_for_m_p(line)        


# In[5]:

def add_to_p_p(line):#문장의 시작인 '$' 부분을 찾아서 계산을 위해 품사로 저장하고, 또 시작부와 그 다음 형태소를 묶어서 품사/품사로 저장
    line = line.split('\t')[1]
    
    if line[0] == '/' or line[0] == '+':
        line = line[2:]
        component = line.split('+')[0]
    else:
        line = line.split('/')[1]
        component = line.split('+')[0]
    component = '$,' + component 
    
    #C('$')를 바로 pos에 추가하는 부분
    if '$' in pos:
        pos['$'][0] += 1
    else:
        pos['$'] = [1]
    
    if component in p_p:
        p_p[component][0] += 1
    else:
        p_p[component] = [1]


# In[6]:

#P(POS | $) 를 구하기 위해서 $,POS count 구해야 한다. 이를 위해 아래 부분에서 다시 세종 corpus를 읽어들여 처리한다.
with open('train.txt') as f:
    flg = 0
    first = 1
    for i in f.readlines():
        line = i.splitlines()[0]
        if flg == 1 or first == 1:
            add_to_p_p(line)
            first = 0
        if line is "":
            flg = 1
            continue
        else:
            flg = 0


# In[7]:

def calculation(component, p_p_component, p_p_):#앞에 것이 분자, 중간 것은 분모, 마지막 것이 분자가 된다.
    #어절의 생성확률을 계산하는 부분이다.
    ret_val = 0
    
    smoothing_factor = len(pos.keys())
    #분자 계산
    for i in range(len(component)):
        if component[i] in m_p:
            ret_val += math.log( (m_p[component[i]][0] + 1) ) #Laplace smoothing 적용
    #분자 계산 2
    for i in range(len(p_p_)):
        if p_p_[i] in p_p:
            ret_val += math.log( (p_p[p_p_[i]][0] + 1) )
            
    #분모 계산
    for i in range(len(p_p_component)):
        if p_p_component[i] in pos:
            ret_val -= ( 2 * math.log( (pos[p_p_component[i]][0] + smoothing_factor) ) )
        else:#분모가 unseen일 경우도 smoothing factor는 더해줘야한다.
            ret_val -= ( 2 * math.log( smoothing_factor))
    if p_p_component[i] in pos:
        ret_val += math.log( (pos[p_p_component[i]][0] + smoothing_factor) )
    else:#마찮가지로 smoothing_factor를 더해줘야...
        ret_val += math.log( somoothing_factor )
    
    return ret_val    


# In[8]:


# In[ ]:




# In[2]:

def cal_prob_phrase(phrase): #결과적으로 어절의 확률과, 어절의 첫 품사 끝 품사를 list로 반환한다.
    line = phrase
   
    
    if '++' not in line and line[0] != '+':
        component = line.split('+')         
            
    else:#예외 처리하는 부분 '++'일 경우 바로 tokenizing하면 문제가 생겨서 아래의 과정을 거친다.
        component = []
        component_component = ""
        for i in range(len(line)):                        
            
            if  i != 0 and line[i] == '+' and line[i-1] != '+':
                component.append(component_component)
                component_component = ""
            else:
                component_component += line[i]
                
                if i == len(line)-1:#마지막 부분을 빼놓지 않기 위해..(마지막 요소)
                    component.append(component_component)
    
    p_p_component = []
    for i in component:
        i = i.split('/')[1]
        if i in pos:
            pos[i][0] += 1
        else:
            pos[i] = [1]
        
        p_p_component.append(i)
    
    p_p_ = []
    for i in range(len(p_p_component)-1):
        p_p_new_component = p_p_component[i]+','+p_p_component[i+1]
        p_p_.append(p_p_new_component)
    
    prob = 0
    
    prob = calculation(component, p_p_component, p_p_)
    
    ret = prob, p_p_component[0], p_p_component[-1]
    return ret


# In[ ]:




# In[3]:





# In[9]:

#back-trace하여 최적의 확률들을 골라 최적의 품사열을 구한다.
def backtrace(viterbi_cal, index_history, sentence):
    out_str = ""
    for i in range(        len(viterbi_cal)-1, 0, -1):
        if i == len(viterbi_cal) - 1:
            max_index = 0
            max_prob = viterbi_cal[i][0]
            for j in range(len(viterbi_cal[i])):
                if viterbi_cal[i][j] > max_prob:
                    max_index = j
                    max_prob = viterbi_cal[i][j]
            out_str += sentence[i][max_index]
        else:
            next_index = index_history[i+1][max_index]
            out_str = sentence[i][next_index] + ' ' + out_str            
            max_index = next_index
    return out_str                        
        


# In[10]:

#어절의 확률을 구하고 이를 바탕으로 Viterbi algorithm을 적용한다. back-trace도 한다. 최종 품사열을 반환하여 준다.
def viterbi(sentence):
    prob_of_phrases = []
    first_last_pos = []
    index_history = []
    viterbi_cal = []
    
    #어절의 확률을 구하고 어절의 첫 품사와 끝 품사를 구해서, 전자는 prob_of_phrases에 후자는 first_last_pos에 저장한다.
    for i in range(len(sentence)):
        prob_p = []
        f_l_p = []
        
        for j in range(len(sentence[i])):
            prob, start, end = cal_prob_phrase(sentence[i][j])
            prob_p.append(prob)
            f_l_p.append([start, end])
        prob_of_phrases.append(prob_p)
        first_last_pos.append(f_l_p)
    
    #Viterbi algorithm으로 prob을 구한다.
    for i in range(len(prob_of_phrases)):
        viterbi_component = []
        index_component = []
        for j in range(len(prob_of_phrases[i])):
            if i == 1 :
                prob = prob_of_phrases[i][j]
                if '$,'+first_last_pos[i][j][0] in p_p:
                    prob += math.log( ( p_p[ '$,'+first_last_pos[i][j][0] ][0] + 1 ) )
                #if '$' in pos: 시작기호는 무조건 있으니깐 굳이 조건부로 해줄 필요가 없는듯
                prob -= math.log( (pos['$'][0] + len(pos.keys()) ) )
                viterbi_component.append(prob)
                index_component.append(0)
                
            else:
                max_index = 0
                max_prob = viterbi_cal[i-1][0]
                if first_last_pos[i-1][0][1] in pos:
                    max_prob -= math.log( ( pos[ first_last_pos[i-1][0][1] ][0]  + len(pos.keys()) ) )
                else:
                    max_prob -= math.log( len(pos.keys()) )
                
                if first_last_pos[i-1][0][1]+','+first_last_pos[i][j][0] in p_p:
                    max_prob += math.log( (  p_p[ first_last_pos[i-1][0][1]+','+first_last_pos[i][j][0] ][0] + 1 ) )
                for k in range( len(prob_of_phrases[i-1]) ):
                    prob = viterbi_cal[i-1][k]
                    if first_last_pos[i-1][k][1] in pos:
                        prob -= math.log( ( pos[ first_last_pos[i-1][k][1] ][0]  + len(pos.keys()) ) )
                    else:
                        prob -= math.log( len(pos.keys()) )                  
                    
                    if first_last_pos[i-1][k][1]+','+first_last_pos[i][j][0] in p_p:                          
                        prob += math.log( (  p_p[ first_last_pos[i-1][k][1]+','+first_last_pos[i][j][0] ][0] + 1 ) )                        
                    if prob > max_prob :
                        max_prob = prob
                        max_index = k
                max_prob += prob_of_phrases[i][j]
                viterbi_component.append(max_prob)
                index_component.append(max_index)
                    
        
        viterbi_cal.append(viterbi_component)
        index_history.append(index_component)
    
    
    return backtrace(viterbi_cal, index_history, sentence)
    
    
    
    


# In[11]:

#result.txt부분을 읽어들여서 전처리를 하고 viterbi algorithm을 적용한다. 이를 위해 모듈로 보낸다. 그리고 반환값을 받아들여 출력시킨다.    


# In[4]:

with open("result.txt", 'r') as f:
    sentman = []
    
    flg = 0
    cnt = 0
    for i in f.readlines():
        line = i.splitlines()[0]
        
        if flg == 0:
            sentence = []
            phrases = []
            flg = 1
            
        if line is '':
            cnt += 1
            if cnt == 3:
                sentence.append(phrases)    
                sentman.append(sentence)    
                flg = 0
            continue
        else:
            cnt = 0
        
        if line[0] == ' ' or line[0] == '1' or line[0] == '2' or line[0] == '3':
            if line[4:] == '': ################## 수정한 부분...
                sentence.append(phrases)#########
                phrases = []####################
            else:###############################
                phrases.append(line[4:])########            
        else:
            sentence.append(phrases)
            phrases = []
    
    output_text = open("output.txt", "w")
    for i in range(len(sentman)):
        output_component = viterbi(sentman[i])
        output_text.write(output_component+'\n')
    output_text.close()              


# In[ ]:



