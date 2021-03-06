
# coding: utf-8

# In[1]:


import model
import utils
import sys
from importlib import reload#


reload(sys)

#sys.setdefaultencoding('utf-8') #python3 do not need (StackOverflow.com)


#import tensorflow as tf


# create instance of config
config = utils.Config()

# build model
model = model.NERmodel(config)
model.build()
model.restore_session(config.dir_model)

#saver = tf.train.Saver()
#with tf.Session() as sess :
#    saver.restore(sess, config.dir_model)

# create dataset
test  = utils.data_read(config.filename_test, config.processing_word,
                            config.processing_tag, config.max_iter)

# evaluate and interact
print ("Testing model over test set")
res = model.run_evaluate(test)
print("acc : "+str('%.2f'%res['acc'])+" - "+"f1 : "+str('%.2f'%res['f1']))






# In[2]:

#input example
#input_sent= ['23/SN', '일/NNB', '기성용/NNP', '의/JKG' ,'활약/NNG', '으로/JKB', '스완지시티/NNP', '는/JX', '리버풀/NNP', '전/NNG', '에서/JKB', '승리/NNG', '를/JKO', '얻/VV', '었/EP', '다/EC', './SF']
f = open('element.txt', 'r')
line = f.readline()
line = line.replace('+', ' ')
input_sent = line.split()

f.close()


# In[3]:

res = model.predict(input_sent)

f = open('mid_element.txt', 'w')

for inseq, label in zip(input_sent,res) :
    print (inseq+" "+label, end = ' ', file = f)
    

f.close()


# In[4]:




# In[ ]:



