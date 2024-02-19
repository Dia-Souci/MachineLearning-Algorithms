import pandas as pd 
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from collections import Counter
from itertools import combinations
import tkinter as tk
import time



#define data path 
DATA_PATH = os.path.join('.','data')
ORIGINAL_PATH = os.path.join(DATA_PATH, 'Dataset2.xlsx')

global df 
df= pd.read_excel(ORIGINAL_PATH)


df["definition"].fillna("hd",inplace=True)
categoryseries=df['videoCategoryId'].copy()
df.drop("videoCategoryId",axis=1,inplace=True)

df['videoCategoryLabel'] = df['videoCategoryLabel'].str.strip()

def detection_element_freq(min_sup):
    items = []
    truth = True
    for x in df['Watcher '].unique():
        df2 = df[df['Watcher '] == x]
        inside_list = []
        for y in df.columns:
            if(y=='Watcher '):
                val = df2[y].unique()[0]
                inside_list.append(val)
            else:        
                values = df2[y].unique()
                inside_list.append(values.tolist())
        items.append(inside_list)


    init = []
    for item in items:
        for x in item[1]:
            if(x not in init):
                init.append(x)
    init = sorted(init)
    minn=min_sup.split('.')
    if(int(minn[0])>0):
        sp = int(minn[0])
        supp_min = int(sp*len(items))/100
    else:
        sp=0.2
        supp_min = int(sp*len(items))

    print(f"le support minimum est {supp_min}")

    C = Counter()

    for i in init:
        for d in items:
            if(i in d[1]):
                C[i]+=1


    L = Counter()
    for i in C:
        if(C[i] >= supp_min):
            L[frozenset([i])]+=C[i]
    global pl , pc
    pl = []
    pc = []
    pl.append(L)
    pc.append(C)
    temp = list(L)

    for count in range (2,len(init)):
        nc = set()
        for i in range(0,len(temp)):
            for j in range(i+1,len(temp)):
                t = temp[i].union(temp[j])
                if(len(t) == count):
                    nc.add(t)
        nc = list(nc)
        c = Counter()
        for i in nc:
            c[i] = 0
            for q in items:
                temp = set(q[1])
                if(i.issubset(temp)):
                    c[i]+=1

        l = Counter()
        for i in c:
            if(c[i] >= supp_min):
                l[i]+=c[i]
        
        if(len(l) == 0):
            break
        pl.append(l)
        pc.append(c)
        temp = list(l)


def affichageElemFreq():
    top = tk.Toplevel()
    top.geometry("1200x800") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size. # makes the root window fixed in size.
    top.update()
    top.title('frequent items')
    top['bg']='#EFF5F5'
    insideFrame1 = tk.LabelFrame(top,text='Display',font=('Arial 8'))
    insideFrame1.place(height=800, width=1200 , rely=0 ,relx=0)
    insideFrame1['bg']='#D6E4E5'

    text = tk.Text(insideFrame1, height=700, width=450)
    #scroll = tk.Scrollbar(insideFrame1)
    #text.configure(yscrollcommand=scroll.set)
    text.place( rely=0 ,relx=0)

    text2=tk.Text(insideFrame1, height=700, width=450)
    #scroll = tk.Scrollbar(insideFrame1)
    #text.configure(yscrollcommand=scroll.set)
    text2.place( rely=0 ,relx=0.5) 
    #scroll.config(command=text.yview)
    #scroll.place(relx=1, rely=0, relheight=1, anchor='ne')
    
    insert_text = ""
    insert_text2 = ""
    count =0
    #---------------------Affichage  :C: / :L: ------------------------------------
    for i in range(len(pl)):
        print(f"C{i+1} :")
        if(count>36):
            insert_text2 += "C"+str(i+1)+" :\n"
        else:
            insert_text += "C"+str(i+1)+" :\n"
            count+=1
        
        for x in pc[i]:
            if(i==0):
                print(str(x)+": "+str(pc[i][x]))
                if(count>36):
                    insert_text2 += str(x)+": "+str(pc[i][x]) +"\n"
                else:
                    insert_text+=str(x)+": "+str(pc[i][x]) +"\n"
                    count+=1
            else:
                print(str(list(x))+": "+str(pc[i][x]))
                if(count>36):
                    insert_text2 += str(list(x))+": "+str(pc[i][x]) +"\n"
                else:
                    insert_text+=str(list(x))+": "+str(pc[i][x]) +"\n"
                    count+=1
                
        print(f"L{i+1} :")
        
        if(count>36):
            insert_text2+="L"+str(i+1)+" :\n"
        else:
            insert_text+="L"+str(i+1)+" :\n"
            count+=1
        for x in pl[i]:
            print(str(list(x))+": "+str(pl[i][x]))
            if(count>36):
                    insert_text2+=str(list(x))+": "+str(pl[i][x]) +"\n"
            else:
                insert_text+=str(list(x))+": "+str(pl[i][x]) +"\n"
                count+=1
            
    text.insert(tk.END, insert_text)
    text2.insert(tk.END,insert_text2)
        
    
    

print()


def extractionReglesAssociationetCor(minConf,bool):
    top = tk.Toplevel()
    top.geometry("1024x600") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.resizable(0, 0) # makes the root window fixed in size.
    top.update()
    top.title('Association rules')
    top['bg']='#EFF5F5'
    insideFrame1 = tk.LabelFrame(top,text='Display')
    insideFrame1.place(height=600, width=1024 , rely=0.05 ,relx=0)
    insideFrame1['bg']='#D6E4E5'

    text = tk.Text(insideFrame1, height=500, width=1000)
    scroll = tk.Scrollbar(insideFrame1)
    text.configure(yscrollcommand=scroll.set)
    text.place( rely=0.05 ,relx=0)
    
    insert_text = ""


 #--------------------------------------------- 
         
    
    minn=minConf.split('.')
    if(int(minn[0])>0):
        sp = int(minn[0])
        min_conf =int (sp)/100
    else:
        min_conf=0.3

    min_lift = 1
    correlation = False
    if(len(bool)==1):
        if(int(bool)==1):
            correlation=True
    
    global inference
    inference = {}
    antecedents = []
    result =[]

    for i in range(1,len(pl)):
        #print(f"Les règles du niveau {i+1} :\n")
        for x in pl[i] : 
            c = [frozenset(itemset) for itemset in combinations(x,i)]
            for item in c :
                A = x - item
                supp_item = pl[len(item)-1][item]
                supp_A_item = pl[i][x]
                supp_A = pl[len(A)-1][A]
                #confiance(A -> item) =  supp({A,item})/supp(A)
                confiance_A_item = supp_A_item / supp_A 
                confiance_item_A = supp_A_item / supp_item
                lift_A_item = confiance_A_item*100 /supp_item
                lift_item_A = confiance_item_A*100 / supp_A
                
                if(correlation == False):
                    if(confiance_A_item > min_conf) :
                        antecedents.append(tuple(list(A)))
                        result.append(tuple(list(item)))
                        insert_text += str(list(A)) + " ---> "+str(list(item)) + " || Confiance : "+ str(confiance_A_item) +"\n" 
                    if(confiance_item_A > min_conf) :
                        antecedents.append(tuple(list(item)))
                        result.append(tuple(list(A)))
                        insert_text += str(list(item)) + " ---> "+str(list(A)) + " || Confiance : "+ str(confiance_item_A) +"\n"
                else : 
                    if(confiance_A_item > min_conf  and lift_A_item > min_lift) :
                        antecedents.append(tuple(list(A)))
                        result.append(tuple(list(item))) 
                        insert_text += str(list(A)) + " ---> "+str(list(item)) + " || Confiance : "+ str(confiance_A_item) + " || Lift : "+ str(lift_A_item)+"\n"
                    if(confiance_item_A > min_conf and lift_item_A > min_lift) :
                        antecedents.append(tuple(list(item)))
                        result.append(tuple(list(A)))
                        insert_text += str(list(item)) + " ---> "+str(list(A)) + " || Confiance : "+ str(confiance_item_A) + " || Lift : "+ str(lift_item_A) +"\n"
    inference = dict.fromkeys(antecedents,result)
    print(len(antecedents))
    text.insert(tk.END, insert_text)


def recommend(info):
    top = tk.Toplevel()
    top.geometry("800x600") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.resizable(0, 0) # makes the root window fixed in size.
    top.update()
    top.title('Recomendation')
    top['bg']='#EFF5F5'
    insideFrame1 = tk.LabelFrame(top,text='Display')
    insideFrame1.place(height=600, width=800 , rely=0.05 ,relx=0)
    insideFrame1['bg']='#D6E4E5'

    text = tk.Text(insideFrame1, height=500, width=700)
    scroll = tk.Scrollbar(insideFrame1)
    text.configure(yscrollcommand=scroll.set)
    text.place( rely=0.05 ,relx=0)
    
    insert_text = ""


 #--------------------------------------------- 
    info.replace(' , ' , ',')
    info.replace(', ',',')
    info.replace(' ,',',')  
    test = info.split(',')
    print(test)
    done = []
    insert_text += "Recomended watch list : \n"
    if(len(test) !=0):
        for i in range(1,len(test)+1):
            for item in combinations(test,i) :
                if(item in inference.keys()):
                    
                    for value in inference[item]:
                        if value not in done :
                            if(frozenset(value).issubset(frozenset(test))):
                                continue
                            else:
                                done.append(value)
                                insert_text +=str(value) + "\n" 
                    print(done)
    if(insert_text == "Recomended watch list : \n" ):
        insert_text += "No recomendation"
    text.insert(tk.END, insert_text)
def fun():

    return 0

root = tk.Tk()

root.geometry("1024x400") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0) # makes the root window fixed in size.
root.update()

root.title("Apriori Predictions")
root['bg'] = '#EFF5F5'

frame1 = tk.LabelFrame(root,text='Choosing Min_supp')
frame1.place(height=150, width=450, rely=0.05 ,relx=0.03)
frame1['bg'] ='#D6E4E5'

Label1_= tk.Label(frame1,text="Support Min_Value")
Label1_.place(rely=0.01 , relx = 0.05) 
Label1_['bg']='#D6E4E5'


Entry_min_ = tk.Entry(frame1,width=40 ,font=('Arial 8'))
Entry_min_.place(rely=0.015,relx=0.39)
Entry_min_.insert(0,"Min_Value (Numeric)")

after__btn = tk.Button(frame1,text="Apply appriori", command = lambda : detection_element_freq(Entry_min_.get()))
after__btn.place(rely=0.6 , relx = 0.05,width=150)
after__btn['bg']='#EFF5F5'

after__btn2 = tk.Button(frame1,text="show frequent items", command = lambda : affichageElemFreq())
after__btn2.place(rely=0.6 , relx = 0.55,width=150)
after__btn2['bg']='#EFF5F5'

insideFrame = tk.LabelFrame(root,text='Confidence param')
insideFrame.place(height=150, width=450 , rely=0.05 ,relx=0.5)
insideFrame['bg']='#D6E4E5'

Label1_conf= tk.Label(insideFrame,text="Confidence Min_Value")
Label1_conf.place(rely=0.01 , relx = 0.05) 
Label1_conf['bg']='#D6E4E5'


Entry_min_conf = tk.Entry(insideFrame,width=40 ,font=('Arial 8'))
Entry_min_conf.place(rely=0.015,relx=0.39)
Entry_min_conf.insert(0,"Min_Value (Numeric)")

Label2_conf= tk.Label(insideFrame,text="Correlation :")
Label2_conf.place(rely=0.2 , relx = 0.05) 
Label2_conf['bg']='#D6E4E5'


Entry__conf = tk.Entry(insideFrame,width=40 ,font=('Arial 8'))
Entry__conf.place(rely=0.2,relx=0.39)
Entry__conf.insert(0,"True or False (0,1) otherwise default 0")


after_thresh_btn = tk.Button(insideFrame,text="Generate rules", command = lambda : extractionReglesAssociationetCor(Entry_min_conf.get(),Entry__conf.get()))
after_thresh_btn.place(rely=0.6 , relx = 0.15,width=300)
after_thresh_btn['bg']='#EFF5F5'


frameRecomend = tk.LabelFrame(root,text="Recomendation")
frameRecomend.place(rely=0.5, relx = 0,height=150,width=1024)
frameRecomend['bg']='#D6E4E5'

Label1_rec= tk.Label(frameRecomend,text="Insert informations about the user")
Label1_rec.place(rely=0.01 , relx = 0.1) 
Label1_rec['bg']='#D6E4E5'

Entry_ = tk.Entry(frameRecomend,width=70 ,font=('Arial 8'))
Entry_.place(rely=0.015,relx=0.39)
Entry_.insert(0,"Watching Information ( exmpl : 'People&Blogs' )")

recommend_btn = tk.Button(frameRecomend,text="Show Recomendation", command = lambda : recommend(Entry_.get()))
recommend_btn.place(rely=0.6 , relx = 0.35,width=300)
recommend_btn['bg']='#EFF5F5'

root.mainloop()