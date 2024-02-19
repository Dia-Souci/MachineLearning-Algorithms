import pandas as pd 
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


#define data path 
DATA_PATH = os.path.join('.','data')
ORIGINAL_PATH = os.path.join(DATA_PATH, 'Dataset1.xlsx')
#READING THE FILE

global df 
df= pd.read_excel(ORIGINAL_PATH)
def PartieA_1() :
    print("visualisation")
    print(df)

def PartieA_2(Dataframe):
    print("description du data frame")
    print(Dataframe.info())
    print(Dataframe.describe())

def PartieA_3() :
    for x in df.columns : 
        print("description de l'attribut : ",x)
        print(df[x].info())
        print(df[x].describe())

def PartieA_4_1(attribut,NewVal,OldVal) :
    if(attribut in df.columns) : 
        df[attribut]=df[attribut].replace(OldVal,NewVal)

def PartieA_4_2(attribut,NewVal,index) :
    if(attribut in df.columns) : 
        df[attribut][index]= NewVal


def PartieA_5_Save() :
    OUTPUT_FILE = os.path.join(DATA_PATH, 'output.xlsx')
    df.to_excel(OUTPUT_FILE,header = True)

def PartieB_1() :
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            print('mesures de tendance centrale')
            OutputBar.delete(0,"end")
            mean=df[x].mean()
            mode=df[x].mode()
            median = df[x].median()
            mesures = "Mean : "+ str(mean) + " / Mode : "+ str(mode) + " / Median : "+ str(median) + " / " 
            bool = False
            if(mean > median) :
                if(median > mode[0]) :
                    bool = True 
                    print('column ',x)
                    print()
                    print("positively Skewed")
                    Affichage(mesures+"positively Skewed")
            else : 
                if (median < mode[0]) :
                    bool = True
                    print('column ',x)
                    print()
                    print('negatively Skewed')
                    Affichage(mesures+"negatively Skewed")
            if(bool == False):
                if(mean-median<=1) and (mean-median>=-1) : 
                    if(mode[0]-median<=1) and (mode[0]-median>=-1) :
                        print('column ',x)
                        print()
                        print('Symetric')
                        Affichage(mesures+"Symetric")
            print()
            return None
def PartieB_1_withVar(x) :
        if(df[x].dtype.kind in 'iufcb' ):
            print('mesures de tendance centrale')
            OutputBar.delete(0,"end")
            mean=df[x].mean()
            mode=df[x].mode()
            median = df[x].median()
            mesures = "Mean : "+ str(mean) + " / Mode : "+ str(mode) + " / Median : "+ str(median) + " / " 
            bool = False
            if(mean > median) :
                if(median > mode[0]) :
                    bool = True 
                    print('column ',x)
                    print()
                    print("positively Skewed")
                    Affichage(mesures+"positively Skewed")
            else : 
                if (median < mode[0]) :
                    bool = True
                    print('column ',x)
                    print()
                    print('negatively Skewed')
                    Affichage(mesures+"negatively Skewed")
            if(bool == False):
                if(mean-median<=1) and (mean-median>=-1) : 
                    if(mode[0]-median<=1) and (mode[0]-median>=-1) :
                        print('column ',x)
                        print()
                        print('Symetric')
                        Affichage(mesures+"Symetric")
            print()

def PartieB_2() :
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            print('mesures de dispersion')
            OutputBar.delete(0,"end")
            mesures =""
            ecart_type = df[x].std()
            mesures = mesures + "ecart type : "+ str(ecart_type) +" / "
            print(ecart_type)
            
            Q1= df[x].quantile(0.25)
            mesures = mesures + "Q1 : " + str(Q1) + " / "
            Q3= df[x].quantile(0.75)
            mesures = mesures + "Q3 : " + str(Q3) + " / "
            print(Q1)
            print(Q3)

            IQR =Q3-Q1
            mesures = mesures + "IQR : " + str(IQR) + " / "
            print(IQR)

            lower_lim= Q1-1.5*IQR
            upper_lim = Q3 + 1.5*IQR

            print(lower_lim)
            print(upper_lim)
            Outliers_low = (df[x] < lower_lim )
            Outliers_up = (df[x] > upper_lim )
            if(len(df[x][(Outliers_low|Outliers_up)])>0) :
                mesures = mesures + " On peut trouver des Outliers dans cet attribut "
                print("------------------------------------------------------------")
                print(df[x][(Outliers_low|Outliers_up)])
                print("------------------------------------------------------------")
            else:
                mesures = mesures + " No outliers"
                print("No Outliers in : ",x)
            print()
            Affichage(mesures)


def PartieB_2_withVar(x) :
        if(df[x].dtype.kind in 'iufcb' ):
            print('mesures de dispersion')
            OutputBar.delete(0,"end")
            mesures =""
            ecart_type = df[x].std()
            mesures = mesures + "ecart type : "+ str(ecart_type) +" / "
            print(ecart_type)
            
            Q1= df[x].quantile(0.25)
            mesures = mesures + "Q1 : " + str(Q1) + " / "
            Q3= df[x].quantile(0.75)
            mesures = mesures + "Q3 : " + str(Q3) + " / "
            print(Q1)
            print(Q3)

            IQR =Q3-Q1
            mesures = mesures + "IQR : " + str(IQR) + " / "
            print(IQR)

            lower_lim= Q1-1.5*IQR
            upper_lim = Q3 + 1.5*IQR

            print(lower_lim)
            print(upper_lim)
            Outliers_low = (df[x] < lower_lim )
            Outliers_up = (df[x] > upper_lim )
            if(len(df[x][(Outliers_low|Outliers_up)])>0) :
                mesures = mesures + " On peut trouver des Outliers dans cet attribut "
                print("------------------------------------------------------------")
                print(df[x][(Outliers_low|Outliers_up)])
                print("------------------------------------------------------------")
            else:
                mesures = mesures + " No outliers"
                print("No Outliers in : ",x)
            print()
            Affichage(mesures)

def simplify(x) :
    if(df[x].dtype.kind in 'iufcb' ):
        Q1= df[x].quantile(0.25)
        Q3= df[x].quantile(0.75)
        IQR =Q3-Q1
        lower_lim= Q1-1.5*IQR
        upper_lim = Q3 + 1.5*IQR
        Outliers_low = (df[x] < lower_lim )
        Outliers_up = (df[x] > upper_lim )
        return Outliers_low , Outliers_up

def simplify_(dataFrame) :
    for x in dataFrame.columns :
        if(dataFrame[x].dtype.kind in 'iufcb' ):
            Q1= dataFrame[x].quantile(0.25)
            Q3= dataFrame[x].quantile(0.75)
            IQR =Q3-Q1
            lower_lim= Q1-1.5*IQR
            upper_lim = Q3 + 1.5*IQR
            Outliers_low = (dataFrame[x] < lower_lim )
            Outliers_up = (dataFrame[x] > upper_lim )
            dataFrame=dataFrame[~(Outliers_low | Outliers_up)]
            return dataFrame

def boxplotWithOutliers() : 
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            sns.boxplot(x=df[x])
            plt.show()

def boxplotfordisplay1(attr) :
    if(df[attr].dtype.kind in 'iufcb' ):
            sns.boxplot(x=df[attr])
            plt.show()



def boxplotWithout_Outliers(df) :
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            Outliers_low,Outliers_up=simplify(x)
            df=df[~(Outliers_low | Outliers_up)]
            sns.boxplot(x=df[x])
            plt.show()


def boxplotfordisplay2(df,attr) :
    if(df[attr].dtype.kind in 'iufcb' ):
            Outliers_low,Outliers_up=simplify(attr)
            df=df[~(Outliers_low | Outliers_up)]
            sns.boxplot(x=df[attr])
            plt.show()

def histogramWithOutliers() : 
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            sns.distplot(df[x],bins =10 , kde=False)
            plt.show()

def histogramWithOutliers1(attr) : 
        if(df[attr].dtype.kind in 'iufcb' ):
            sns.distplot(df[attr],bins =10 , kde=False)
            plt.show()

def histograWithout_Outliers(df) :
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            Outliers_low,Outliers_up=simplify(x)
            df=df[~(Outliers_low | Outliers_up)]
            sns.distplot(df[x],bins =10 , kde=False)
            plt.show()


def histograWithout_Outliers1(df,attr) :
        if(df[attr].dtype.kind in 'iufcb' ):
            Outliers_low,Outliers_up=simplify(attr)
            df=df[~(Outliers_low | Outliers_up)]
            sns.distplot(df[attr],bins =10 , kde=False)
            plt.show()


def ScatterPlotting(attr1,attr2) : 
    sns.scatterplot(df,x=attr1,y=attr2)
    plt.show()


def Affichage(str) :
    OutputBar.insert(0,str)

#-----------------------Part2----------------------------------------------------------------------------------

#------------------------------A---------------------------------------------------
def CheckHasNull(x) :
    test = False
    if (df[x].count()<1470) :
        print(f"the column {x} has null values")
        test = True
    if (test == False) : 
        print("The Data frame has no Null Values")
    return test

#--------------------------------a---------------------------------------

def fillNaAutoMean(x):
    mean= df[x].mean()
    df[x].fillna(mean,inplace=True)

def fillNaAutoMedian(x):
    median= df[x].median()
    df[x].fillna(median,inplace=True)

def fillNaAutoMode(x):
    mode= df[x].mode()
    df[x].fillna(mode,inplace=True)

def fillNaAutoHotDeck(x):
   i = np.random.randint(df[x].count())
   val = df[x][i]
   df[x].fillna(val,inplace=True)

def fillNaAuto():
    for x in df.columns :
        boolVar = CheckHasNull(x)
        if(boolVar == True) :
            fillNaAutoMean(x)

def fillNaAutoColdDeck(x,index):
    val = df[x][int(index)]
    df[x].fillna(val,inplace=True)

def fillNaAuto():
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):    
            boolVar = CheckHasNull(x)
            if(boolVar == True) :
                fillNaAutoMean(x)
        else :
            fillNaAutoMode(x)

fillNaAuto()
value =df["EnvironmentSatisfaction"][8]
print(value)
#----------------------------------b---------------------------------------
def DropOutliers(x) :
    if(df[x].dtype.kind in 'iufcb' ):
        Q1= df[x].quantile(0.25)
        Q3= df[x].quantile(0.75)
        IQR =Q3-Q1
        lower_lim= Q1-1.5*IQR
        upper_lim = Q3 + 1.5*IQR
        Outliers_low = (df[x] < lower_lim )
        Outliers_up = (df[x] > upper_lim )
        df=df[~(Outliers_low | Outliers_up)]
        return df

def DropOutliers_Calculation(df,x) :
    if(df[x].dtype.kind in 'iufcb' ):
        Q1= df[x].quantile(0.25)
        Q3= df[x].quantile(0.75)
        IQR =Q3-Q1
        lower_lim= Q1-1.5*IQR
        upper_lim = Q3 + 1.5*IQR
        Outliers_low = (df[x] < lower_lim )
        Outliers_up = (df[x] > upper_lim )
        df=df[~(Outliers_low | Outliers_up)]
        return df,Outliers_low,Outliers_up

def replaceMoyenneTranque(x):
    if(df[x].dtype.kind in 'iufcb' ):
        df2 = df.copy()
        df2 , O_low , O_up = DropOutliers_Calculation(df2,x)
        mean = df2[x].mean()
        for y in df[x][(O_low | O_up)] :
            df[x].replace(y,mean)
    else : 
        print("Non numerical Data")
    

def replaceMedianeTranque(x):
    if(df[x].dtype.kind in 'iufcb' ):
        df2 = df.copy()
        df2 , O_low , O_up = DropOutliers_Calculation(df2,x)
        median = df2[x].median()
        for y in df[x][(O_low | O_up)] :
            df[x].replace(y,median)
    else : 
        print("Non numerical Data")

def replaceModeTranque(x):    
    df2 = df.copy()
    df2 , O_low , O_up = DropOutliers_Calculation(df2,x)
    mode = df2[x].mode()
    for y in df[x][(O_low | O_up)] :
               # print (y)
                df[x].replace(y,mode,inplace=True)


def replaceMoyenneTranque2():
    for x in df.columns:    
        if(df[x].dtype.kind in 'iufcb' ):
            df2 = df.copy()
            df2 , O_low , O_up = DropOutliers_Calculation(df2,x)
            #mean1 = df[x].mean()
            mean = df2[x].mean()
            #print(mean1)
            #print(mean)
            for y in df[x][(O_low | O_up)] :
               # print (y)
                if(x=="MonthlyIncome"):
                    print(df[x][O_low])
                df[x].replace(y,mean,inplace=True)
        else : 
            print("Non numerical Data")
        
#replaceMoyenneTranque()



#-------------------------------------------B---------------------------------------------

def BinningQcut(x):
    binLable =["A","B","C","D"]
    df["Quantile"] = pd.qcut(df[x],q=4,labels=binLable)
    for y in binLable :
        df2= df[df.Quantile == y]
        mean = df2[x].mean()
        print(mean)
        df["Quantile"].replace(y,mean,inplace=True)
    df.drop(x,inplace=True , axis=1)
    df.rename(columns={"Quantile" : x} , inplace=True)

def BinningEqualHeightCount(x):
    binLable =["A","B","C","D","E","F"]
    df["EquiHeight"] = pd.cut(df[x],bins=6,labels=binLable)
    for y in binLable :
        df2= df[df.EquiHeight == y]
        mean = df2[x].mean()
        print(mean)
        df["EquiHeight"].replace(y,mean,inplace=True)
    df.drop(x,inplace=True , axis=1)
    df.rename(columns={"EquiHeight" : x} , inplace=True)

#BinningEqualHeightCount("Age")

#------------------------------------C--------------------------------------------

def CleanData():
    df.drop_duplicates(inplace=True)
    
    for x in df.columns :
        test = False
        if(df[x].nunique()==1) :
            df.drop(x,inplace=True,axis=1)
            test = True
        if(test==False):
            if(df[x].nunique()==1470):
                df.drop(x,inplace=True,axis=1)
                test= True
            if(test==False):
                if (x=="Attrition"):
                    df[x].replace(["No","Yes"],[0,1],inplace=True)
                if (x=="BusinessTravel"):
                    df[x].replace(["Non-Travel","Travel_Rarely","Travel_Frequently"],[0,1,2],inplace=True)
                if (x=="Gender"):
                    df[x].replace(["Female","Male"],[0,1],inplace=True)
                if (x=="OverTime"):
                    df[x].replace(["No","Yes"],[0,1],inplace=True)
                if (x=="MaritalStatus"):
                    df[x].replace(["Single","Married","Divorsed"],[0,1,2],inplace=True)
                if(df[x].dtype.kind not in 'iufcb'):
                    df.drop(x,inplace=True,axis=1)
    print("Done ! ")                
        

#-------------------------------------D---------------------------------------------------

#-----------------------------------Normalizaton minmax-----------

def min_max():
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            df[x] = (df[x] - df[x].min()) / (df[x].max() - df[x].min())


#-----------------------------------Normalisation Z-Score -------------------
def z_score():
    for x in df.columns :
        if(df[x].dtype.kind in 'iufcb' ):
            df[x] = (df[x] - df[x].mean()) / df[x].std() 





#------------------------------------Normalization z-score







#---------------------------------Interface--------------Skip it----------------------------------------------

# initalise the tkinter GUI
root = tk.Tk()

root.geometry("800x600") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0) # makes the root window fixed in size.

top = tk.Toplevel()
top.geometry("800x600")
top.resizable(0, 0) 

top2 = tk.Toplevel()
top2.geometry("800x600")
top2.resizable(0, 0)
#-----------------------------------------------------------------------------------------------
frame_missingvalues = tk.LabelFrame(top2,text='Missing Values')
frame_missingvalues.place(height=100, width=800, rely=0.02 ,relx=0)

inputt_attribut = tk.Entry(frame_missingvalues,width=40 )
inputt_attribut.place(rely=0.1,relx=0.35)

button_fill_2 = tk.Button(frame_missingvalues , text = "Fill with mean" , command = lambda :fillNaAutoMean(inputt_attribut.get()))
button_fill_2.place(rely=0.5,relx=0.05)


button_fill_3 = tk.Button(frame_missingvalues , text = "Fill with median" , command = lambda :fillNaAutoMedian(inputt_attribut.get()))
button_fill_3.place(rely=0.5,relx=0.2)

button_fill_4 = tk.Button(frame_missingvalues , text = "Fill with mode" , command = lambda :fillNaAutoMode(inputt_attribut.get()))
button_fill_4.place(rely=0.5,relx=0.35)

button_fill_5 = tk.Button(frame_missingvalues , text = "Fill with HotDeck" , command = lambda :fillNaAutoHotDeck(inputt_attribut.get()))
button_fill_5.place(rely=0.5,relx=0.5)

input_index = tk.Entry(frame_missingvalues,width=15 )
input_index.place(rely=0.5,relx=0.8)

button_fill_6 = tk.Button(frame_missingvalues , text = "Fill with ColdDeck" , command = lambda :fillNaAutoColdDeck(inputt_attribut.get() , input_index.get()))
button_fill_6.place(rely=0.5,relx=0.65)


frame_Outliers = tk.LabelFrame(top2,text='Outliers Treatment')
frame_Outliers.place(height=100, width=800, rely=0.25 ,relx=0)

inputt_attribut_2 = tk.Entry(frame_Outliers,width=40 )
inputt_attribut_2.place(rely=0.1,relx=0.35)

button_drop_2 = tk.Button(frame_Outliers , text = "Drop Outliers" , command = lambda :DropOutliers(inputt_attribut_2.get()))
button_drop_2.place(rely=0.5,relx=0.05)

button_drop_3 = tk.Button(frame_Outliers , text = "Substitute with Mean" , command = lambda :replaceMoyenneTranque(inputt_attribut_2.get()))
button_drop_3.place(rely=0.5,relx=0.25)

button_drop_4 = tk.Button(frame_Outliers , text = "Substitute with Median" , command = lambda :replaceMedianeTranque(inputt_attribut_2.get()))
button_drop_4.place(rely=0.5,relx=0.45)

button_drop_5 = tk.Button(frame_Outliers , text = "Substitute with Mode" , command = lambda :replaceModeTranque(inputt_attribut_2.get()))
button_drop_5.place(rely=0.5,relx=0.65)



frame_Binning = tk.LabelFrame(top2,text='Binning Data')
frame_Binning.place(height=100, width=800, rely=0.5 ,relx=0)

inputt_attribut_3 = tk.Entry(frame_Binning,width=40 )
inputt_attribut_3.place(rely=0.1,relx=0.35)

button_Binning_1 = tk.Button(frame_Binning , text = "Binning Qcut" , command = lambda :BinningQcut(inputt_attribut_3.get()))
button_Binning_1.place(rely=0.5,relx=0.25)

button_Binning_2 = tk.Button(frame_Binning , text = "Binning Equal Cuts" , command = lambda :BinningEqualHeightCount(inputt_attribut_3.get()))
button_Binning_2.place(rely=0.5,relx=0.65)



frame_Cleaning = tk.LabelFrame(top2,text='Cleaning Data')
frame_Cleaning.place(height=100, width=800, rely=0.65 ,relx=0)

button_clean = tk.Button(frame_Cleaning , text = "Clean data" , command = lambda :CleanData())
button_clean.place(rely=0.5,relx=0.5)




frame_Normalization = tk.LabelFrame(top2,text='Normalizing Data')
frame_Normalization.place(height=100, width=800, rely=0.8 ,relx=0)


button_Norm_1 = tk.Button(frame_Normalization , text = "Z-Score" , command = lambda :z_score())
button_Norm_1.place(rely=0.5,relx=0.25)

button_Norm_2 = tk.Button(frame_Normalization , text = "Min-Max" , command = lambda :min_max())
button_Norm_2.place(rely=0.5,relx=0.65)

#------------------------------------------------------------------------------------------------
frame_with_outliers = tk.LabelFrame(top,text='Ploting data with Outliers')
frame_with_outliers.place(height=150, width=800, rely=0.05 ,relx=0)

frame_without_outliers = tk.LabelFrame(top,text='Ploting data without Outliers')
frame_without_outliers.place(height=150, width=800 , rely=0.35 ,relx=0)

frame_corelation = tk.LabelFrame(top,text='Data Corelation')
frame_corelation.place(height=150, width=800 , rely=0.65 ,relx=0)

inputt = tk.Entry(frame_corelation,width=20 )
inputt.place(rely=0.2,relx=0.05)

inputt2 = tk.Entry(frame_corelation,width=20 )
inputt2.place(rely=0.4,relx=0.05)

button_plot = tk.Button(frame_corelation , text = "Scatter Plot & Corelation" , command = lambda :ScatterPlotting(inputt.get(),inputt2.get()))
button_plot.place(rely=0.6,relx=0.05)


# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Excel Data")
frame1.place(height=300, width=800)

# Frame for open file dialog
file_frame = tk.LabelFrame(root, text="Visualisation")
file_frame.place(height=100, width=800, rely=0.5, relx=0)

symetry_frame =tk.LabelFrame(root,text='Symetrie ! Etude des Mesures centrale')
symetry_frame.place(height=100,width=800,rely=0.75,relx=0)

OutputBar=tk.Entry(symetry_frame, width=400)
OutputBar.place(rely=0.25,relx=0)

inputField = tk.Entry(symetry_frame,width = 20 )
inputField.place(rely=0.5,relx=0.2)

buttonShow=tk.Button(symetry_frame,text="Show Symetry" , command = lambda : PartieB_1_withVar(inputField.get()))
buttonShow.place(rely=0.5 , relx=0.5)

button_scutter = tk.Button(symetry_frame,text="Show Scutter" , command = lambda : PartieB_2_withVar(inputField.get()))
button_scutter.place(rely=0.5 , relx = 0.8)



# frame_with_outliers-----------------------------------------------------------------------------------------------------------
button_1 = tk.Button(frame_with_outliers, text="Boxplot_ALL", command=lambda: boxplotWithOutliers())
button_1.place(rely=0.6, relx=0.45)



inputtxt =tk.Entry(frame_with_outliers,width = 20)
inputtxt.place(rely=0.6, relx=0.05)
button_2 = tk.Button(frame_with_outliers, text="Boxplot", command=lambda: boxplotfordisplay1(inputtxt.get()))
button_2.place(rely=0.6, relx=0.25)

#button_3 = tk.Button(frame_top,text="Boxplot", command=lambda: boxplotfordisplay2(df,inputtxt.get()))

button_1_1 = tk.Button(frame_with_outliers, text="Histplot_ALL", command=lambda: histogramWithOutliers())
button_1_1.place(rely=0.25, relx=0.45)



inputtxt3 =tk.Entry(frame_with_outliers,width = 20)
inputtxt3.place(rely=0.25, relx=0.05)
button_2_1 = tk.Button(frame_with_outliers, text="Histplot", command=lambda: histogramWithOutliers1(inputtxt3.get()))
button_2_1.place(rely=0.25, relx=0.25)

#frame_without_outliers-------------------------------------------------------------------------------------------------------
buttons_1 = tk.Button(frame_without_outliers, text="Boxplot_ALL", command=lambda: boxplotWithout_Outliers(df))
buttons_1.place(rely=0.6, relx=0.45)



inputtxts =tk.Entry(frame_without_outliers,width = 20)
inputtxts.place(rely=0.6, relx=0.05)
buttons_2 = tk.Button(frame_without_outliers, text="Boxplot", command=lambda: boxplotfordisplay2(df,inputtxts.get()))
buttons_2.place(rely=0.6, relx=0.25)

#button_3 = tk.Button(frame_top,text="Boxplot", command=lambda: boxplotfordisplay2(df,inputtxt.get()))

buttons_1_1 = tk.Button(frame_without_outliers, text="Histplot_ALL", command=lambda: histograWithout_Outliers(df))
buttons_1_1.place(rely=0.25, relx=0.45)



inputtxts3 =tk.Entry(frame_without_outliers,width = 20)
inputtxts3.place(rely=0.25, relx=0.05)
buttons_2_1 = tk.Button(frame_without_outliers, text="Histplot", command=lambda: histograWithout_Outliers1(df,inputtxts3.get()))
buttons_2_1.place(rely=0.25, relx=0.25)



inputtxt2 =tk.Entry(file_frame,width = 10)
inputtxt2.place(rely=0.65, relx=0.40)
button1 = tk.Button(file_frame, text="Load Portion data", command=lambda: Load_portion_excel_data(inputtxt2.get()))
button1.place(rely=0.65, relx=0.50)



button2 = tk.Button(file_frame, text="Load Full data", command=lambda: Load_excel_data())
button2.place(rely=0.65, relx=0.10)



## Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget
def test():
    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)
    tv1.insert("","end",values=df.describe)



def Load_excel_data():
    """If the file selected is valid this will load the file into the Treeview"""
    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
    return None

def Load_portion_excel_data(attr):
    """If the file selected is valid this will load the file into the Treeview"""
    clear_data()
    df1 = df[[attr]].copy()
    tv1["column"] = list(df1.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)
    for x in df1.to_numpy().tolist() :
        tv1.insert("", "end", values= x)
        #pd_variable.set(pd['Age'].describe())
        

    return None

def clear_data():
    tv1.delete(*tv1.get_children())
    return None


root.mainloop()