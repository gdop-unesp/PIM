#Bibliotecas 
# https://docs.python.org/3/tutorial/venv.html

import pandas as pd
import numpy as np
import pathlib
import os
import shutil
import sys
import matplotlib.pyplot as plt
from datetime import date, datetime
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET 
import xmltodict
import createFileInput 
import variaveisAnaliseDinamica 


diretorio = str(pathlib.Path().home().resolve()) + "/documentos/PIM/"

arquivoDados = 'pastas.txt'


cam = df['camera'].to_list()
camFile =  []
for i in cam:
  for j in cam:
    if j[0:3] != i[0:3]:
      if j!=i:
        camFile.append(i+"_"+j)
        camFile.append(j+"_"+i)
camFile = list(dict.fromkeys(camFile))
print(camFile)

"""Dados orbitais"""

data = []
dfOut = pd.DataFrame(data, columns = ['pair', 'vel','a','e','i','w','W','f'])

def lerDataNum(dataText):
  dataTextSep= dataText.split(':')
  return dataTextSep[1]


for arqOut in camFile:
  lerArq = open(diretorio2+'/'+dirRun+arqOut+'/dados.txt','r')
  text = lerArq.read()
  lerArq.close()
  textLine = text.split("\n")
  if "using cam = 1" in text:
    camUsing = 'cam1'   
  elif "using cam = 2" in text:
    camUsing = 'cam2'
  print(arqOut)
  for numLine,item in enumerate(textLine):
    if ('velocidade meteoro '+camUsing) in item:
      velLine = numLine+1
      velArq = float(textLine[velLine])
    if ('semi-major axis (au) :') in item:
      aArq = float(lerDataNum(item))
    if ('eccentriciy :') in item:
      eArq = float(lerDataNum(item))
    if ('inclination :') in item:
      iArq = float(lerDataNum(item))
    if ('perige :') in item:
      wArq = float(lerDataNum(item))
    if ('node :') in item:
      WArq = float(lerDataNum(item))
    if ('true anomaly :') in item:
      fArq = float(lerDataNum(item))
  camOut = [arqOut,velArq,aArq,eArq,iArq,wArq,WArq,fArq]

  dfOut.loc[len(dfOut)] = camOut
#display(dfOut)

dfOut.to_excel(dirGr2+'/'+dirRun+"finalDataPair.xls")
# print(dfOut.info())
meanOrbIni = []
param = ['a','e','i']
for calc in param:
  print(calc,' mean ',dfOut[calc].mean())
  print(calc,' std ',dfOut[calc].std())
  print(calc,' max ',dfOut[calc].max())
  print(calc,' min ',dfOut[calc].min())
  print(calc,' max-min ',dfOut[calc].max() - dfOut[calc].min())
  print('-----------------')
  meanOrbIni.append(dfOut[calc].mean())
for calc in param:
  print(f'{calc} = {dfOut[calc].mean():.3f} +/- {dfOut[calc].std():.3f}')

"""Encontros próximos"""

#encontros próximos
earthLine=[]
venusLine=[]
marsLine=[]
colunasPlot = []
labelPlot = []
cor = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
uaLua = 149597870.70/384400
listaP=['meteor','EARTH','MOON','MARS','VENUS']
listaHill = [0,0.0098,0,0.0066,0.0067]
tamHill = 5
camLineVEarth = []
for i,arqOut in enumerate(camFile):
  labelC = arqOut[0:3]+arqOut[7:-4]
  labelPlot.append(labelC)

  camLineVEarth.append([])
  meteorData = pd.read_csv(dirGr2+'/'+dirRun+arqOut+'/meteor.aei',sep=" ")
  earthData = pd.read_csv(dirGr2+'/'+dirRun+arqOut+'/Earth.aei',sep=" ")
  marsData = pd.read_csv(dirGr2+'/'+dirRun+arqOut+'/Mars.aei',sep=" ")
  venusData = pd.read_csv(dirGr2+'/'+dirRun+arqOut+'/Venus.aei',sep=" ")

  #isplay(earthData)
  meteorData['r']=np.sqrt((meteorData['x']**2)+(meteorData['y']**2)+(meteorData['z']**2))
  #Encontrar minima aproximacao Terra e Marte
  meteorXYZ=meteorData[['Time(y)','x','y','z']]
  colunas = ['Time(y)','xM','yM','zM']
  meteorXYZ.columns = colunas

  meteorXYZ = pd.concat([meteorXYZ,earthData[['x','y','z']]],axis=1)
  colunas.extend(['xE','yE','zE'])
  meteorXYZ.columns = colunas
  meteorXYZ = pd.concat([meteorXYZ,marsData[['x','y','z']]],axis=1)
  colunas.extend(['xMa','yMa','zMa'])
  meteorXYZ.columns = colunas
  meteorXYZ = pd.concat([meteorXYZ,venusData[['x','y','z']]],axis=1)
  colunas.extend(['xV','yV','zV'])
  meteorXYZ.columns = colunas

  meteorXYZ["distEarth"] =uaLua*np.sqrt((meteorXYZ['xM']-meteorXYZ['xE'])**2+(meteorXYZ['yM']-meteorXYZ['yE'])**2+(meteorXYZ['zM']-meteorXYZ['zE'])**2)
  meteorXYZ["distMars"] =uaLua*np.sqrt((meteorXYZ['xM']-meteorXYZ['xMa'])**2+(meteorXYZ['yM']-meteorXYZ['yMa'])**2+(meteorXYZ['zM']-meteorXYZ['zMa'])**2)
  meteorXYZ["distVenus"]=uaLua*np.sqrt((meteorXYZ['xM']-meteorXYZ['xV'])**2+(meteorXYZ['yM']-meteorXYZ['yV'])**2+(meteorXYZ['zM']-meteorXYZ['zV'])**2)
  # eartEnconter = eartEnconter.concat(meteorXYZ['Time(y)'])
  if (i == 0):
    eartEnconter = meteorXYZ[['Time(y)','distEarth']]
  else:
    eartEnconter = pd.concat([eartEnconter,meteorXYZ[['Time(y)','distEarth']]],axis=1)
  eartEnconter = eartEnconter.rename({'distEarth': 'distEarth'+arqOut,'Time(y)' : 'Time(y)'+arqOut}, axis=1)
  colunasPlot.append('distEarth'+arqOut)




  limT=-0.5

#  graphDist=meteorXYZ.plot( x='Time(y)', y=['distEarth','distMars','distVenus'],
                          #  figsize=(10,5),fontsize=16,color=['r','y','c'],
                          #  ylabel="d (lunar distance)")
  graphDist=meteorXYZ.plot( x='Time(y)', y=['distEarth'],
                           figsize=(10,5),fontsize=16,
                           ylabel="d (lunar distance)",label = ['Meteor-Earth distance'],color = [cor[i]])
  graphDist.set_ylim(0,1.1*tamHill*listaHill[1]*uaLua)
  graphDist.set_xlim(right=limT)
  graphDist.xaxis.get_label().set_fontsize(16)
  graphDist.yaxis.get_label().set_fontsize(16)
  plt.title(labelC)
  plt.savefig(dirGr2+'/'+dirRun+arqOut+'/'+'encounters.png')
  plt.savefig(dirGr2+'/'+dirRun+'/'+arqOut+'_encounters.png')
  
  
  closeP=pd.DataFrame()
  closeP['distEarth'] = meteorXYZ.distEarth[(meteorXYZ.distEarth.shift(1) > meteorXYZ.distEarth) & (meteorXYZ.distEarth.shift(-1) > meteorXYZ.distEarth)]
  mergedStuff = pd.merge(closeP[['distEarth']], meteorXYZ[['Time(y)','distEarth']],on='distEarth')
  mergedStuff=mergedStuff[mergedStuff['distEarth']<tamHill*listaHill[1]*uaLua]
  earthLine = earthLine + mergedStuff['Time(y)'].tolist()
  camLineVEarth[i] = camLineVEarth[i] + mergedStuff['Time(y)'].tolist()

  closeP=pd.DataFrame()
  closeP['distMars'] = meteorXYZ.distMars[(meteorXYZ.distMars.shift(1) > meteorXYZ.distMars) & (meteorXYZ.distMars.shift(-1) > meteorXYZ.distMars)]
  mergedStuff = pd.merge(closeP[['distMars']], meteorXYZ[['Time(y)','distMars']],on='distMars')
  mergedStuff=mergedStuff[mergedStuff['distMars']<tamHill*listaHill[3]*uaLua]
  eartLine=[]
  marsLine = marsLine+ mergedStuff['Time(y)'].tolist()


  closeP=pd.DataFrame()
  closeP['distVenus'] = meteorXYZ.distVenus[(meteorXYZ.distVenus.shift(1) > meteorXYZ.distVenus) & (meteorXYZ.distVenus.shift(-1) > meteorXYZ.distVenus)]
  mergedStuff = pd.merge(closeP[['distVenus']], meteorXYZ[['Time(y)','distVenus']],on='distVenus')
  mergedStuff=mergedStuff[mergedStuff['distVenus']<tamHill*listaHill[4]*uaLua]
  venusLine = venusLine + mergedStuff['Time(y)'].tolist()

#display(eartEnconter)
graphEarthEnconter=eartEnconter.plot(x='Time(y)'+camFile[0], y=colunasPlot,figsize=(10,5),fontsize=16,linewidth=3,
                                     ylabel="d (lunar distance)",label = labelPlot,color = cor,xlabel = 'Time (y)')
graphEarthEnconter.set_ylim(0,1.1*tamHill*listaHill[1]*uaLua)
graphEarthEnconter.set_xlim(right=limT)
graphEarthEnconter.xaxis.get_label().set_fontsize(16)
graphEarthEnconter.yaxis.get_label().set_fontsize(16)
# plt.title("Earth ")
graphEarthEnconter.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
              ncol=3, fancybox=True, shadow=True,fontsize=12)
plt.savefig(dirGr2+'/'+dirRun+arqOut+'/'+'encountersEarth.png')
plt.savefig(dirGr2+'/'+dirRun+'/'+arqOut+'_encountersEarth.png')


#graficos de encontros com a Terra
# for i,graficar in enumerate(colunasGraph):
#   # labelC = list(map(lambda x : x[1:], listaG[i][:-1]))
#   # labelC.append(listaG[i][-1][0:-1])
#   labelC = list(map(lambda x : x[1:], listaG[i]))
#   labelC = list(map(lambda x : x[0:3]+x[7:-4], labelC))
#   graphA=dfM.plot(x="Time(y)"+camFile[1], y=listaG[i],figsize=(10,5),fontsize=16,ylabel=colunasNome[i],
#                   linewidth=3,label=labelC,color=cor)
#   plt.plot(timeList,meanList[i],label='MEAN',linewidth=5.0,color="black")
#   graphA.set_xlim(right=0)
#   # graphA.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#   graphA.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#           ncol=4, fancybox=True, shadow=True,fontsize=12)
#   if i>0:
#     graphA.get_legend().remove()
#   plt.xlabel('Time (years)', fontsize=18)
#   plt.yticks(fontsize=16)
#   plt.xticks(fontsize=16)
#   graphA.xaxis.get_label().set_fontsize(16)
#   graphA.yaxis.get_label().set_fontsize(16)

dfM = pd.DataFrame([])
listaG = [[],[],[]]
for i,arqOut in enumerate(camFile):
  df = pd.read_csv(dirGr2+'/'+dirRun+arqOut+'/meteor.aei',sep=" ")
  df = df.drop(["peri","node","f","x","y","z"],axis=1)
  columnName = ["Time(y)"+arqOut,"a"+arqOut,"e"+arqOut,"i"+arqOut]
  listaG[0].append("a"+arqOut)
  listaG[1].append("e"+arqOut)
  listaG[2].append("i"+arqOut)
  #display(df)
  df.columns = columnName
  dfM = pd.concat([dfM,df],axis=1)

dfM['meanA'] = dfM[listaG[0]].mean(axis=1)
dfM['meanE'] = dfM[listaG[1]].mean(axis=1)
dfM['meanI'] = dfM[listaG[2]].mean(axis=1)

dfM=dfM[dfM["Time(y)"+camFile[0]]<-0.5]

# listaG[0].append('meanA')
# listaG[1].append('meanE')
# listaG[2].append('meanI')
colunasGraph=["a","e","i"]
colunasNome=['a (au)','e','i ('+u"\N{DEGREE SIGN}"+')']
arq=['semiMajor','exc','inc']
timeList = dfM['Time(y)'+camFile[0]].to_list()
meanList = [dfM['meanA'].to_list(),dfM['meanE'].to_list(),dfM['meanI'].to_list()]
plt.figure()
cor = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
for i,graficar in enumerate(colunasGraph):
  # labelC = list(map(lambda x : x[1:], listaG[i][:-1]))
  # labelC.append(listaG[i][-1][0:-1])
  labelC = list(map(lambda x : x[1:], listaG[i]))
  labelC = list(map(lambda x : x[0:3]+x[7:-4], labelC))
  graphA=dfM.plot(x="Time(y)"+camFile[1], y=listaG[i],figsize=(10,5),fontsize=16,ylabel=colunasNome[i],
                  linewidth=3,label=labelC,color=cor)
  plt.plot(timeList,meanList[i],label='MEAN',linewidth=5.0,color="black")
  graphA.set_xlim(right=0)
  # graphA.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  graphA.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4, fancybox=True, shadow=True,fontsize=12)
  if i>0:
    graphA.get_legend().remove()
  plt.xlabel('Time (years)', fontsize=18)
  plt.yticks(fontsize=16)
  plt.xticks(fontsize=16)
  graphA.xaxis.get_label().set_fontsize(16)
  graphA.yaxis.get_label().set_fontsize(16)
  # for xc in earthLine:
  #   graphA.axvline(x=xc,color='r',linestyle='dashed',linewidth=1)
  # for xc in marsLine:
  #   graphA.axvline(x=xc,color='y',linestyle='dashed',linewidth=1)
  # for xc in venusLine:
  #   graphA.axvline(x=xc,color='c',linestyle='dashed',linewidth=1)
  for j,arqOut in enumerate(camFile):
    for k in camLineVEarth[j]:
      graphA.axvline(x=k,linestyle='dashed',linewidth=1.2,color=cor[j])

  # plt.tight_layout()
  plt.savefig(dirGr2+'/'+dirRun+ arq[i]+'.png')

"""Médias, máximos e mínimos no tempo"""

for i,arqOut in enumerate(camFile):
  dfArq = pd.read_csv(dirGr2+'/'+dirRun+arqOut+'/meteor.aei',sep=" ")
  dfArq = dfArq.drop(["peri","node","f","x","y","z"],axis=1)
  indexTime = dfArq[ (dfArq['Time(y)'] > -1)].index
  dfArq.drop(indexTime , inplace=True)
  if i==0:
    dfSt=dfArq
  else:
    dfSt = dfSt.append(dfArq,ignore_index = True)
print(dfSt.info())
#display(dfSt)
param = ['a','e','i']
for calc in param:
  print(calc,' mean ',dfSt[calc].mean())
  print(calc,' std ',dfSt[calc].std())
  print(calc,' max ',dfSt[calc].max())
  print(calc,' min ',dfSt[calc].min())
  print(calc,' max-min ',dfSt[calc].max() - dfSt[calc].min())
  print('-----------------')

"""Comparação com asteroides"""

#ler listas de asteroides
arqAst = open(str(pathlib.Path().resolve()) + "/"+"drive/MyDrive/Colab Notebooks/MPCORB.DAT",'r')
arqAstF = open(str(pathlib.Path().resolve()) + "/"+"drive/MyDrive/Colab Notebooks/MPCORB_Orbital_Elements.DAT",'w')
listaAst = arqAst.read()
listaAst= listaAst.split('\n')
for i,asteroid in enumerate(listaAst):
  astSimp = asteroid[26:104] 
  arqAstF.write(astSimp + '\n')
arqAst.close()
arqAstF.close()

aMG = meanOrbIni[0]
eMG = meanOrbIni[1]
iMG = meanOrbIni[2]
#filtrar dados
pMPCORB = pd.read_csv(str(pathlib.Path().resolve()) + "/"+"drive/MyDrive/Colab Notebooks/MPCORB_Orbital_Elements.DAT", delim_whitespace=True,on_bad_lines='warn')
newMPCORB = pMPCORB[pMPCORB['a'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
#pd.read_csv("whitespace.csv", header=None, delimiter=r"\s+")

#graficar


graphA=newMPCORB.plot(x="a", y="e",figsize=(10,5),fontsize=16,ylabel="Eccentricity",xlabel = "Semi-major axis (au)",
                      linestyle = 'None',marker='o',markersize=0.5,color='gray',xlim=(1.,5.5),ylim=(0.,1.),label='_nolegend_')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Semi-major axis (au)", fontsize=18)
plt.ylabel("Eccentricity", fontsize=18)
plt.plot([aMG],[eMG],marker='o',markersize=15,color='red')
plt.legend(frameon=False)

#plotar  Grupo Hungarian
hung = newMPCORB[(newMPCORB['a']>1.78) & (newMPCORB['a']<2.) & (newMPCORB['e']<0.18) & (newMPCORB['Incl.']>16.) & (newMPCORB['Incl.']<34) ]
#plt.plot(hung['a'].to_list(),hung['e'].to_list(),marker='o',markersize=0.5,color='blue')

plt.tight_layout()
plt.savefig(dirGr2+'/'+dirRun+ 'excMPCORB.png')
### inclinacao
graphA=newMPCORB.plot(x="a", y="Incl.",figsize=(10,5),fontsize=16,ylabel="Eccentricity",xlabel = "Semi-major axis (au)",
                      linestyle = 'None',marker='o',markersize=0.5,color='gray',xlim=(1.,5.5),ylim=(0.,30),label='_nolegend_')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Semi-major axis (au)", fontsize=18)
plt.ylabel("Inclination ("+u"\N{DEGREE SIGN}"+")", fontsize=18)
plt.plot([aMG],[iMG],marker='o',markersize=15,color='red')
plt.legend(frameon=False)

#plotar  Grupo Hungarian
#plt.plot(hung['a'].to_list(),hung['Incl.'].to_list(),marker='o',markersize=0.5,color='blue')

plt.tight_layout()
plt.savefig(dirGr2+'/'+dirRun+ 'incMPCORB.png')

data = []
dfOut = pd.DataFrame(data, columns = ['cam1', 'dur1','cam2','dur2','vel'])

def lerDataNum(dataText):
  dataTextSep= dataText.split(':')
  return dataTextSep[1]


for arqOut in camFile:
  #SJB_0.5_SAL_0.5
  cam1 = arqOut[0:3]
  dur1 = arqOut[4:7]
  cam2 = arqOut[8:11]
  dur2 = arqOut[12:15]
  lerArq = open(dirGr2+'/'+dirRun+arqOut+'/dados.txt','r')
  text = lerArq.read()
  lerArq.close()
  textLine = text.split("\n")
  if "using cam = 1" in text:
    camUsing = 'cam1'   
  elif "using cam = 2" in text:
    camUsing = 'cam2'
#  print(arqOut)
  for numLine,item in enumerate(textLine):
    if ('velocidade meteoro '+camUsing) in item:
      velLine = numLine+1
      velArq = float(textLine[velLine])
    
  camOut = [cam1,dur1,cam2,dur2,velArq]

  dfOut.loc[len(dfOut)] = camOut
# display(dfOut)

plt.rcParams['font.size'] = '16'
dfOut.to_excel(dirGr2+'/'+dirRun+"finalVelocities.xls")
dfOut.hist(column='vel',bins=15,figsize = (10,7))
plt.xlabel('v (km/s)', fontsize=16)
plt.ylabel('N', fontsize=16)




'''
cam = df['camera'].to_list()
camFile =  []
for i in cam:
  for j in cam:
    if j!=i:
      camFile.append(i+"_"+j)
      camFile.append(j+"_"+i)
camFile = list(dict.fromkeys(camFile))
print(camFile)


data = []
dfOut = pd.DataFrame(data, columns = ['pair', 'vel','a','e','i','w','W','f'])

def lerDataNum(dataText):
  dataTextSep= dataText.split(':')
  return dataTextSep[1]


for arqOut in camFile:
  print(dirGr+'/'+dirRun+arqOut+'/dados.txt')
  lerArq = open(dirGr+'/'+dirRun+arqOut+'/dados.txt','r')
  text = lerArq.read()
  lerArq.close()
  textLine = text.split("\n")
  if "using cam = 1" in text:
    camUsing = 'cam1'   
  elif "using cam = 2" in text:
    camUsing = 'cam2'
  print(arqOut)
  for numLine,item in enumerate(textLine):
    if ('velocidade meteoro '+camUsing) in item:
      velLine = numLine+1
      velArq = float(textLine[velLine])
    if ('semi-major axis (au) :') in item:
      aArq = float(lerDataNum(item))
    if ('eccentriciy :') in item:
      eArq = float(lerDataNum(item))
    if ('inclination :') in item:
      iArq = float(lerDataNum(item))
    if ('perige :') in item:
      wArq = float(lerDataNum(item))
    if ('node :') in item:
      WArq = float(lerDataNum(item))
    if ('true anomaly :') in item:
      fArq = float(lerDataNum(item))
  camOut = [arqOut,velArq,aArq,eArq,iArq,wArq,WArq,fArq]

  dfOut.loc[len(dfOut)] = camOut
display(dfOut)

dfOut.to_excel(dirGr+'/'+dirRun+"finalDataPair.xls")
# print(dfOut.info())


param = ['a','e','i']
for calc in param:
  print(calc,' mean ',dfOut[calc].mean())
  print(calc,' std ',dfOut[calc].std())
  print(calc,' max ',dfOut[calc].max())
  print(calc,' min ',dfOut[calc].min())
  print(calc,' max-min ',dfOut[calc].max() - dfOut[calc].min())
  print('-----------------')


  #encontros próximos
earthLine=[]
venusLine=[]
marsLine=[]
uaLua = 149597870.70/384400
listaP=['meteor','EARTH','MOON','MARS','VENUS']
listaHill = [0,0.0098,0,0.0066,0.0067]
tamHill = 5
camLineVEarth = []
for i,arqOut in enumerate(camFile):
  camLineVEarth.append([])

  meteorData = pd.read_csv(dirGr+'/'+dirRun+arqOut+'/meteor.aei',sep=" ")
  earthData = pd.read_csv(dirGr+'/'+dirRun+arqOut+'/Earth.aei',sep=" ")
  marsData = pd.read_csv(dirGr+'/'+dirRun+arqOut+'/Mars.aei',sep=" ")
  venusData = pd.read_csv(dirGr+'/'+dirRun+arqOut+'/Venus.aei',sep=" ")

  #isplay(earthData)
  meteorData['r']=np.sqrt((meteorData['x']**2)+(meteorData['y']**2)+(meteorData['z']**2))
  #Encontrar minima aproximacao Terra e Marte
  meteorXYZ=meteorData[['Time(y)','x','y','z']]
  colunas = ['Time(y)','xM','yM','zM']
  meteorXYZ.columns = colunas

  meteorXYZ = pd.concat([meteorXYZ,earthData[['x','y','z']]],axis=1)
  colunas.extend(['xE','yE','zE'])
  meteorXYZ.columns = colunas
  meteorXYZ = pd.concat([meteorXYZ,marsData[['x','y','z']]],axis=1)
  colunas.extend(['xMa','yMa','zMa'])
  meteorXYZ.columns = colunas
  meteorXYZ = pd.concat([meteorXYZ,venusData[['x','y','z']]],axis=1)
  colunas.extend(['xV','yV','zV'])
  meteorXYZ.columns = colunas

  meteorXYZ["distEarth"] =uaLua*np.sqrt((meteorXYZ['xM']-meteorXYZ['xE'])**2+(meteorXYZ['yM']-meteorXYZ['yE'])**2+(meteorXYZ['zM']-meteorXYZ['zE'])**2)
  meteorXYZ["distMars"] =uaLua*np.sqrt((meteorXYZ['xM']-meteorXYZ['xMa'])**2+(meteorXYZ['yM']-meteorXYZ['yMa'])**2+(meteorXYZ['zM']-meteorXYZ['zMa'])**2)
  meteorXYZ["distVenus"]=uaLua*np.sqrt((meteorXYZ['xM']-meteorXYZ['xV'])**2+(meteorXYZ['yM']-meteorXYZ['yV'])**2+(meteorXYZ['zM']-meteorXYZ['zV'])**2)

  limT=-0.5

  graphDist=meteorXYZ.plot( x='Time(y)', y=['distEarth','distMars','distVenus'],figsize=(10,5),fontsize=16,color=['r','y','c'],ylabel="d (lunar distance)")
  graphDist.set_ylim(0,1.1*tamHill*listaHill[1]*uaLua)
  graphDist.set_xlim(right=limT)
  graphDist.xaxis.get_label().set_fontsize(16)
  graphDist.yaxis.get_label().set_fontsize(16)
  plt.savefig(dirGr+'/'+dirRun+arqOut+"_"+'encounters.png')
  
  
  closeP=pd.DataFrame()
  closeP['distEarth'] = meteorXYZ.distEarth[(meteorXYZ.distEarth.shift(1) > meteorXYZ.distEarth) & (meteorXYZ.distEarth.shift(-1) > meteorXYZ.distEarth)]
  mergedStuff = pd.merge(closeP[['distEarth']], meteorXYZ[['Time(y)','distEarth']],on='distEarth')
  mergedStuff=mergedStuff[mergedStuff['distEarth']<tamHill*listaHill[1]*uaLua]
  earthLine = earthLine + mergedStuff['Time(y)'].tolist()
  camLineVEarth[i] = camLineVEarth[i] + mergedStuff['Time(y)'].tolist()

  closeP=pd.DataFrame()
  closeP['distMars'] = meteorXYZ.distMars[(meteorXYZ.distMars.shift(1) > meteorXYZ.distMars) & (meteorXYZ.distMars.shift(-1) > meteorXYZ.distMars)]
  mergedStuff = pd.merge(closeP[['distMars']], meteorXYZ[['Time(y)','distMars']],on='distMars')
  mergedStuff=mergedStuff[mergedStuff['distMars']<tamHill*listaHill[3]*uaLua]
  eartLine=[]
  marsLine = marsLine+ mergedStuff['Time(y)'].tolist()


  closeP=pd.DataFrame()
  closeP['distVenus'] = meteorXYZ.distVenus[(meteorXYZ.distVenus.shift(1) > meteorXYZ.distVenus) & (meteorXYZ.distVenus.shift(-1) > meteorXYZ.distVenus)]
  mergedStuff = pd.merge(closeP[['distVenus']], meteorXYZ[['Time(y)','distVenus']],on='distVenus')
  mergedStuff=mergedStuff[mergedStuff['distVenus']<tamHill*listaHill[4]*uaLua]
  venusLine = venusLine + mergedStuff['Time(y)'].tolist()



dfM = pd.DataFrame([])
listaG = [[],[],[]]
for i,arqOut in enumerate(camFile):
  df = pd.read_csv(dirGr+'/'+dirRun+arqOut+'/meteor.aei',sep=" ")
  df = df.drop(["peri","node","f","x","y","z"],axis=1)
  columnName = ["Time(y)"+arqOut,"a"+arqOut,"e"+arqOut,"i"+arqOut]
  listaG[0].append("a"+arqOut)
  listaG[1].append("e"+arqOut)
  listaG[2].append("i"+arqOut)
  #display(df)
  df.columns = columnName
  dfM = pd.concat([dfM,df],axis=1)

dfM['meanA'] = dfM[listaG[0]].mean(axis=1)
dfM['meanE'] = dfM[listaG[1]].mean(axis=1)
dfM['meanI'] = dfM[listaG[2]].mean(axis=1)

dfM=dfM[dfM["Time(y)"+camFile[0]]<-0.5]

# listaG[0].append('meanA')
# listaG[1].append('meanE')
# listaG[2].append('meanI')
colunasGraph=["a","e","i"]
colunasNome=['a (au)','e','i ('+u"\N{DEGREE SIGN}"+')']
arq=['semiMajor','exc','inc']
timeList = dfM['Time(y)'+camFile[0]].to_list()
meanList = [dfM['meanA'].to_list(),dfM['meanE'].to_list(),dfM['meanI'].to_list()]
plt.figure()
cor = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
for i,graficar in enumerate(colunasGraph):
  # labelC = list(map(lambda x : x[1:], listaG[i][:-1]))
  # labelC.append(listaG[i][-1][0:-1])
  labelC = list(map(lambda x : x[1:], listaG[i]))
  graphA=dfM.plot(x="Time(y)"+camFile[1], y=listaG[i],figsize=(10,5),fontsize=16,ylabel=colunasNome[i],
                  linewidth=3,label=labelC,color=cor)
  plt.plot(timeList,meanList[i],label='MEAN',linewidth=5.0,color="black")
  graphA.set_xlim(right=0)
  graphA.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.xlabel('Time (years)', fontsize=18)
  plt.yticks(fontsize=16)
  plt.xticks(fontsize=16)
  graphA.xaxis.get_label().set_fontsize(16)
  graphA.yaxis.get_label().set_fontsize(16)
  # for xc in earthLine:
  #   graphA.axvline(x=xc,color='r',linestyle='dashed',linewidth=1)
  # for xc in marsLine:
  #   graphA.axvline(x=xc,color='y',linestyle='dashed',linewidth=1)
  # for xc in venusLine:
  #   graphA.axvline(x=xc,color='c',linestyle='dashed',linewidth=1)
  for j,arqOut in enumerate(camFile):
    for k in camLineVEarth[j]:
      graphA.axvline(x=k,linestyle='dashed',linewidth=1.2,color=cor[j])

  plt.tight_layout()
  plt.savefig(dirGr+'/'+dirRun+ arq[i]+'.png')



aMG = (np.mean(meanList[0]))
eMG = (np.mean(meanList[1]))
iMG = (np.mean(meanList[2]))
print(aMG);
print(np.std(meanList[0]))
print(np.std(meanList[1]))
print(np.std(meanList[2]))


arqAst = open(str(pathlib.Path().home().resolve()) + "/documentos/PIM/MPCORB.DAT",'r')
arqAstF = open(str(pathlib.Path().home().resolve()) + "/documentos/PIM//MPCORB_Orbital_Elements.DAT",'w')
listaAst = arqAst.read()
listaAst= listaAst.split('\n')
for i,asteroid in enumerate(listaAst):
  astSimp = asteroid[26:104] 
  arqAstF.write(astSimp + '\n')
arqAst.close()
arqAstF.close()


pMPCORB = pd.read_csv(str(pathlib.Path().home().resolve()) + "/documentos/PIM/MPCORB_Orbital_Elements.DAT", delim_whitespace=True,on_bad_lines='warn')
newMPCORB = pMPCORB[pMPCORB['a'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
#pd.read_csv("whitespace.csv", header=None, delimiter=r"\s+")



graphA=newMPCORB.plot(x="a", y="e",figsize=(10,5),fontsize=16,ylabel="Eccentricity",xlabel = "Semi-major axis (au)",
                      linestyle = 'None',marker='o',markersize=0.5,color='gray',xlim=(1.,5.5),ylim=(0.,1.),label='_nolegend_')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Semi-major axis (au)", fontsize=18)
plt.ylabel("Eccentricity", fontsize=18)
plt.plot([aMG],[eMG],marker='o',markersize=15,color='red')
plt.legend(frameon=False)

#plotar  Grupo Hungarian
hung = newMPCORB[(newMPCORB['a']>1.78) & (newMPCORB['a']<2.) & (newMPCORB['e']<0.18) & (newMPCORB['Incl.']>16.) & (newMPCORB['Incl.']<34) ]
plt.plot(hung['a'].to_list(),hung['e'].to_list(),marker='o',markersize=0.5,color='blue')

plt.tight_layout()
plt.savefig(dirGr+'/'+dirRun+ 'excMPCORB.png')
### inclinacao
graphA=newMPCORB.plot(x="a", y="Incl.",figsize=(10,5),fontsize=16,ylabel="Eccentricity",xlabel = "Semi-major axis (au)",
                      linestyle = 'None',marker='o',markersize=0.5,color='gray',xlim=(1.,5.5),ylim=(0.,30),label='_nolegend_')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Semi-major axis (au)", fontsize=18)
plt.ylabel("Inclination ("+u"\N{DEGREE SIGN}"+")", fontsize=18)
plt.plot([aMG],[iMG],marker='o',markersize=15,color='red')
plt.legend(frameon=False)

#plotar  Grupo Hungarian
plt.plot(hung['a'].to_list(),hung['Incl.'].to_list(),marker='o',markersize=0.5,color='blue')

plt.tight_layout()
plt.savefig(dirGr+'/'+dirRun+ 'incMPCORB.png')



print(pMPCORB.info())
print(newMPCORB.info())
print(hung.info()) '''