import time
import rebound
import folium
import msise00
import datetime
from datetime import date,datetime, timedelta
from datetime import datetime
import pandas as pd
import numpy as np
from numpy import sin,cos,sqrt
import pyproj
from pyproj import Transformer
from PyAstronomy import pyasl
import pathlib
import os
import math
import matplotlib.pyplot as plt
from datetime import date, datetime
import sympy as sym
import variablesPIM as variaveis

# Conversão de Coordenadas Equatoriais para Horizontais

def convEqToHor(y, m, d, ho, mi, se, alt, lon, lat, ra, dec):
  ''' NOME convEqToHor
      FUNÇÃO Realiza a conversão das coordenadas equatoriais para coordenadas horizontais, retornando o azimute e a altitude'''
  jd = datetime.datetime(y, m, d, ho, mi, se)
  jds = pyasl.jdcnv(jd)                 # Converte do calendário gregoriano para o juliano
  alti, az, ha = pyasl.eq2hor(jds, ra, dec, lon=lon, lat=lat, alt=alt)
  return az[0],alti[0]


# Conversão Sistemas de Coordenadas
  # Raio da Terra de acordo com a latitude e longitude

def sphToCartGeo(RTP):
    ''' NOME sphToCartGeo
        FUNÇÃO Realiza a conversão das coordenadas esfericas para coordenadas geograficas'''
    rc = RTP[0]*1000
    lonc = np.rad2deg(RTP[1])
    latc = np.rad2deg(RTP[2])
    transprojGeo = Transformer.from_crs("lla",{"proj":'geocent',"ellps":'WGS84',"datum":'WGS84'})
    xc, yc, zc = transprojGeo.transform(latc,lonc,rc)
    return (np.array([xc,yc,zc])/1000)

def sphToCart(RTP):
    ''' NOME sphToCart
        FUNÇÃO Realiza a conversão das coordenadas esfericas para coordenadas cartesianas'''
    xc = RTP[0]*cos(RTP[2])*cos(RTP[1])
    yc = RTP[0]*cos(RTP[2])*sin(RTP[1])
    zc = RTP[0]*sin(RTP[2])
    return (np.array([xc, yc, zc]))

def carttoSphGeo(XYZ):
    ''' NOME carttoSphGeo
        FUNÇÃO Realiza a conversão das coordenadas cartesianas para coordenadas cartograficas'''
    transprojCart = Transformer.from_crs({"proj":'geocent',"datum":'WGS84',"ellps":'WGS84'},"lla")
    XYZc = XYZ*1000
    lat,lon,alt = transprojCart.transform(XYZc[0],XYZc[1],XYZc[2])
    return np.array([alt/1000,np.radians(lon),np.radians(lat)])

def carttoSph(XYZ):
    ''' NOME carttoSph
        FUNÇÃO Realiza a conversão das coordenadas cartesianas para coordenadas esfericas'''
    r = sqrt(XYZ[0]**2 + XYZ[1]**2 + XYZ[2]**2)
    theta = np.arctan2(XYZ[1], XYZ[0])
    phi = np.arctan2(XYZ[2], sqrt(XYZ[0]**2 + XYZ[1]**2))
    return np.array([r, theta, phi])

def translation(A,B):
    return (A+B)


# Coordenadas geográficas de dois pontos do meteoro
# Método do cruzamento de planos a partir de dados astométricos de duas cameras
# Distância entre dois pontos geográficos 

def distMet(P1,P2): # Ângulos em graus
    ''' NOME distMet
        FUNÇÃO Calcula a distância entre a posição inicial e final do meteoro'''
    P1d = np.copy(P1)
    P2d = np.copy(P2)
    for i in range(1,3):
        P1d[i] = np.radians(P1d[i])
        P2d[i] = np.radians(P2d[i])
    cartP1 = sphToCartGeo(P1d)
    cartP2 = sphToCartGeo(P2d)
    return np.linalg.norm(cartP1-cartP2)

# Velocidade do meteoro

def velMet(P1v,P2v,time): # Ângulos em graus
    ''' NOME velMet
        FUNÇÃO Calcula a velocidade do meteoro a partir distância entre a posição inicial e final do meteoro e do tempo entre essas posições'''
    return distMet(P1v,P2v)/time

def coordGeo(m,sta):  #determinar a linha do meteoro
    rCam,xM,yM,zM = sym.symbols('rCam xM yM zM')
    mCL = sphToCart(m)
    mCG = sym.rot_axis2(np.pi/2 - sta[2])@(mCL)
    mCG = sym.rot_axis3(sta[1])@(mCG)
    mCG = np.array([-mCG[0],mCG[1],mCG[2]])
    staCL = sphToCartGeo(sta)
    mCG = translation(mCG,staCL)
    return mCG

def detPlano(sta,mA,mB):  #determina a equaçao do plano de cada camera
    mAG = coordGeo(mA,sta)
    mBG = coordGeo(mB,sta)
    staCL = sphToCartGeo(sta)
    planM = np.cross((mAG-staCL),(mBG-staCL))
    xM, yM, zM = sym.symbols('xM yM zM')
    mM = np.array([xM,yM,zM])
    planMn=np.dot(planM,mM)
    d= -1.*planMn.subs([(xM, staCL[0]), (yM, staCL[1]), (zM, staCL[2])])    #equacao do plano
    return (planMn + d)

def ponto(sta,sol,m100):    #determinar o cruzamento dos planos, retorna os pontos de cada camera no meteoro
    rCam,xM,yM,zM = sym.symbols('rCam xM yM zM')
    mCam = np.array([rCam,m100[1],m100[2]])
    mCamG = coordGeo(mCam,sta)
    f=mCamG[1]-(sol[yM].subs(zM,mCamG[2]))
    rF = sym.solve(f,rCam)
    x = mCamG[0].subs(rCam,rF[0])
    y = mCamG[1].subs(rCam,rF[0])
    z = mCamG[2].subs(rCam,rF[0])
    vF1 = np.array([x,y,z])
    vF1G = carttoSphGeo(vF1)
    vF1G[1] = np.rad2deg(vF1G[1])
    vF1G[2] = np.rad2deg(vF1G[2])
    return vF1G
    
def meteorDadosG(alt1, lon1, lat1, alt2, lon2, lat2, az1A, h1A, az2A, h2A, az1B, h1B, az2B, h2B):  #lista com o que cada camera viu em coordenadas geograficas
    sta1 = np.array([alt1,np.radians(lon1),np.radians(lat1)])
    sta2 = np.array([alt2,np.radians(lon2),np.radians(lat2)])
    m1A100 = np.array([100.,np.radians(az1A),np.radians(h1A)])
    m2A100 = np.array([100.,np.radians(az2A),np.radians(h2A)])
    m1B100 = np.array([100.,np.radians(az1B),np.radians(h1B)])
    m2B100 = np.array([100.,np.radians(az2B),np.radians(h2B)])
    xM, yM, zM = sym.symbols('xM yM zM')
    plano1 = detPlano(sta1,m1A100,m1B100)
    plano2 = detPlano(sta2,m2A100,m2B100)
    sol = sym.solve([plano1,plano2],(xM,yM,zM))
    v1Acam =  ponto(sta1, sol, m1A100)
    v1Bcam =  ponto(sta1, sol, m1B100)
    v2Acam =  ponto(sta2, sol, m2A100)
    v2Bcam =  ponto(sta2, sol, m2B100)
    return {'v1Acam' : v1Acam,'v1Bcam' : v1Bcam,'v2Acam' : v2Acam,'v2Bcam' : v2Bcam}

# Leitura arquivo entrada
def leituraDoArquivoEntrada (arquivoMeteoroEntrada):  
  arquivo = variaveis.directorystr + arquivoMeteoroEntrada

  leitura = pd.read_csv(arquivo,sep='=',comment='#',index_col=0).transpose()
  leitura = pd.read_csv(arquivo,sep='=',comment='#',index_col=0).transpose()
  leitura = leitura.apply(pd.to_numeric, errors='ignore')

  dataMeteoro = [leitura['ano'][0],leitura['mes'][0],leitura['dia'][0],leitura['hora'][0],
                leitura['minuto'][0],leitura['segundo'][0]]
  leitura['massaPont'] = leitura['massaPont'].astype(str)

  meteorN = str(leitura['meteorN'][0])  # Nome do meteoro e diretório para cálculo

  return arquivo, leitura, dataMeteoro, meteorN

# Integração dos pontos de queda (dados salvos em arquivos .out)


def PIMRun(arquivoMeteoroEntrada):


# Leitura arquivo entrada

  arquivo, leitura, dataMeteoro, meteorN = leituraDoArquivoEntrada(arquivoMeteoroEntrada)

###############################################################################################
# Pontos e Intervalo entre os pontos

  if leitura['opcao'][0] == 1: # Opção 1

      P1lat = leitura['P1lat'][0]
      P1lon = leitura['P1lon'][0]
      P1alt =  leitura['P1alt'][0]
      P2lat =  leitura['P2lat'][0]
      P2lon = leitura['P2lon'][0]
      P2alt = leitura['P2alt'][0]
      deltaT = leitura['deltaT'][0]
      vCam = leitura['cam'][0]


  elif leitura['opcao'][0] == 2 or leitura['opcao'][0] == 3:

      alt1,lon1,lat1 = leitura['alt1'][0],leitura['lon1'][0],leitura['lat1'][0]
      alt2,lon2,lat2 = leitura['alt2'][0],leitura['lon2'][0],leitura['lat2'][0]

      if leitura['opcao'][0] == 2: # Opção 2

          az1Ini, h1Ini =  leitura['az1Ini'][0],leitura['h1Ini'][0]
          az2Ini, h2Ini =  leitura['az2Ini'][0],leitura['h2Ini'][0]
          az1Fin, h1Fin = leitura['az1Fin'][0],leitura['h1Fin'][0]
          az2Fin, h2Fin = leitura['az2Fin'][0],leitura['h2Fin'][0]

      else: # Opção 3

          ra1Ini, dec1Ini =  leitura['ra1Ini'][0],leitura['dec1Ini'][0]
          ra2Ini, dec2Ini =  leitura['ra2Ini'][0],leitura['dec2Ini'][0]
          ra1Fin, dec1Fin =  leitura['ra1Fin'][0],leitura['dec1Fin'][0]
          ra2Fin, dec2Fin =  leitura['ra2Fin'][0],leitura['dec2Fin'][0]
          y, m, d, ho, mi, se = dataMeteoro[0],dataMeteoro[1],dataMeteoro[2],dataMeteoro[3],dataMeteoro[4],dataMeteoro[5]
          az1Ini, h1Ini = convEqToHor(y, m, d, ho, mi, se, alt1, lon1, lat1, ra1Ini, dec1Ini)
          az2Ini, h2Ini = convEqToHor(y, m, d, ho, mi, se, alt2, lon2, lat2, ra2Ini, dec2Ini)
          az1Fin, h1Fin = convEqToHor(y, m, d, ho, mi, se, alt1, lon1, lat1, ra1Fin, dec1Fin)
          az2Fin, h2Fin = convEqToHor(y, m, d, ho, mi, se, alt2, lon2, lat2, ra2Fin, dec2Fin)

      pontosMeteoro = (meteorDadosG(alt1, lon1, lat1, alt2, lon2, lat2, 
                                    az1Ini, h1Ini, az2Ini, h2Ini, az1Fin, h1Fin, az2Fin, h2Fin))

      if leitura['cam'][0] == 1:

          P1alt,P1lon,P1lat = pontosMeteoro['v1Acam'][0],pontosMeteoro['v1Acam'][1],pontosMeteoro['v1Acam'][2]
          P2alt,P2lon,P2lat = pontosMeteoro['v1Bcam'][0],pontosMeteoro['v1Bcam'][1],pontosMeteoro['v1Bcam'][2]
          deltaT = leitura['deltaT1'][0]

      else:

          P1alt,P1lon,P1lat = pontosMeteoro['v2Acam'][0],pontosMeteoro['v2Acam'][1],pontosMeteoro['v2Acam'][2]
          P2alt,P2lon,P2lat = pontosMeteoro['v2Bcam'][0],pontosMeteoro['v2Bcam'][1],pontosMeteoro['v2Bcam'][2]
          deltaT = leitura['deltaT2'][0]

  else:
    
      P1alt,P1lon,P1lat = leitura['alt4d'][0],leitura['lon4d'][0],leitura['lat4d'][0]
      P2alt,P2lon,P2lat = P1alt,P1lon,P1lat
      Vx4,Vy4,Vz4= leitura['Vx4d'][0]*1000.,leitura['Vy4d'][0]*1000.,leitura['Vz4d'][0]*1000.
      deltaT=0

###############################################################################################

# Instante meteoro (ano,mes,dia,hora,minuto,segundo)

  horaMeteoro=datetime(dataMeteoro[0],dataMeteoro[1],dataMeteoro[2],dataMeteoro[3],dataMeteoro[4],dataMeteoro[5])

###############################################################################################

 # Massas para o pontos de queda kg      MassaPont = [0.001,0.01,0.1,1,10,50,100,150]

  massaPont = []
  if leitura['massaPont'][0].find(',') == -1:
      massaPont.append(float(leitura['massaPont'][0]))

  else:
      massaPontString= leitura['massaPont'][0].split(sep=',')
      for i in massaPontString:
          massaPont.append(float(i))

###############################################################################################


  CD=leitura['CD'][0]

###############################################################################################

  # Densidade
  densMeteor = leitura['densMeteor'][0]

###############################################################################################

  # Integracão: massa do meteoro, tempo de integração (dias), passo de integração

  massaInt = leitura['massaInt'][0]
  tInt = leitura['tInt'][0]
  tIntStep = leitura['tIntStep'][0]

###############################################################################################

  # Tamanho da esfera de Hill para close enconter
  tamHill=leitura['tamHill'][0]

###############################################################################################
# Criação do Diretório

  dirM = os.path.join(variaveis.directory, meteorN)
  print(dirM)

  try:
      os.mkdir(dirM)
  except OSError:
      print ("Creation of the directory %s failed" % meteorN)
  else:
      print ("Successfully created the directory %s " % meteorN)

###############################################################################################

  # Gravar informações gerais

  gravarEntrada = open((dirM+'/dados.txt'),'w')
  gravarEntrada.write(("Meteor: "+meteorN+'\n \n'))
  linha='P1: lat: '+str(P1lat)+'  lon: '+str(P1lon)+'  alt: '+str(P1alt) +'\n'
  gravarEntrada.write(linha)

  if leitura['opcao'][0] != 4:
      linha='P2: lat: '+ str(P2lat)+'  lon: '+ str(P2lon)+'  alt: '+str(P2alt) +'\n'
      gravarEntrada.write(linha)

  else:
      linha='Vx (km/s): '+str(Vx4/1000)+'   Vy (km/s): '+str(Vy4/1000)+'  Vz (km/s): '+str(Vz4/1000) +'\n'
      gravarEntrada.write(linha)

  linha='time: '+str(deltaT)+' \n'
  gravarEntrada.write(linha)
  linha='date: '+horaMeteoro.strftime("%Y-%m-%d %H:%M:%S")+'\n'
  gravarEntrada.write(linha)
  gravarEntrada.close()


###############################################################################################

# Unir dados

  if leitura['opcao'][0] == 1:

      strPontosCam ='comprimento meteoro (km)\n'
      strPontosCam +=str(distMet(np.array([P1alt, P1lon, P1lat]),np.array([P2alt, P2lon, P2lat]))) + '\n'
      strPontosCam +='velocidade do meteoro (km/s)\n'
      strPontosCam +=str(velMet(np.array([P1alt, P1lon, P1lat]),np.array([P2alt, P2lon, P2lat]),leitura['deltaT'][0]))+'\n --- \n'
      with open(dirM+ '/'+'dados.txt',"a") as filesCam:
          filesCam.write(strPontosCam)
      if (velMet(np.array([P1alt, P1lon, P1lat]),np.array([P2alt, P2lon, P2lat]),leitura['deltaT'][0]) < 11.):
        with open(dirM+ '/'+'dados.txt',"a") as filesCam:
          filesCam.write("slow velocity")




  if leitura['opcao'][0] == 2 or leitura['opcao'][0] == 3:
    
      strPontosCam = str(pontosMeteoro) + '\n'
      strPontosCam += '--------------\n'
      strPontosCam +='distancia entre as estações \n'
      strPontosCam +=str(distMet(np.array([alt1, lon1, lat1]),np.array([alt2, lon2, lat2]))) + '\n'
      strPontosCam +='--------------\n'
      strPontosCam +='comprimento meteoro cam1 (km)\n'
      strPontosCam +=str(distMet(pontosMeteoro['v1Acam'],pontosMeteoro['v1Bcam'])) + '\n'
      strPontosCam +='comprimento meteoro cam2 (km)\n'
      strPontosCam +=str(distMet(pontosMeteoro['v2Acam'],pontosMeteoro['v2Bcam']))+'\n'
      strPontosCam +='velocidade meteoro cam1 (km/s)\n'
      strPontosCam +=str(velMet(pontosMeteoro['v1Acam'],pontosMeteoro['v1Bcam'],leitura['deltaT1'][0]))+'\n'
      strPontosCam +='velocidade meteoro cam2 (km/s)\n'
      strPontosCam +=str(velMet(pontosMeteoro['v2Acam'],pontosMeteoro['v2Bcam'],leitura['deltaT2'][0]))+'\n'
      strPontosCam +="-----distPontos A B entre cameras-------\n"
      strPontosCam +="distância inicial do meteoro entre as cameras (km)\n"
      strPontosCam +=str(distMet(pontosMeteoro['v2Acam'],pontosMeteoro['v1Acam']))+'\n'
      strPontosCam +="distância final do meteoro entre as cameras (km)\n"
      strPontosCam +=str(distMet(pontosMeteoro['v2Bcam'],pontosMeteoro['v1Bcam']))+'\n ----- \n'
      with open(dirM+ '/'+'dados.txt',"a") as filesCam:
          filesCam.write(strPontosCam)
          filesCam.write("\n using cam = " + str(leitura['cam'][0])+ '\n')
      if (leitura['cam'][0] == 1):
        if (velMet(pontosMeteoro['v1Acam'],pontosMeteoro['v1Bcam'],leitura['deltaT1'][0]) < 11.):
          with open(dirM+ '/'+'dados.txt',"a") as filesCam:
            filesCam.write("slow velocity")
          return
      elif (leitura['cam'][0] == 2):
        if (velMet(pontosMeteoro['v2Acam'],pontosMeteoro['v2Bcam'],leitura['deltaT2'][0]) < 11.):
          with open(dirM+ '/'+'dados.txt',"a") as filesCam:
            filesCam.write("slow velocity")
          return


  if leitura['opcao'][0] == 4:

      strPontosCam ='velocidade do meteoro (km/s)\n'
      strPontosCam +=str(sqrt(Vx4**2+Vy4**2+Vz4**2)/1000.)
      with open(dirM+ '/'+'dados.txt',"a") as filesCam:
          filesCam.write(strPontosCam)
      if ((sqrt(Vx4**2+Vy4**2+Vz4**2)/1000.)<11.):
        with open(dirM+ '/'+'dados.txt',"a") as filesCam:
          filesCam.write("slow velocity")
        # return

    
  print(strPontosCam)


###############################################################################################

# Dados iniciais do meteoro em coordenadas geocêntricas
# Cria as listas de acordo com número de massas a serem integradas

        
  transprojCart = Transformer.from_crs({"proj":'geocent',"datum":'WGS84',"ellps":'WGS84'},"lla")  
  transprojGeo = Transformer.from_crs("lla",{"proj":'geocent',"ellps":'WGS84',"datum":'WGS84'})
  #print( pyproj.Transformer(init='epsg:4326',ecef,lla, x, y, z))

  A=[]
  for i in massaPont:    #determina as massas q serao rodadas
      print(i)
      v=i/densMeteor
      r=(v*3./(4.*math.pi))**(1./3.)/100.
      A.append(math.pi*r*r)

# condiçoes iniciais do meteoro, posiçao e velocidade
  X1=[None] * len(massaPont)
  Y1=[None] * len(massaPont)
  Z1=[None] * len(massaPont)
  X2=[None] * len(massaPont)
  Y2=[None] * len(massaPont)
  Z2=[None] * len(massaPont)
  Vx1=[None] * len(massaPont)
  Vy1=[None] * len(massaPont)
  Vz1=[None] * len(massaPont)
  altM=[None] * len(massaPont)
  latM=[None] * len(massaPont)
  lonM=[None] * len(massaPont)
  altM2=[None] * len(massaPont)
  latM2=[None] * len(massaPont)
  lonM2=[None] * len(massaPont)

  particulas=[None] * len(massaPont)

  for i in range (len(massaPont)):
      X1[i],Y1[i],Z1[i]=transprojGeo.transform(P1lat,P1lon,P1alt*1000.)
      X2[i],Y2[i],Z2[i]=transprojGeo.transform(P2lat,P2lon,P2alt*1000.)
      if leitura['opcao'][0] == 4:
          Vx1[i]= Vx4
          Vy1[i]= Vy4
          Vz1[i]= Vz4
      else:
          Vx1[i]= (X2[i]-X1[i])/deltaT
          Vy1[i]= (Y2[i]-Y1[i])/deltaT
          Vz1[i]= (Z2[i]-Z1[i])/deltaT
      particulas[i]=i
      
  X=[None] * len(massaPont)
  Y=[None] * len(massaPont)
  Z=[None] * len(massaPont)
  Vx=[None] * len(massaPont)
  Vy=[None] * len(massaPont)
  Vz=[None] * len(massaPont)

###############################################################################################

#integraçao pra frente
# Integração dos pontos de queda (dados salvos em arquivos .out)
# (este procedimento é demorado devido ao arrasto atmosférico)


  #particulas=[0]
  #while len(particulas)!=0:

  for i in range (len(massaPont)):
      X[i],Y[i],Z[i]=(X1[i]+X2[i])/2,(Y1[i]+Y2[i])/2,(Z1[i]+Z2[i])/2
      #X[i],Y[i],Z[i]=(X2[i]),(Y2[i]),(Z2[i])
      Vx[i],Vy[i],Vz[i]=Vx1[i],Vy1[i],Vz1[i]
      latM[i],lonM[i],altM[i] = transprojCart.transform(X[i],Y[i],Z[i])

  arquivo=open((dirM+'/saida.out'),'w')
  arquivoQueda=open((dirM+'/queda.dat'),'w')
  arquivoQueda.write("time(s) vel alt(km) lon lat \n")

  for j in range (len(massaPont)):

      tempo=0
      passo=5
      ASection=A[j]
      massaParticula=massaPont[j]

      while altM[j]>0.:

          os.system('clear')
          Vm=np.sqrt(Vx[j]**2+Vy[j]**2+Vz[j]**2)
          if (((altM[j]/1000)<0.005)):
              passo=0.1/Vm
          elif (((altM[j]/1000)<0.040)):
              passo=2/Vm
          elif (((altM[j]/1000)<0.150)):
              passo=20/Vm
          elif (((altM[j]/1000)<0.2)):
              passo=50/Vm
          elif (((altM[j]/1000)<0.4)):
              passo=100/Vm            
          elif (((altM[j]/1000)<1)):
              passo=200/Vm
          elif (((altM[j]/1000)<3)):
              passo=500/Vm
          elif ((altM[j]/1000)<5):
              passo=1000/Vm
          else:
              passo=2000/Vm
          
          print(tempo,"massa(kg): ",massaPont[j],"altura atual (km): ",altM[j]/1000.)
          arquivoQueda.write(str(str(tempo)+" "+str(Vm)+" "+str(altM[j]/1000)+" "+str(lonM[j])+" "+str(latM[j])+" \n"))
          sim = rebound.Simulation()
          sim.integrator = "ias15"
          #sim.ri_ias15.epsilon=1e-3
          sim.units = ('m', 's', 'kg')
          sim.add(m=5.97219e24,hash='earth')
          meteor = rebound.Particle(m=massaPont[j],x=X[j],y=Y[j],z=Z[j],vx=Vx[j],vy=Vy[j],vz=Vz[j])
          sim.add(meteor)
          #sim.status()
          ps=sim.particles
          atmos = msise00.run(time=datetime(horaMeteoro.year,horaMeteoro.month,horaMeteoro.day,horaMeteoro.hour,horaMeteoro.minute,horaMeteoro.second), altkm=altM[j]/1000., glat=latM[j], glon=lonM[j])
          RHO=(float(atmos['Total']))
          
          def arrasto(reb_sim):
              #latA,lonA,altA = transprojCart.transform(ps[1].x,ps[1].y,ps[1].z)
              #atmos = msise00.run(time=datetime(horaMeteoro.year,horaMeteoro.month,horaMeteoro.day,horaMeteoro.hour,horaMeteoro.minute,horaMeteoro.second), altkm=altA/1000., glat=latA, glon=lonA)
              #RHO=(float(atmos['Total']))
              ps[1].ax -= RHO*CD*ASection*(np.sqrt(ps[1].vx**2+ps[1].vy**2+ps[1].vz**2))*ps[1].vx/(2.*massaParticula)
              ps[1].ay -= RHO*CD*ASection*(np.sqrt(ps[1].vx**2+ps[1].vy**2+ps[1].vz**2))*ps[1].vy/(2.*massaParticula)
              ps[1].az -= RHO*CD*ASection*(np.sqrt(ps[1].vx**2+ps[1].vy**2+ps[1].vz**2))*ps[1].vz/(2.*massaParticula)

          sim.additional_forces = arrasto
          sim.force_is_velocity_dependent = 1
          sim.integrate(passo)
          tempo+=passo
          latA,lonA,altA = latM[j],lonM[j],altM[j]
          X[j],Y[j],Z[j],Vx[j],Vy[j],Vz[j] =ps[1].x,ps[1].y,ps[1].z,ps[1].vx,ps[1].vy,ps[1].vz
          latM[j],lonM[j],altM[j] = transprojCart.transform(ps[1].x,ps[1].y,ps[1].z)

          if(((altM[j]/1000)<1)):
              passo=0.00001/Vm
          if ((altM[j]/1000)<5):
              passo=0.2/Vm
          if ((altM[j]/1000)<10):
              passo=100/Vm
          
  #        if ((latM[j]/latA)<1):
  #            passo=-passo/10
  #        print(tempo,latM[j],lonM[j],altM[j]/1000)
      print(massaPont[j],latA,lonA,altA)
      arquivo.write('  1-  '+str(tempo)+' g: '+str(lonA)+" , "+str(latA)+' , '+str(altA/1000.)+' @399 \n')
  arquivo.close()
  arquivoQueda.close()

###################################################################################################################
#integracao pra tras
# Integração reversa até 1000 km de altitude (dados salvos em arquivos .out e .dat)
# Usada a massa em massaInt

  for i in range (len(massaPont)):
      X[i],Y[i],Z[i]=(X1[i]+X2[i])/2,(Y1[i]+Y2[i])/2,(Z1[i]+Z2[i])/2
     # X[i],Y[i],Z[i] = X1[i],Y1[i],Z1[i]
      Vx[i],Vy[i],Vz[i]=Vx1[i],Vy1[i],Vz1[i]
      latM[i],lonM[i],altM[i] = transprojCart.transform(X[i],Y[i],Z[i])

  arquivoCart=open((dirM+'/Cartesian.dat'),'w')
  arquivoCoord=open((dirM+'/Coordinate.dat'),'w')

  arquivoCart.write("time(s) x y z vx vy vz \n")
  arquivoCoord.write("time(s) vel alt(km) lon lat \n")
  
  tempo=0
  passo=-5
  j=massaPont.index(massaInt)
  ASection=A[j]

  massaParticula=massaPont[j]

  while altM[j]<1000e3:
      os.system('clear')
      print(tempo,"altura atual (km): ",altM[j]/1000.)
      print(i, flush=True)
      Vm=np.sqrt(Vx[j]**2+Vy[j]**2+Vz[j]**2)
      if ((altM[j]/1000)>150) and ((altM[j]/1000)<990):
          passo=-10000/Vm
      else:
          passo=-2000/Vm
      strCart=str(str(tempo)+" "+str(X[j]/1000.)+" "+str(Y[j]/1000.)+" "+str(Z[j]/1000.)+" "+str(Vx[j]/1000.)+" "+
                  str(Vy[j]/1000.)+" "+str(Vz[j]/1000.)+" \n")
      arquivoCart.write(strCart)
      arquivoCoord.write(str(str(tempo)+" "+str(Vm)+" "+str(altM[j]/1000)+" "+str(lonM[j])+" "+str(latM[j])+" \n"))
      sim = rebound.Simulation()
      sim.integrator = "ias15"
  #   sim.ri_ias15.epsilon=1e-3
      sim.units = ('m', 's', 'kg')
      sim.add(m=5.97219e24,hash='earth')
      meteor = rebound.Particle(m=massaPont[j],x=X[j],y=Y[j],z=Z[j],vx=Vx[j],vy=Vy[j],vz=Vz[j])
      sim.add(meteor)
  #    sim.status()
      ps=sim.particles
      atmos = msise00.run(time=datetime(horaMeteoro.year,horaMeteoro.month,horaMeteoro.day,horaMeteoro.hour,horaMeteoro.minute,horaMeteoro.second), altkm=altM[j]/1000., glat=latM[j], glon=lonM[j])
      RHO=(float(atmos['Total']))
      def arrasto(reb_sim):
                  ps[1].ax -= RHO*CD*ASection*(np.sqrt(ps[1].vx**2+ps[1].vy**2+ps[1].vz**2))*ps[1].vx/(2.*massaParticula)
                  ps[1].ay -= RHO*CD*ASection*(np.sqrt(ps[1].vx**2+ps[1].vy**2+ps[1].vz**2))*ps[1].vy/(2.*massaParticula)
                  ps[1].az -= RHO*CD*ASection*(np.sqrt(ps[1].vx**2+ps[1].vy**2+ps[1].vz**2))*ps[1].vz/(2.*massaParticula)
      sim.additional_forces = arrasto
      sim.force_is_velocity_dependent = 1
      sim.integrate(passo)
  #    sim.status()
      tempo+=passo
      X[j],Y[j],Z[j],Vx[j],Vy[j],Vz[j] =ps[1].x,ps[1].y,ps[1].z,ps[1].vx,ps[1].vy,ps[1].vz
      latM[j],lonM[j],altM[j] = transprojCart.transform(ps[1].x,ps[1].y,ps[1].z)
  #        print(tempo,latM[j],lonM[j],altM[j]/1000)
  with open((dirM+'/FinalCartesian.dat'),'w') as f:
    f.write(strCart)


  arquivoCart.close()
  arquivoCoord.close()
############################################################################################################
# fazer arquivo com a trajetoria total do meteoro
##
#  for line in reversed(list(open(dirM+'/coordinates.txt'))):
#    print(line.rstrip())
    
################################################################################################################
#salvar os dados dos pontos de queda do meteorito e vai gerar ao mapa de queda (primeira integraçao para frente)
#   Análise dos pontos de queda e da integração reversa até 1000 km
# (usa os arquivos .out salvo nos procedimentos anteriores)
  saida = pd.read_csv((dirM+'/saida.out'), sep='\s+',header=None)
  excluir = [0,2,4,6,7,8]
  saida.drop(saida.columns[excluir],axis=1,inplace=True)

  saida.insert(0, "mass", "Any")


  saida['mass'] = massaPont
  colunas = ['mass','time(s)','lon','lat']
  saida.columns = colunas
  #display(saida)
  arquivo=open((dirM+'/dados.txt'),'a')
  arquivo.write(('\n \n Strewn Field: \n'))
  arquivo.write(saida.to_string(index=False))
  arquivo.write('\n \n')
  arquivo.close()

# referente a integraçao pra tras (qunto tempo ele estava da entrada da atmosfera até chegar oa ponto do meteoro/ instante q ele entrou na atmosfera)
  with open(dirM+'/FinalCartesian.dat','r') as f:
      ent=f.read()
  fimTempoVoo=ent.index(" ")
  tempoVoo=float(ent[0:fimTempoVoo])
  tempoVooNum= timedelta(seconds=-tempoVoo)
  print("tempo de Voo para",massaInt," kg =",tempoVooNum)
  initialT=(horaMeteoro - tempoVooNum).strftime("%Y-%m-%d %H:%M:%S")
  #finalT=(horaMeteoro - tempoVooNum+timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
  print("hora de entrada: ",initialT)

#mapa de queda (integraçao para frente)

  saida['mass'] = saida['mass'].astype(str) + " kg"
  map_osm = folium.Map(
      location = [saida['lat'][saida.shape[0]//2], saida['lon'][saida.shape[0]//2]],
      zoom_start = 14
  )
  map_osm
  folium.Marker(
      location=[P1lat,P1lon],
      popup=folium.Popup('Initial Meteor',show=True),
      icon=folium.map.Icon(color='blue')
  ).add_to(map_osm)  
  folium.Marker(
      location=[P2lat,P2lon],
      popup=folium.Popup('Final Meteor',show=True),
      icon=folium.map.Icon(color='blue')
  ).add_to(map_osm)  

  for indice, row in saida.iterrows():
      folium.Marker(
          location=[row["lat"], row["lon"]],
          popup=folium.Popup(row['mass'],show=True),
          icon=folium.map.Icon(color='yellow')
      ).add_to(map_osm)
  map_osm.save(dirM+'/strewnField.html')
  return
