from astropy import units as u
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

def earthR(lonMedia,latMedia):
    lon = np.rad2deg(lonMedia)
    lat = np.rad2deg(latMedia)
    transprojGeo = Transformer.from_crs("lla",{"proj":'geocent',"ellps":'WGS84',"datum":'WGS84'})
    xc,yc,zc=transprojGeo.transform(lat,lon,0.)
    return ((sqrt(xc**2+yc**2+zc**2)/1000))


# Conersão ECEF - ECI
 # "Earth-centered - Earth-fixed"  to Earth-centered inertial (ECI)

def ecf2eci(tempo,Pos,Vel):
    # now = Time('2018-03-14 23:48:00')
    now = time(tempo)
    pos = [Pos[0], Pos[1], Pos[2]]*u.km
    vel = [Vel[0], Vel[1], Vel[2]]*u.km/u.s

    gcrs = sym.coord.ITRS(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2], representation_type='cartesian', differential_type='cartesian', obstime=now)
    itrs = gcrs.transform_to(sym.coord.GCRS(obstime=now))

    return itrs.cartesian.xyz.value,itrs.cartesian.differentials['s'].get_d_xyz().value


"""## Pim Trajectory"""

#######################################################################################################################

# Dados para integração Reversa por 10 dias
#import rebound

def PIMTrajectory(arquivoMeteoroEntrada):
  
  transprojCart = Transformer.from_crs({"proj":'geocent',"datum":'WGS84',"ellps":'WGS84'},"lla")
  transprojGeo = Transformer.from_crs("lla",{"proj":'geocent',"ellps":'WGS84',"datum":'WGS84'})

  # arquivo = str(pathlib.Path().resolve()) + "/"+"drive/MyDrive/Colab Notebooks/"+sys.argv[1]
  arquivo = str(pathlib.Path().resolve()) + "/"+"drive/MyDrive/Colab Notebooks/"+arquivoMeteoroEntrada
  leitura = pd.read_csv(arquivo,sep='=',comment='#',index_col=0).transpose()
  leitura = leitura.apply(pd.to_numeric, errors='ignore')
# Leiruta de dados de entrada
#nome do meteoro e diretório para cálculo
  dataMeteoro = [leitura['ano'][0],leitura['mes'][0],leitura['dia'][0],leitura['hora'][0],leitura['minuto'][0]
                ,leitura['segundo'][0]]
  horaMeteoro=datetime(dataMeteoro[0],dataMeteoro[1],dataMeteoro[2],dataMeteoro[3],dataMeteoro[4],dataMeteoro[5])
  CD=leitura['CD'][0]
  densMeteor=leitura['densMeteor'][0]
  #integracão: massa do meteoro, tempo de integração (dias), passo de integração
  massaInt=leitura['massaInt'][0]
  tInt=leitura['tInt'][0]
  tIntStep=leitura['tIntStep'][0]
  #tamanho esfera de Hill para close enconter
  tamHill=leitura['tamHill'][0]
  meteorN=str(leitura['meteorN'][0])
  dirG = str(pathlib.Path().resolve()) + "/"+"drive/MyDrive/Colab Notebooks"
  
  dirM = os.path.join(dirG,meteorN)
  print(dirM)

  #verifica se foi possível integrar o meteoro
  with open(dirM+'/dados.txt','r') as f:
    ent = f.read()
    if ('slow velocity') in ent:
      return
#dados do  meteoro entrou na atmosfera
  with open(dirM+'/FinalCartesian.dat','r') as f:
    ent=f.read()
  dadosEntradaAtmos =  ent.split(" ")
  tempoVoo=float(dadosEntradaAtmos[0])
  tempoVooNum= timedelta(seconds=-tempoVoo)
  print("tempo de Voo para",massaInt," kg =",tempoVooNum)
  initialT=(horaMeteoro - tempoVooNum).strftime("%Y-%m-%d %H:%M:%S")
  coordEntAtmos = list(map(float, dadosEntradaAtmos[1:4]))
  velEntAtmos = list(map(float, dadosEntradaAtmos[4:7]))
  
#ajustando a velocidade km/s
  LatEnt,LonEnt,HEnt = transprojCart.transform(coordEntAtmos[0]*1000.,coordEntAtmos[1]*1000.,coordEntAtmos[2]*1000.) 
#VE: EARTH ROTATION VELOCITY AT THE STARTING POINT
  VE = (2.*np.pi*(earthR(np.radians(LonEnt),np.radians(LatEnt)))
                  *cos(np.radians(LatEnt)))/(23.9344696*3600.)
  #CORRECTION WITH EARTH ROTATION VELOCITY	
  V0X = velEntAtmos[0] -VE*np.sin(np.radians(LonEnt))
  V0Y = velEntAtmos[1] +VE*np.cos(np.radians(LonEnt))
  V0Z = velEntAtmos[2]
  V0Meteor = [V0X,V0Y,V0Z]


  coordXYZAtmos,velXYZAtmos = ecf2eci(initialT,coordEntAtmos,V0Meteor)
  coordXYZAtmos = sym.rot_axis1(np.deg2rad(-23.4392911111))@(np.array(coordXYZAtmos))
  velXYZAtmos = sym.rot_axis1(np.deg2rad(-23.4392911111))@(np.array(velXYZAtmos))


  dateT = initialT[:-3] #retirando os segundos (o Rebound vai somente até minuto)

  sim = rebound.Simulation()
  sim.units = ('s', 'km', 'kg')
  sim.add("sun", date=dateT,hash='sol')
  sim.add("399", date=dateT,hash='terra')
  sim.convert_particle_units('km', 's', 'kg')
  ps = sim.particles
  xyzIniMeteor = [None]*6
  xyzIniMeteor[0] = coordXYZAtmos[0] + (ps[1].x)
  xyzIniMeteor[1] = coordXYZAtmos[1] + (ps[1].y)
  xyzIniMeteor[2] = coordXYZAtmos[2] + (ps[1].z)
  xyzIniMeteor[3] = velXYZAtmos[0] + (ps[1].vx)
  xyzIniMeteor[4] = velXYZAtmos[1] + (ps[1].vy)
  xyzIniMeteor[5] = velXYZAtmos[2] + (ps[1].vz)  
  
  sim = rebound.Simulation()
  sim.units = ('s', 'km', 'kg')
  
  sim.add("sun", date=dateT,hash='sol')
  sim.add(x=xyzIniMeteor[0],y=xyzIniMeteor[1],z=xyzIniMeteor[2],
          vx=xyzIniMeteor[3],vy=xyzIniMeteor[4],vz=xyzIniMeteor[5],
          date=dateT,hash = "meteoro",m=0.)
  sim.add("399", date=dateT,hash='terra')
  sim.add("301", date=dateT,hash='lua')
  ps = sim.particles
  
  sim.move_to_com()
  sim.convert_particle_units('AU', 'day', 'Msun')
  
  os = sim.calculate_orbits()

  sim.status()
  for o in sim.calculate_orbits(): 
      print(o)
      
  #salvando dados da partícula
  rebound.OrbitPlot(sim, unitlabel="[AU]",color=True)
  print(dateT)


######################################################################################################

# Integração Reversa por 10 dias

  sim.integrator = "ias15"
  Noutputs = int(60000)
  times = np.linspace(0.,-60, Noutputs)
  elInt =   np.zeros((6,3,Noutputs))
  xyzInte = np.zeros((6,3,Noutputs))
  for i,time in enumerate(times):
      sim.integrate(time)
      os = sim.calculate_orbits()
      for k in range(3):
          xyzInte[0][k][i] = ps[k+1].x   # This stores the data which allows us to plot it later
          xyzInte[1][k][i] = ps[k+1].y
          xyzInte[2][k][i] = ps[k+1].z
          xyzInte[3][k][i] = ps[1].vx
          xyzInte[4][k][i] = ps[1].vy
          xyzInte[5][k][i] = ps[1].vz
          elInt[0][k][i]   = os[k].a
          elInt[1][k][i]   = os[k].e
          elInt[2][k][i]   = np.rad2deg(os[k].inc)
          elInt[3][k][i]   = np.rad2deg(os[k].omega)
          elInt[4][k][i]   = np.rad2deg(os[k].Omega)
          elInt[5][k][i]   = np.rad2deg(os[k].f)      
  hill = 0.01
  iHill = True
  distMinLua = 100.
  for i,time in enumerate(times):
      xL,yL,zL = xyzInte[0][2][i],xyzInte[1][2][i],xyzInte[2][2][i]
      xT,yT,zT = xyzInte[0][1][i],xyzInte[1][1][i],xyzInte[2][1][i]
      xM,yM,zM = xyzInte[0][0][i],xyzInte[1][0][i],xyzInte[2][0][i]
      distTerra = (sqrt((xM-xT)**2+(yM-yT)**2 +(zM-zT)**2))
      # print(distTerra,hill)
      if (( distTerra > hill) and iHill):
          print("passagem na esfera de Hill")
          print('t = ',times[i],'dist Terra = ',distTerra,
                '\n a=', elInt[0][0][i],'\n e=',elInt[1][0][i],'\n i= ',elInt[2][0][i],
                '\n omega=', elInt[3][0][i],'\n Omega=', elInt[4][0][i],'\n f=', elInt[5][0][i])
          tHill,aHill,eHill,incHill,oHill,OHill,fHill = i,elInt[0][0][i],elInt[1][0][i],elInt[2][0][i],elInt[3][0][i],elInt[4][0][i],elInt[5][0][i]
          xHill,yHill,zHill,vxHill,vyHill,vzHill =xyzInte[0][0][i],xyzInte[1][0][i],xyzInte[2][0][i],xyzInte[3][0][i],xyzInte[4][0][i],xyzInte[5][0][i]
          iHill = False
      distLua = (sqrt((xM-xL)**2+(yM-yL)**2 +(zM-zL)**2))
      if distLua < distMinLua:
          distMinLua = distLua
          tLua = i
          # print(times[tLua])

  print("Distancia Minima a Lua = ",distMinLua*149597870.70/384400, " distancia Lunares em ",times[tLua],"dias")

  aHillS = "{:.4f}".format(aHill)
  eHillS = "{:.4f}".format(eHill)
  iHillS = "{:.4f}".format(incHill)
  plt.figure()
  fig,ax = plt.subplots(2,2,figsize=(10,5))
  ax[0][0].plot(times[0:tHill+1000],elInt[0][0][0:tHill+1000])
  ax[0][0].plot(times[tHill],aHill,marker = 'o',label=aHillS)
  ax[0][0].legend()
  ax[0][1].plot(times[0:tHill+1000],elInt[1][0][0:tHill+1000],color='red')
  ax[0][1].plot(times[tHill],eHill,marker = 'o',label=eHillS)
  ax[0][1].legend()
  ax[1][0].plot(times[0:tHill+1000],(elInt[2][0][0:tHill+1000]),color= 'green')
  ax[1][0].plot(times[tHill],incHill,marker = 'o',label=iHillS)
  ax[1][0].legend()
  #ax[1,1].set_axis_off() eliminar o gráfico da lista
  ax[1][1].plot((xyzInte[0][0][0:tHill]-xyzInte[0][1][0:tHill]),(xyzInte[1][0][0:tHill]-xyzInte[1][1][0:tHill]),color= 'yellow')
  ax[1][1].plot((xyzInte[0][2]-xyzInte[0][1]),(xyzInte[1][2]-xyzInte[1][1]),color= 'red')
  ax[1][1].plot((xyzInte[0][1]-xyzInte[0][1]),(xyzInte[1][1]-xyzInte[1][1]),color= 'black',marker ='o')
  ax[1][1].plot((xyzInte[0][2][0]-xyzInte[0][1][0]),(xyzInte[1][2][0]-xyzInte[1][1][0]),color= 'blue',marker ='o')
  ax[1][1].set_aspect('equal', adjustable='box')
  ax[0][0].set(xlabel="time (days)", ylabel="a (UA)")
  ax[0][1].set(xlabel="time (days)", ylabel="e")
  ax[1][0].set(xlabel="time (days)", ylabel="inc (deg)")
  ax[1][1].set(xlabel="x", ylabel="y")

  #if (aHill*(1+eHill)>4.5):
  #    sim.add()

  fig.tight_layout()
  fig.savefig(dirM+'/orbitasHill.png')
  #fig.savefig(dirM+'/orbitasHill.pdf')
  #plt.show()

#####################################################################################################################

# Gravar dados de 10 dias e Hill
  elementosHill = [times[tHill],aHill,eHill,incHill,oHill,OHill,fHill]
  colunasHiilNomes=["timeimpact(days)","semi-major axis (au)","eccentriciy","inclination","perige","node","true anomaly"]
  texto=''
  for i in range(7):
      texto=texto+colunasHiilNomes[i]+" : "+str(elementosHill[i])+" \n "
  print(texto)
  with open(dirM+'/dados.txt','a') as f:
      f.write("Moon minimum distance: "+str(distMinLua*149597870.70/384400)+" lunar distance \n")
      f.write(texto)

  arquivos = ['meteor10d.aei','Earth10d.aei','Moon10d.aei']
  colunas = ['Time(d)','a','e','i','peri','node','f','x','y','z']
  dadosOrbitas=[]
  c = 0 #meteoro
  for c in range(3):
      dadosOrbitas.append(pd.DataFrame([times,elInt[0][c],elInt[1][c],elInt[2][c],elInt[3][c],elInt[4][c],elInt[5][c],
                          xyzInte[0][c],xyzInte[1][c],xyzInte[2][c]]))
      dadosOrbitas[c] = dadosOrbitas[c].swapaxes("index", "columns")
      dadosOrbitas[c].columns = colunas
      dadosOrbitas[c].to_csv(dirM+'/'+arquivos[c],sep=' ',index=False)
      
  sim = rebound.Simulation()
  sim.add("sun", date=dateT,hash='sol')
  planetasLista = []
  for i in range (199,599,100):
      sim.add(str(i), date=dateT)
  sim.move_to_hel()
  sim.add(m=0,a=aHill,e=eHill,inc=np.radians(incHill),omega=np.radians(oHill),Omega=np.radians(OHill),f=np.radians(fHill),primary=sim.particles[0],hash='particle')
  orbit = sim.particles["particle"].calculate_orbit(sim.particles[0])
  if (orbit.a*(1+orbit.e))>4.5:
    sim.add(str(599), date=dateT)

  figOrb = rebound.OrbitPlot(sim, unitlabel="[AU]",color=True)
  figOrb.fig.savefig(dirM+'/'+'orbitaMeteoro.png')
  #figOrb.savefig(dirM+'/'+'orbitaMeteoro.pdf')

  #figOrb2 = rebound.OrbitPlot(sim,xlim=[-4.,4],ylim=[-4.,4],slices=0.5,color=True)
  figOrb2 = rebound.OrbitPlotSet(sim,color=True)
  figOrb2.fig.savefig(dirM+'/'+'orbitaMeteoroPlanos.png')

#########################################################################################################################################################

# Integração por  anos informado no input

  sim = rebound.Simulation()
  sim.units = ('s', 'km', 'kg')
  sim.add("sun", date=dateT,hash='sol')
  sim.add(x=xyzIniMeteor[0],y=xyzIniMeteor[1],z=xyzIniMeteor[2],
          vx=xyzIniMeteor[3],vy=xyzIniMeteor[4],vz=xyzIniMeteor[5],
          date=dateT,hash = "meteoro",m=0.)
  planetasLista = []
  for i in range (199,999,100):
      planetasLista.append(str(i))
  planetasLista.append('301')

  dateT = initialT[:-3] #retirando os segundos (o Rebound vai somente até minuto)
  for i in planetasLista:
      sim.add(i, date=dateT)
  sim.move_to_hel()
  # sim.move_to_com()
  sim.convert_particle_units('AU', 'day', 'Msun')
  ps = sim.particles
  os = sim.calculate_orbits()
  # ps[1].x,ps[1].y,ps[1].z,ps[1].vx,ps[1].vy,ps[1].vz = xyzIniMeteor[0],xyzIniMeteor[1],xyzIniMeteor[2],xyzIniMeteor[3],xyzIniMeteor[4],xyzIniMeteor[5]
  sim.convert_particle_units('AU', 'yr', 'Msun')

  sim.integrator = "ias15"
  #sim.integrator = "mercurius"
  #sim.ri_mercurius.hillfac = 4.
  Noutputs = int(tInt/tIntStep)
  times = np.linspace(0.,tInt, Noutputs)
  elInt =   np.zeros((6,10,Noutputs))
  xyzInte = np.zeros((6,10,Noutputs))
  for i,time in enumerate(times):
      os.system('clear')
      print('tempo atual:',time)
      sim.integrate(time)
      os = sim.calculate_orbits()
      for k in range(10):
          xyzInte[0][k][i] = ps[k+1].x-ps[0].x   # This stores the data which allows us to plot it later
          xyzInte[1][k][i] = ps[k+1].y-ps[0].y
          xyzInte[2][k][i] = ps[k+1].z-ps[0].z
          elInt[0][k][i]   = os[k].a
          elInt[1][k][i]   = os[k].e
          elInt[2][k][i]   = np.rad2deg(os[k].inc)
          elInt[3][k][i]   = np.rad2deg(os[k].omega)
          elInt[4][k][i]   = np.rad2deg(os[k].Omega)
          elInt[5][k][i]   = np.rad2deg(os[k].f)
      # if (elInt[1][1][i]) > 1.1: #saiu do sistema solar
        # break

##########################################################################################################

# Salvando arquivos

  corpos = ['meteor','Earth','Mars','Venus'] #objetos para graficar
  corposI = [0,3,4,2] # indice dos objetos para graficar na simulação

  colunas = ['Time(y)','a','e','i','peri','node','f','x','y','z']
  dadosOrbitas=[]
  indice = 0 
  for c in corposI:
      dadosOrbitas.append(pd.DataFrame([times,elInt[0][c],elInt[1][c],elInt[2][c],elInt[3][c],elInt[4][c],elInt[5][c],
                          xyzInte[0][c],xyzInte[1][c],xyzInte[2][c]]))
      dadosOrbitas[indice] = dadosOrbitas[indice].swapaxes("index", "columns")
      dadosOrbitas[indice].columns = colunas
      dadosOrbitas[indice].to_csv(dirM+'/'+corpos[indice]+'.aei',sep=' ', index=False)
      indice+=1
################################################################################################################

# Gráficos 100 anos

  listaP=['meteor','EARTH','MOON','MARS','VENUS']
  listaHill = [0,0.0098,0,0.0066,0.0067]
  uaLua = 149597870.70/384400
  meteorData=dadosOrbitas[0]
  earthData =dadosOrbitas[1]
  marsData  =dadosOrbitas[2]
  venusData= dadosOrbitas[3]

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
  plt.savefig(dirM+'/'+'encounters.png')
  #plt.savefig(dirM+'/'+'encounters.pdf')

  closeP=pd.DataFrame()
  closeP['distEarth'] = meteorXYZ.distEarth[(meteorXYZ.distEarth.shift(1) > meteorXYZ.distEarth) & (meteorXYZ.distEarth.shift(-1) > meteorXYZ.distEarth)]
  mergedStuff = pd.merge(closeP[['distEarth']], meteorXYZ[['Time(y)','distEarth']],on='distEarth')
  mergedStuff=mergedStuff[mergedStuff['distEarth']<tamHill*listaHill[1]*uaLua]
  eartLine=[]
  earthLine=mergedStuff['Time(y)'].tolist()

  closeP=pd.DataFrame()
  closeP['distMars'] = meteorXYZ.distMars[(meteorXYZ.distMars.shift(1) > meteorXYZ.distMars) & (meteorXYZ.distMars.shift(-1) > meteorXYZ.distMars)]
  mergedStuff = pd.merge(closeP[['distMars']], meteorXYZ[['Time(y)','distMars']],on='distMars')
  mergedStuff=mergedStuff[mergedStuff['distMars']<tamHill*listaHill[3]*uaLua]
  eartLine=[]
  marsLine=mergedStuff['Time(y)'].tolist()


  closeP=pd.DataFrame()
  closeP['distVenus'] = meteorXYZ.distVenus[(meteorXYZ.distVenus.shift(1) > meteorXYZ.distVenus) & (meteorXYZ.distVenus.shift(-1) > meteorXYZ.distVenus)]
  mergedStuff = pd.merge(closeP[['distVenus']], meteorXYZ[['Time(y)','distVenus']],on='distVenus')
  mergedStuff=mergedStuff[mergedStuff['distVenus']<tamHill*listaHill[4]*uaLua]
  eartLine=[]
  venusLine=mergedStuff['Time(y)'].tolist()

  #meteorData2=meteorData[meteorData["Time(y)"]<limT]
  meteorGraph=meteorData[["Time(y)","a","e","i","r"]]
  meteorGraph=meteorGraph[meteorGraph["Time(y)"]<limT]

  #meteorGraph['type']
  meteorGraph['q']=meteorGraph['a']*(1-meteorGraph['e'])
  meteorGraph['Q']=meteorGraph['a']*(1+meteorGraph['e'])

  #display(meteorGraph['a'].max())
  #display(meteorGraph['a'].min())
  #display(meteorGraph['e'].max())
  #display(meteorGraph['e'].min())

  # 1 - Amors, 2- Apollo, 3- Atens, 4 Atiras 5- no neos
  #https://cneos.jpl.nasa.gov/about/neo_groups.html
  def tipo(m):
      if (m.a>1.017) and (1.017 <m.q) and (m.q <1.3):
          return("Amor")
      elif (m.a>1.0)  and (m.q<1.017):
          return("Apollo")
      elif (m.a<1.0) and (m.Q>0.983):
          return ("Aten")
      elif (m.a<1.0) and (m.Q<0.983):
          return("Atira")
      else:
          return ("no neo")
  meteorGraph['Type']= meteorGraph.apply(tipo,axis=1)

  plt.figure() 
  fig,axT = plt.subplots(figsize=(10,5))
  axT=plt.plot(meteorGraph['Time(y)'].tolist(),meteorGraph['Type'].tolist())
  plt.xlabel('Time (years)', fontsize=18)
  plt.yticks(fontsize=16)
  plt.xticks(fontsize=16)
  plt.xlim(right=0)
  plt.tight_layout()
  fig.savefig(dirM+'/'+'typeName'+'.png')

  colunasGraph=["a","e","i","r"]
  yrangeT=[1.395,0.29 ,1.3,2]
  yrangeB=[meteorData['a'].min()-0.001,meteorData['e'].min()-0.001,meteorData['i'].min()-0.001,meteorData['r'].min()-0.1]
  arq=['semiMajor','exc','inc','distR']
  colunasNome=['a (UA)','e','i (d)','r (UA)']
  i=0
  plt.figure() 
  for graficar in colunasGraph:
      graphA=meteorGraph.plot(x="Time(y)", y=graficar,figsize=(10,5),fontsize=16,ylabel=colunasNome[i],linewidth=5)
      graphA.set_xlim(right=0)
  #    graphA.set_ylim(yrangeB[i],yrangeT[i])
      graphA.xaxis.get_label().set_fontsize(16)
      graphA.yaxis.get_label().set_fontsize(16)
      graphA.get_legend().remove()
      for xc in earthLine:
          graphA.axvline(x=xc,color='r',linestyle='dashed',linewidth=1)
      for xc in marsLine:
          graphA.axvline(x=xc,color='y',linestyle='dashed',linewidth=1)
      for xc in venusLine:
          graphA.axvline(x=xc,color='c',linestyle='dashed',linewidth=1)
      plt.tight_layout()
      plt.savefig(dirM+'/'+ arq[i]+'.png')
  #    plt.savefig(dirM+'/'+arq[i]+'.pdf')
  #    plt.show()
      i+=1
  
  with open(dirM+'/dados.txt','a') as f:
    f.write("\n\ n ==============================")
    f.write("\n integration time "+str(tInt))
    f.write("\n semi-major axis (AU) -  max: "+str(meteorGraph['a'].max())+ " min: "+str(meteorGraph['a'].min()))
    f.write("\n eccentricity-  max: "+str(meteorGraph['e'].max())+ " min: "+str(meteorGraph['e'].min()))
    f.write("\n inclination (d)-  max: "+str(meteorGraph['i'].max())+ " min: "+str(meteorGraph['i'].min()))
    f.write("\n radial distance (AU)-  max: "+str(meteorGraph['r'].max())+ " min: "+str(meteorGraph['r'].min()))
