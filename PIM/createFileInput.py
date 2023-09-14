import pandas as pd
import shutil
import xmltodict
import variablesPIM  
import validationPIM  
import warnings
import os

warnings.filterwarnings('ignore')


def convertToDictionary (cameraList, dirRun):
  """
  This function converts data from a list of camera files to a dictionary and writes the data to two CSV files named Position.csv and Frames.csv.

  Arguments:
        - cameraList: A list of camera file names to be processed.
        - dirRun: The directory path where the camera files are located.

  Returns: None

  Example Usage:
      >>> convertToDictionary(['camera1.xml', 'camera2.xml'], '/path/to/camera/files/') 
  """

  for file in cameraList:
    with validationPIM.createFileRead(variablesPIM.directory.joinpath(dirRun), file) as f:            # Open files in read mode
      validationPIM.createDirIfDoesntExist(variablesPIM.directory,file[:-4])         # Create a directory for this integration
      data = validationPIM.createFileWrite(file[:-4],'Position.csv' )                     # Create file inside the directory with the positions
      data.write('name;fps;y;mo;d;h;m;s;lng;lat;alt\n')                              # Write the name of the variables in the file
      dataFrame = validationPIM.createFileWrite(file[:-4],'Frames.csv')                   # Create file inside the directory with the frames
      dataFrame.write('time;fAbs;fno;ev;az;ra;dec\n')                                # Write the name of the variables in the file

      my_dict = xmltodict.parse(f.read())                                            # Convert to a dictionary
      fps = float(my_dict['ufoanalyzer_record']['@fps'])
      year= (my_dict['ufoanalyzer_record']['@y'])
      month = (my_dict['ufoanalyzer_record']['@mo'])
      day = (my_dict['ufoanalyzer_record']['@d'])
      hour = (my_dict['ufoanalyzer_record']['@h'])
      minute = (my_dict['ufoanalyzer_record']['@m'])
      second = (my_dict['ufoanalyzer_record']['@s'])
      longitude = (my_dict['ufoanalyzer_record']['@lng'])
      latitude = (my_dict['ufoanalyzer_record']['@lat'])
      altitude = (my_dict['ufoanalyzer_record']['@alt'])
      name = (my_dict['ufoanalyzer_record']['@lid'])
      
      # Adds each of the variables read in the dictionary
      data.write(name+';'+str(fps)+';'+year+';'+month+';'+day+';'+hour+';'+minute+';'+second+';'+longitude+';'+latitude+';'+altitude) 
      listFrames = my_dict['ufoanalyzer_record']['ua2_objects']['ua2_object']['ua2_objpath']['ua2_fdata2']
      frame0 = int(listFrames[0]['@fno'])
      time = 0.
      for frame in listFrames:
        absoluteFrame = int(frame['@fno']) - frame0
        dataFrame.write(f'{time:.2f}'.strip()+';'+str(absoluteFrame).strip()+';'+frame['@fno'].strip()+';'+frame['@ev'].strip()+';'+frame['@az'].strip()+';'+
                        frame['@ra'].strip()+';'+frame['@dec'].strip()+'\n')
        time += 1/fps
      data.close()
      dataFrame.close()

def saveDataInList(cameraList, dirRun):
  """
  Save data from '.XML' file in a list

  Arguments:

      cameraList: a list of '.XML' files.
      dirRun: a directory containing the '.XML' files.

  Returns:

      A tuple containing two lists of Pandas DataFrames: dfP and dfF.

  Raises:

      None.

  Example:  dfP, dfF = saveDataInList(['file1.XML', 'file2.XML'], 'data/directory')

  """
  dfP = []
  dfF = []
  for file in cameraList:
    with validationPIM.createFileRead(variablesPIM.directory.joinpath(dirRun), file) as f:
      namePosition = variablesPIM.directory.joinpath(file[:len(file[:-4])]).joinpath('Position.csv').resolve()
      nameFrames = variablesPIM.directory.joinpath(file[:len(file[:-4])]).joinpath('Frames.csv').resolve()
      dfPosition = pd.read_csv(namePosition, separator = ';')
      dfFrames = pd.read_csv(nameFrames, separator = ';')
      dfP.append(dfPosition)
      dfF.append(dfFrames)
  print(dfF[0].info())
  return dfP, dfF

def saveDataInVariables(dfP, dfF):
  '''
  Save data of '.csv' files in variables

  Args:
  - dfP (list): List of pandas dataframes with position data
  - dfF (list): List of pandas dataframes with frame data

  Returns:
  - dfR (pd.DataFrame): Dataframe with the following columns:
      - camera: name of the camera
      - lat: latitude of the camera
      - lon: longitude of the camera
      - alt: altitude of the camera (converted from meters to kilometers)
      - dur: duration of the observation (in seconds)
      - azIni: initial azimuth angle (in degrees)
      - elIni: initial elevation angle (in degrees)
      - azFin: final azimuth angle (in degrees)
      - elFin: final elevation angle (in degrees)
      - raIni: initial right ascension angle (in degrees)
      - decIni: initial declination angle (in degrees)
      - raFin: final right ascension angle (in degrees)
      - decFin: final declination angle (in degrees)

  Raises:

      None.

  Example:  saveDataInVariables(dfP: list, dfF: list)

  '''

  moments = [0.5,1.0,1.5]
  dfR = pd.DataFrame([], columns = ['camera', 'lat','lon','alt','dur','azIni','elIni','azFin','elFin','raIni','decIni','raFin','decFin'])
  for i,cam in enumerate(dfP):
    for time in moments:
      camera = cam['name'][0] +'_'+f'{time:.1f}'
      lat = cam['lat'][0]
      lon = cam['lng'][0]
      alt = cam['alt'][0]
      
      azIni = (dfF[i]['az'][0])
      elIni = (dfF[i]['ev'][0])
      raIni = (dfF[i]['ra'][0])
      decIni = (dfF[i]['dec'][0])
    

      linha = dfF[i].iloc[(dfF[i]['time']-time).abs().argsort()[:1]]
      dur = (linha['time'].values[0])
      azFin = (linha['az'].values[0])
      elFin = (linha['ev'].values[0])
      raFin = (linha['ra'].values[0])
      decFin = (linha['dec'].values[0])

      dataCam = [camera,lat,lon,alt,dur,azIni,elIni,azFin,elFin,raIni,decIni,raFin,decFin]
      dfR.loc[len(dfR)] = dataCam

  dfR['alt'] = dfR['alt']/1000
  return dfR

def writeFilesRun(df, dateM, option,dirRun):
  ''' Writes the file FilesRun.txt '''

  readFile = open(str(variablesPIM.directory.joinpath('standart.txt').resolve()))         # Open standard.txt file
  standard = readFile.read()                                                              # Read standard.txt file
  readFile.close()                                                                        # Close standard.txt file
  FilesRun = validationPIM.createFileWrite(variablesPIM.directory, f'filesRun{dirRun}.txt')             # Open filesRun.txt file in write mode
  FilesRun.write("#comments lines #\n")                                                   # Writes in the filesRun.txt file

  for k in range(0,2):
    for i in range(0,len(df)-1):
      for j in range(i+1,len(df)):
        if (df.loc[i,'camera'][0:3] != df.loc[j,'camera'][0:3]):
          print(df.loc[i,'camera'],df.loc[j,'camera'])
          fileCamera = standard[:]
          if (k==0):
            nameCamera = df.loc[i,'camera']+"_"+df.loc[j,'camera']
            fileCamera = fileCamera.replace("cam=cam","cam=1")
            camoption1 = i
          else:
            nameCamera = df.loc[j,'camera']+"_"+df.loc[i,'camera']
            fileCamera = fileCamera.replace("cam=cam","cam=2")
            camoption1 = j
          fileCamera = fileCamera.replace("ano=ano","ano="+str(dateM[0]))
          fileCamera = fileCamera.replace("mes=mes","mes="+str(dateM[1]))
          fileCamera = fileCamera.replace("dia=dia","dia="+str(dateM[2]))
          fileCamera = fileCamera.replace("hora=hora","hora="+str(dateM[3]))
          fileCamera = fileCamera.replace("minuto=minuto","minuto="+str(dateM[4]))
          fileCamera = fileCamera.replace("segundo=segundo","segundo="+str(dateM[5]))

          fileCamera = fileCamera.replace("meteorN=meteorN","meteorN="+dirRun+nameCamera)
          fileCamera = fileCamera.replace("option=option","option="+str(option))

          if (option == 1):
            fileCamera = fileCamera.replace("P1lat=P1lat","P1lat="+str(df.loc[camoption1,'_lat1']))
            fileCamera = fileCamera.replace("P1lon=P1lon","P1lon="+str(df.loc[camoption1,'_lng1']))
            fileCamera = fileCamera.replace("P1alt=P1alt","P1alt="+str(df.loc[camoption1,'_H1']))
            fileCamera = fileCamera.replace("P2lat=P2lat","P2lat="+str(df.loc[camoption1,'_lat2']))
            fileCamera = fileCamera.replace("P2lon=P2lon","P2lon="+str(df.loc[camoption1,'_lng2']))
            fileCamera = fileCamera.replace("P2alt=P2alt","P2alt="+str(df.loc[camoption1,'_H2']))
            fileCamera = fileCamera.replace("deltaT=deltaT","deltaT="+str(df.loc[camoption1,'dur']))
          
          if (option == 2 or option == 3):        
            fileCamera = fileCamera.replace("deltaT1=deltaT1","deltaT1="+str(df.loc[i,'dur']))
            fileCamera = fileCamera.replace("deltaT2=deltaT2","deltaT2="+str(df.loc[j,'dur']))
            fileCamera = fileCamera.replace("alt1=alt1","alt1="+str(df.loc[i,'alt']))
            fileCamera = fileCamera.replace("lon1=lon1","lon1="+str(df.loc[i,'lon']))
            fileCamera = fileCamera.replace("lat1=lat1","lat1="+str(df.loc[i,'lat']))
            fileCamera = fileCamera.replace("alt2=alt2","alt2="+str(df.loc[j,'alt']))
            fileCamera = fileCamera.replace("lon2=lon2","lon2="+str(df.loc[j,'lon']))
            fileCamera = fileCamera.replace("lat2=lat2","lat2="+str(df.loc[j,'lat']))

            if (option == 2):
              fileCamera = fileCamera.replace("az1Ini=az1Ini","az1Ini="+str(df.loc[i,'azIni']))
              fileCamera = fileCamera.replace("h1Ini=h1Ini","h1Ini="+str(df.loc[i,'elIni']))
              fileCamera = fileCamera.replace("az1Fin=az1Fin","az1Fin="+str(df.loc[i,'azFin']))
              fileCamera = fileCamera.replace("h1Fin=h1Fin","h1Fin="+str(df.loc[i,'elFin']))

              fileCamera = fileCamera.replace("az2Ini=az2Ini","az2Ini="+str(df.loc[j,'azIni']))
              fileCamera = fileCamera.replace("h2Ini=h2Ini","h2Ini="+str(df.loc[j,'elIni']))
              fileCamera = fileCamera.replace("az2Fin=az2Fin","az2Fin="+str(df.loc[j,'azFin']))
              fileCamera = fileCamera.replace("h2Fin=h2Fin","h2Fin="+str(df.loc[j,'elFin']))
            else:
              fileCamera = fileCamera.replace("ra1Ini=ra1Ini","ra1Ini="+str(df.loc[i,'Ra1']))
              fileCamera = fileCamera.replace("dec1Ini=dec1Ini","dec1Ini="+str(df.loc[i,'Dec1']))
              fileCamera = fileCamera.replace("ra2Ini=ra2Ini","ra2Ini="+str(df.loc[j,'Ra1']))
              fileCamera = fileCamera.replace("dec2Ini=dec2Ini","dec2Ini="+str(df.loc[j,'Dec1']))

              fileCamera = fileCamera.replace("ra1Fin=ra1Fin","ra1Fin="+str(df.loc[i,'Ra2']))
              fileCamera = fileCamera.replace("dec1Fin=dec1Fin","dec1Fin="+str(df.loc[i,'Dec2']))
              fileCamera = fileCamera.replace("ra2Fin=ra2Fin","ra2Fin="+str(df.loc[j,'Ra2']))
              fileCamera = fileCamera.replace("dec2Fin=dec2Fin","dec2Fin="+str(df.loc[j,'Dec2']))

          
          fileName = nameCamera+".txt"
          with validationPIM.createFileWrite(variablesPIM.directory, fileName) as infile:
            infile.write(fileCamera)

          shutil.copyfile(variablesPIM.directory.joinpath(fileName).resolve(),variablesPIM.directory.joinpath(dirRun).joinpath(fileName).resolve())
          FilesRun.write(fileName+"\n")


  FilesRun.write("#not delete this line #")
  FilesRun.close()
  shutil.copyfile(variablesPIM.directory.joinpath(f'filesRun{dirRun}.txt').resolve(),variablesPIM.directory.joinpath(dirRun).joinpath("filesRun.txt").resolve())

def createFiles(dirRun,dateM,option):
  """
  Parameters:

    dirRun: string, name of the directory to be created and where the files will be saved
    dateM: string, date of the analysis in format 'YYYY-MM-DD'
    option: string, option for the analysis ('manual' or 'auto')

  Return: None

  Raises:

    ValueError: if option parameter is not 'manual' or 'auto'
    FileNotFoundError: if dirRun directory does not exist

  Examples: createFiles('example_dir', '2023-04-28', 'auto')

  """
  cameraList = validationPIM.check_existence(variablesPIM.directory.joinpath(dirRun).resolve(), '.XML')       # Checks the existence and lists '.XML' files  
  print(cameraList)
  convertToDictionary(cameraList, dirRun)         # Convert data to a dictionary and pass this data to position.csv and frames.csv files
  dfP, dfF = saveDataInList(cameraList, dirRun)   # Save the data in list
  dfR = saveDataInVariables(dfP, dfF)     # Save data of list in variables
  df = dfR
  validationPIM.createDirIfDoesntExist(variablesPIM.directory, dirRun)                    # Creates a directory for the analysis if it does not exist
  df.to_excel(variablesPIM.directory.joinpath(dirRun).joinpath(dirRun+".xls").resolve())  # Save a sheets in directory
  writeFilesRun(df, dateM, option,dirRun)                                                 # Writes the variables in a copy of a model file
  print(f'The initial files for the integration of {dirRun} have been created.')  

def multiCreate(directoriesList,dateList,optionList):
  """
  This function receives three lists, containing directories, dates, and options. It loops through each element in the lists, calling the createFiles function for each element.

Parameters:

    directoriesList: A list containing directories as strings.
    dateList: A list containing dates as strings.
    optionList: A list containing options as strings.

Return:
    This function does not return anything.

Raises:
    This function does not raise any exceptions.

Examples:

    directories = ['dir1', 'dir2', 'dir3']
    dates = ['2022-01-01', '2022-01-02', '2022-01-03']
    options = ['option1', 'option2', 'option3']

    multiCreate(directories, dates, options)
  """
  for i in range(len(directoriesList)):
    createFiles(directoriesList[i],dateList[i],optionList[i])
