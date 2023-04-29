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
import variaveisPIM  


# Verfica a existência do diretório pedido e o cria, caso não exista
def createDirIfDoesntExist(diretorioBase, diretorio):  
  if not pathlib.Path(diretorioBase.joinpath(diretorio)).is_dir():
    os.mkdir(str(diretorioBase.joinpath(diretorio).resolve()))


# Verifica a existência de arquivos requisitos do programa
def checkExistence(diretorioDoArq, extensaoArq):   
    lista = []
    for _, _, arquivo in os.walk(diretorioDoArq):
        for i in arquivo:
            if ( (extensaoArq in i)):
                lista.append(i)
    if lista == []:
        print("Error: Não existem arquivos .XML para a integração")
        exit()
    else: 
        return lista

# Cria arquivos em modo escrita
def criarArq(diretorioDoArq, nome):
    return open(str(variaveisPIM.diretorio.joinpath(diretorioDoArq).joinpath(nome).resolve()),'w+')

