--> Ativar o Ambiente Virtual:

```bash
source AmbientePIM-env/bin/activate
``` 

--> Desativar o Ambiente Virtual:

```bash
deactivate
``` 

1.  Para instalar as bibliotecas necessárias utilizar o comando:

$ pip install -r requirements.txt

2.  Para testar o código, é necessário rodar apenas o programa analiseDinamica.py


--> git:

1. Para linkar o repositorio local com o do github (remoto):
    git remote add origin {link do github} 

ou

1. Baixar repositorio remoto para sua maquina:
    git clone {link}
    
2. Baixar as atualizações do git (primeira coisa a se fazer no dia antes de começar as alterações):
    git pull origin {nome da branch}

3. Incluir arquivos na lista de modificados:
    git add {nome do arquivo/pasta ou .} 

4. Commit (Incluir uma nova versão):
    git commit -m "{mensagem}"

5. Atualizar repositorio remoto:
    git push origin {nome da branch}
