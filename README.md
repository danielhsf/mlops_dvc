# Aula 1 de DVC
Primeira prática sobre uso do DVC

## Instalação
Crie um ambiente virtual para instala as bibliotecas necessárias
```
user@user$ python3 -m venv dvc_env
user@user$ source dvc_env/bin/activate
(dvc_env) user@user$
```
Instale as bibliotecas necessarias usando o arquivo de requirements
```
(dvc_env) user@user$ pip install -r requirements.txt
```
## Primeiros passos

Adicione arquivos de dataset que gostaria de versionar usando o dvc, para esse projeto, colocamos parte do dataset do MNIST em um pasta chamada data/
Adicione a pasta data/ usando "dvc add" para que o dvc comece a monitor a pasta
```
dvc add data/
```

O passo seguinte será de configuração com o gdrive.

## Configurando a comunicação com o gdrive
No Google Drive, defina um local para armazenar os dados. Copie o final da URL (FINAL_URL) para o dvc.
```
(dvc_env) user@user$ dvc remote add -d gdrive gdrive://FINAL_URL
```
Troque os campos YOUR_CLIENT_ID e CLIENT_SECRET pelas configurações da sua máquina

```
(dvc_env) user@user$ dvc remote modify --local gdrive gdrive_client_id "YOUR_CLIENT_ID"
```
```
(dvc_env) user@user$ dvc remote modify --local gdrive gdrive_client_secret "CLIENT_SECRET"
```

## Faça o Push pro repositorio DVC

Com as configurações definidas, faça o push pro repositorio dvc com o seguinte comando:
```
(dvc_env) user@user$ dvc push
```