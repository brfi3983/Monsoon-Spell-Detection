import requests
import urllib.request
import time
import os
from bs4 import BeautifulSoup

variable_name = 'vwnd.sig995'
folder_name = variable_name + '/'
url = 'https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=74684&vid=278'
path = 'C:/Users/user/Desktop/original_datasets/' + folder_name
os.makedirs(path)

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

links = []
for link in soup.find_all('a'):
	links.append(link.get('href'))

print(links[54])
print(links[120])

print('\nDownload files...\n')
for i in range(54,121):
	print('Downloaded: ' + variable_name + '.' + str(1894+i) + '.nc')
	urllib.request.urlretrieve(links[i], path + variable_name + '.' + str(1894+i) + '.nc')	
	time.sleep(1)