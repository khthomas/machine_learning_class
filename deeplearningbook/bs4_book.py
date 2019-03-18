import httplib2
from bs4 import BeautifulSoup, SoupStrainer

book_link = 'http://www.deeplearningbook.org/'

http = httplib2.Http()
status, response = http.request(link)

links = []
for link in BeautifulSoup(response, parse_only=SoupStrainer('a')):
    print(link['href'])
links


resp = urllib2.urlopen(book_link)
soup = BeautifulSoup(resp, from_encoding=resp.info().getparam('charset'))

from bs4 import BeautifulSoup
import urllib.request
import urllib.urlretrieve

resp = urllib.request.urlopen(book_link)
soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))

link2 = []
lnames = []
for link in soup.find_all('a', href=True):
    link2.append(link['href'])
    ll = str(link).split(">")[1].split("<")[0]
    lnames.append(ll)


import os
for l in range(8,len(link2)):
    name = str(lnames[l] + ".html").replace(" ","_")
    save_name = '/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/deeplearningbook/' + lnames[l]
    url_name = book_link + link2[l]
    print(name)
    call_this = f'wget -O {name} {url_name}'
    os.system(call_this)


os.system()