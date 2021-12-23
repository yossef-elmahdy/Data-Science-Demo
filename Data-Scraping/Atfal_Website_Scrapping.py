import os
import json
import shutil
import requests
from bs4 import BeautifulSoup


def extract_people_url(url):
    print('********   Extracting URL   ************')
    cnt = []
    response = requests.get(url)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup.find_all('button', {'class':'btn ebtn-4 ebtn-sm p-1','data-target':'#modal_persons_missing'}):
        cnt.append({"id": int(tag['data-id']), "URL": "https://atfalmafkoda.com"+tag['data-url']})
    return cnt

def extract_people_info(base):
    print('********   Extracting INFO   ************')
    response = requests.get(base['URL'])
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    base['Name'] = soup.find('h2', {"class":"person_name"}).text.strip()
    cnt = soup.find_all('h4', {"class":"date_loss"})
    base['missingDate'] = cnt[0].text
    base['currentAge'] = cnt[1].text
    base['image'] = []
    for photo in soup.find_all('img', {"class": "img-fluid"}):
        if photo['alt'] == base['Name']:
            base['image'].append("https://atfalmafkoda.com/" + photo['src'])
    return base

def downlad_extracted_img(base):
    imageURLs = base['image']
    id = base['id']
    sub = len(str(id))
    fileName = base['Name']
    os.makedirs((f'./o/{fileName}'), exist_ok=True)
    imgName = (4 - sub) * '0' + str(id) + '_'
    base['imageRef'] = imgName + '0' + '.jpg'
    base['imageRefExtra'] = []
    n = len(imageURLs)
    for i in range(n):
        imageURL = imageURLs[i]
        r = requests.get(imageURL, stream=True)
        r.raw.decode_content = True
        currentImagName= imgName + str(i) + '.jpg'
        print(f"Downloading {fileName} ---- {currentImagName} .....")
        base['imageRefExtra'].append(currentImagName)
        with open(f'./o/{fileName}/{currentImagName}', 'wb') as f:
            shutil.copyfileobj(r.raw, f)

def extract_people_info_download_image(pageURL):
    cnt = extract_people_url(pageURL)
    peapleInfo = []
    for base in cnt:
        personInfo = extract_people_info(base)
        downlad_extracted_img(personInfo)
        peapleInfo.append(personInfo)
    return peapleInfo

def ExtractMissingPeopleInfoT0Json():
    page = 1
    data = []
    while page <= 1:
        data +=extract_people_info_download_image(f"https://atfalmafkoda.com/ar/seen-him?page={page}&per-page=18")
        page+=1
    with open("missing_peaple.json",'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii = False)

ExtractMissingPeopleInfoT0Json()