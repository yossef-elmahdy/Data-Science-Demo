import os
import json
import shutil
import requests
from bs4 import BeautifulSoup


GOVS_MAPPING = {
    'اسيوط':'Assiut',
    'أسيوط':'Assiut',
    'الجيزه':'Giza',
    'الجيزة':'Giza',
    'جيزة':'Giza',
    'جيزه':'Giza',
    'أسوان':'Aswan', 
    'اسوان':'Aswan', 
    'الاسكندرية':'Alexandria', 
    'الاسكندريه':'Alexandria', 
    'الإسكندرية':'Alexandria', 
    'الإسكندريه':'Alexandria', 
    'الأسكندرية':'Alexandria', 
    'الأسكندريه':'Alexandria', 
    'اسكندرية':'Alexandria', 
    'اسكندريه':'Alexandria', 
    'أسكندرية':'Alexandria', 
    'أسكندريه':'Alexandria', 
    'إسكندرية':'Alexandria',
    'إسكندريه':'Alexandria', 
    'المنيا':'Minya', 
    'منيا':'Minya', 
    'القاهرة':'Cairo', 
    'القاهره':'Cairo', 
    'قاهرة':'Cairo', 
    'قاهره':'Cairo', 
    'الدقهلية':'Dakahlia', 
    'الدقهليه':'Dakahlia', 
    'دقهلية':'Dakahlia', 
    'دقهليه':'Dakahlia', 
    'سوهاج':'Sohag', 
    'الغربية':'Gharbia', 
    'الغربيه':'Gharbia', 
    'غربية':'Gharbia', 
    'غربيه':'Gharbia', 
    'البحيرة':'Beheira', 
    'البحيره':'Beheira', 
    'بحيرة':'Beheira', 
    'بحيره':'Beheira', 
    'القليوبية':'Qualyubia', 
    'القليوبيه':'Qualyubia', 
    'قليوبية':'Qualyubia', 
    'قليوبيه':'Qualyubia', 
    'الشرقية':'Al-Sharqia', 
    'الشرقيه':'Al-Sharqia', 
    'شرقية':'Al-Sharqia', 
    'شرقيه':'Al-Sharqia', 
    'المنوفية':'Menofia', 
    'المنوفيه':'Menofia', 
    'منوفية':'Menofia',
    'منوفيه':'Menofia', 
    'بني سويف':'Beni Suef', 
    'بنى سويف':'Beni Suef',
    'سويف':'Beni Suef',
    'قنا':'Qena', 
    'بور سعيد':'Port Said',
    'بور':'Port Said',
    'سعيد':'Port Said',
    'البحر الأحمر':'Red Sea', 
    'البحر الاحمر':'Red Sea', 
    'بحر أحمر':'Red Sea', 
    'بحر احمر':'Red Sea', 
    'البحر':'Red Sea',
    'الأحمر':'Red Sea', 
    'الاحمر':'Red Sea', 
    'أحمر':'Red Sea', 
    'احمر':'Red Sea', 
    'دمياط':'Damietta', 
    'الفيوم':'Fayoum', 
    'فيوم':'Fayoum', 
    'كفر الشيخ':'Kafr el-Sheikh', 
    'كفر شيخ':'Kafr el-Sheikh', 
    'كفر':'Kafr el-Sheikh',
    'شيخ':'Kafr el-Sheikh',
    'الشيخ':'Kafr el-Sheikh',
    'مرسى مطروح':'Matrouh', 
    'مطروح':'Matrouh', 
    'مرسى':'Matrouh', 
    'المرسى':'Matrouh', 
    'الوادي الجديد':'New Valley', 
    'الوادى الجديد':'New Valley', 
    'وادي':'New Valley', 
    'وادى':'New Valley', 
    'الوادي':'New Valley', 
    'الوادى':'New Valley',
    'الجديد':'New Valley', 
    'جديد':'New Valley', 
    'شمال سيناء':'North Sinai', 
    'شمال سينا':'North Sinai', 
    'شمال':'North Sinai', 
    'الشمال':'North Sinai', 
    'جنوب سيناء':'South Sinai', 
    'جنوب سينا':'South Sinai', 
    'جنوب':'South Sinai', 
    'الجنوب':'South Sinai',
    'سيناء':'North Sinai', 
    'سينا':'North Sinai', 
    'السويس':'Suez', 
    'سويس':'Suez', 
    'قناة السويس':'Suez', 
    'قناه السويس':'Suez',
    'القناة':'Suez',
    'القناه':'Suez',
    'الأقصر':'Luxor', 
    'الاقصر':'Luxor', 
    'أقصر':'Luxor', 
    'اقصر':'Luxor', 
    'الاسماعيلية':'Ismailia', 
    'الاسماعيليه':'Ismailia',
    'الإسماعيلية':'Ismailia', 
    'الإسماعيليه':'Ismailia', 
    'الأسماعيلية':'Ismailia', 
    'الأسماعيليه':'Ismailia', 
    'اسماعيلية':'Ismailia', 
    'اسماعيليه':'Ismailia', 
    'إسماعيلية':'Ismailia', 
    'إسماعيليه':'Ismailia',
    'أسماعيلية':'Ismailia', 
    'أسماعيليه':'Ismailia', 
    'الاسماعلية':'Ismailia', 
    'الاسماعليه':'Ismailia',
    'الإسماعلية':'Ismailia', 
    'الإسماعليه':'Ismailia', 
    'الأسماعلية':'Ismailia', 
    'الأسماعليه':'Ismailia', 
    'اسماعلية':'Ismailia', 
    'اسماعليه':'Ismailia', 
    'إسماعلية':'Ismailia', 
    'إسماعليه':'Ismailia',
    'أسماعلية':'Ismailia', 
    'أسماعليه':'Ismailia', 
    'مفقود':'Null'
}

def extract_people_url(url):
    print('********   Extracting URL   ************')
    cnt = []
    response = requests.get(url)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup.find_all('button', {'class':'btn ebtn-4 ebtn-sm p-1','data-target':'#modal_persons_missing'}):
        cnt.append({"id": int(tag['data-id']), "URL": "https://atfalmafkoda.com"+tag['data-url']})
    return cnt

def find_gov(content): 
    for gov in GOVS_MAPPING.keys():
        if content.find(gov) >= 0: 
            return gov, GOVS_MAPPING[gov]
    return 'مفقود', 'Null'

def extract_people_info(base):
    print('********   Extracting INFO   ************')
    response = requests.get(base['URL'])
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    base['Name'] = soup.find('h2', {"class":"person_name"}).text.strip()
    base['Government_Arabic'], base['Government_English'] = find_gov(soup.find('p', {'class': 'date_loss'}).text)
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
    os.makedirs((f'./Scrapped_Data/{fileName}'), exist_ok=True)
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
        with open(f'./Scrapped_Data/{fileName}/{currentImagName}', 'wb') as f:
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

if __name__ == '__main__': 
    SAVE_DIR = "./Scrapped_Data"
    ExtractMissingPeopleInfoT0Json()