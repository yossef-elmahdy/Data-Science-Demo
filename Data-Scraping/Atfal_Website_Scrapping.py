import os
import json
import shutil
import requests
from bs4 import BeautifulSoup
from translate import Translator


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
    'البحر الأحمر':'Red Sea', 
    'البحر الاحمر':'Red Sea', 
    'بحر أحمر':'Red Sea', 
    'بحر احمر':'Red Sea', 
    'دمياط':'Damietta', 
    'الفيوم':'Fayoum', 
    'فيوم':'Fayoum', 
    'كفر الشيخ':'Kafr el-Sheikh', 
    'كفر شيخ':'Kafr el-Sheikh', 
    'مرسى مطروح':'Matrouh', 
    'مطروح':'Matrouh', 
    'مرسى':'Matrouh', 
    'المرسى':'Matrouh', 
    'الوادي الجديد':'New Valley', 
    'الوادى الجديد':'New Valley', 
    'الوادي':'New Valley', 
    'الوادى':'New Valley',
    'شمال سيناء':'North Sinai', 
    'شمال سينا':'North Sinai', 
    'جنوب سيناء':'South Sinai', 
    'جنوب سينا':'South Sinai', 
    'سيناء':'North Sinai', 
    'سينا':'North Sinai', 
    'السويس':'Suez', 
    'سويس':'Suez', 
    'قناة السويس':'Suez', 
    'قناه السويس':'Suez',
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
ARABIC_MAPPING = {
    'أ':'a', 
    'ا':'a',
    'إ':'e', 
    'ب':'b',
    'ت':'t',
    'ث':'th',
    'ج':'g',
    'ح':'h',
    'خ':'kh', 
    'د':'d', 
    'ذ':'th', 
    'ر':'r', 
    'ز':'z', 
    'س':'s', 
    'ش':'sh', 
    'ص':'s', 
    'ض':'d', 
    'ط':'t', 
    'ظ':'th', 
    'ع':'a', #or : aa
    'غ':'gh', 
    'ف':'f', 
    'ق':'q', #or : kh 
    'ك':'k', 
    'ل':'l', 
    'م':'m', 
    'ن':'n', 
    'ه':'h', 
    'و':'ou', 
    'ي':'i', 
    'ى':'a', 
    'ؤ':'ou', 
    'ء':'aa', 
    'ئ':'e', 
    'ة':'h', 
}

def extract_people_url(url):
    """
    Extract the page of each missing person on Atfal Mafkoda website based on the url. 

    Parameters
    ----------
    url : str
        the url of the page will be scrapped from Atfal Mafkoda website.

    Returns
    -------
    cnt : list
        list of the links of the content of each person on the page.

    """
    print('********   Extracting URL   ************')
    cnt = []
    response = requests.get(url)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup.find_all('button', {'class':'btn ebtn-4 ebtn-sm p-1','data-target':'#modal_persons_missing'}):
        cnt.append({"id": int(tag['data-id']), "URL": "https://atfalmafkoda.com" + tag['data-url']})
    return cnt

def find_gov(content):
    """
    Search for the government name in the arabic text content and return the arabic and english
    government name based on the GOVS_MAPPING dict.
    
    Parameters
    ----------
    content : str
        the text content scapped from the website.

    Returns
    -------
    gov_arabic : str
        the arabic name of the government.
    gov_english : str
        the english name of the government.

    """
    gov_arabic = ""
    gov_english = ""
    for gov in GOVS_MAPPING.keys():
        if content.find(gov) >= 0:
            gov_arabic = gov
            gov_english = GOVS_MAPPING[gov]
            return gov_arabic, gov_english 
    gov_arabic = 'مفقود'
    gov_english = 'Null'
    return gov_arabic, gov_english

def translate_content(content, from_language='ar', to_language='en'):
    """
    Translate the content (mostly: name, gov) string from one 
    language (mostly: 'ar') to another (mostly: 'en') using 
    translate open source library. 
    Note: the library has daily limited times of usage. 
    
    Parameters
    ----------
    content : str
        the content list to be translated (mostly: names, govs).
    from_language : str, optional
        the language of your content. The default is 'ar'.
    to_language : str, optional
        the language you need to translate to. The default is 'en'.

    Returns
    -------
    content_translated : str 
        string of the translated content.

    """
    
    translator = Translator(from_lang=from_language, to_lang=to_language)
    content_translated = translator.translate(content)

    return content_translated

   
def mapping_to_english(name):
    """
     Mapping the Arabic name to English using list of 
     pre-written dict according to ARABIC_MAPPING.
     
    Parameters
    ----------
    name : str
        list of Arabic names to be mapped.

    Returns
    -------
    names_mapped : str
        string of the mapped name.

    """
 
    mapped_name = ""
    for c in name: 
        try: 
            mapped_name += ARABIC_MAPPING[c]
        except: 
            mapped_name += c
    
    return mapped_name.title()


def extract_people_info(base, mapping_method="mapping"):
    """
    Extract the information from the information page of the person  (id, Name_Arabic, Name_English, Government_Arabic, 
    Government_English, Missing_Date, Current_Age, images).

    Parameters
    ----------
    base : dict
        the information data about the person. 
    mapping_method : str, optional
        the method of mapping the arabic name to english name. 
        methods: mapping (default), translating. 
                                               
    Returns
    -------
    base : dict
        the dict of the data about the person after appending the data.

    """
    
    print('********   Extracting INFO   ************')
    response = requests.get(base['URL'])
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    name_arabic = soup.find('h2', {"class":"person_name"}).text.strip()
    base['Name_Arabic'] = name_arabic
    if mapping_method=="translating": 
        base['Name_English'] = translate_content(name_arabic)    
    elif mapping_method=="mapping":
        base['Name_English'] = mapping_to_english(name_arabic) 
    else: 
        base['Name_English'] = mapping_to_english(name_arabic) 
    print(base['Name_English'] + ", " +  base['Name_Arabic']) 
    base['Government_Arabic'], base['Government_English'] = find_gov(soup.find('p', {'class': 'date_loss'}).text)
    cnt = soup.find_all('h4', {"class":"date_loss"})
    base['Missing_Date'] = cnt[0].text.replace("\n", "").strip()
    base['Current_Age'] = cnt[1].text.replace("\n", "").strip()
    base['image'] = []
    for photo in soup.find_all('img', {"class": "img-fluid"}):
        if photo['alt'] == base['Name_Arabic']:
            base['image'].append("https://atfalmafkoda.com/" + photo['src'])
    return base

def downlad_extracted_img(base, save_path):
    """
    Download the images that are extracted from the person content. 

    Parameters
    ----------
    base : dict
         the information data about the person..
    save_path : str
        the path (directory) the data will be saved in it.

    Returns
    -------
    None.

    """
    imageURLs = base['image']
    id = base['id']
    sub = len(str(id))
    fileName = base['Name_Arabic']
    os.makedirs((f'{save_path}/{fileName}'), exist_ok=True)
    imgName = (4 - sub) * '0' + str(id) + '_'
    base['imageRef'] = imgName + '0' + '.jpg'
    base['imageRefExtra'] = []
    n = len(imageURLs)
    # Prevent any duplicated images by just downloading half of them
    if n%2 == 0: 
        n = n//2
    else: 
        n = n//2 + 1
    for i in range(n):
        imageURL = imageURLs[i]
        r = requests.get(imageURL, stream=True)
        r.raw.decode_content = True
        currentImagName= imgName + str(i) + '.jpg'
        print(f"Downloading {fileName} ---- {currentImagName} .....")
        base['imageRefExtra'].append(currentImagName)
        with open(f'{save_path}/{fileName}/{currentImagName}', 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    

def extract_people_info_download_image(pageURL, save_path):
    """
    Extract and download the people information in the page. 

    Parameters
    ----------
    pageURL : str
        the page url to be extracted.
    save_path : str
        the path (directory) the data will be saved in it.

    Returns
    -------
    peapleInfo : list
        list of all the people information in the page.

    """
    cnt = extract_people_url(pageURL)
    peapleInfo = []
    for base in cnt:
        personInfo = extract_people_info(base)
        downlad_extracted_img(personInfo, save_path)
        peapleInfo.append(personInfo)
    return peapleInfo

def ExtractMissingPeopleInfoT0Json(save_path="./Scrapped_Data", number_of_pages=1):
    """
    Extract the information from all pages (limited bt number_of_pages) and save 
    to JSON file in the same directory. 

    Parameters
    ----------
    save_path : str, optional
        the path (directory) the data will be saved in it.
        default: the current director/Scrapped_Data.
    number_of_pages : int, optional
        number of pages you want to scrape. The default is 1.

    Returns
    -------
    None.

    """
    page = 1
    data = []
    if number_of_pages == -1: number_of_pages = 90
    while page <= number_of_pages:
        data += extract_people_info_download_image(f"https://atfalmafkoda.com/ar/seen-him?page={page}&per-page=18", save_path)
        page+=1
    print("="*70)
    print("\n==>All images are scrapped and downloaded successfully in directory: {}".format(save_path))
    with open(f"{save_path}/missing_people.json",'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii = False)
    print("\n==>JSON file with all scrapped data is successfully downloaded in directory: {}".format(save_path))

if __name__ == '__main__': 
    SAVE_DIR = r".\Scrapped_Data"   # You may need to change the SAVE_DIR to another directory
    ExtractMissingPeopleInfoT0Json(SAVE_DIR)