import requests
from bs4 import BeautifulSoup  # xpath is better but too lazy
import random
import time
import re
import json

# this script collects a list of top 5000 popular codegolf questions on the
# codegolf.exchange website, using requests and beautifulsoup.
# each question page contains multiple solutions with different
# programming languages
# for a specific task. For each question page, this script collects all
# solutions using python and golfscript
# that have high votes. All question list and topic-solutions
# are saved into files.
# the script shows how original data was collected.


def get_golfs(posts, lang):
    # posts: list of small soups
    # lang: tar PL
    golfs = []
    for apost in posts:
        if ((apost.h1) and (lang in apost.h1.text)) or ((apost.h2) and (lang in apost.h2.text)) or ((apost.h3) and (lang in apost.h3.text)):
            golfs.append(apost.code.text)
    return golfs


def get_page(url_token, maxpage=5):
    topic = url_token.split('/')[-1]
    # url_token something like this /questions/107937/war-is-peace-freedom-is-slavery-ignorance-is-strength'
    Python_out = []
    Gs_out = []

    for pn in range(1, maxpage+1):
        url = f"https://codegolf.stackexchange.com/{url_token}?page={pn}&tab=votes#tab-top"

        soup = BeautifulSoup(requests.get(url, verify=False).content, "html.parser")
        posts = soup.find_all('div', {'class': "answercell"}, limit=None)

        Python_out += get_golfs(posts, 'Python')
        Gs_out += get_golfs(posts, 'GolfScript')

        t = random.randint(100, 300)/300
        print(f"sleeping for {t} sec...(between each page of the question)")
        time.sleep(t)

    return topic, {'Python': list(set(Python_out)),
                   'GolfScript': list(set(Gs_out))}

# func get all questions/topics/index


def get_topic_lst(tar):
    main_soup = BeautifulSoup(requests.get(tar, verify=False).content, "html.parser")

    candi_urls = []
    pattern = re.compile(r'^\/questions\/[0-9]+\/', re.I)
    for a in main_soup.find_all('a', href=True):
        candi = a['href']
        if bool(pattern.search(candi)):
            candi_urls.append(candi)
    return candi_urls


if __name__ == "__main__":

    page_size = 50
    all_urls = []  # all urls to be scraped

    for page_num in range(1, 101):
        print(f"working on {page_num}:")
        t = random.randint(100, 300)/100
        print(f"sleeping for {t} sec...")
        time.sleep(t)
        index_page = f"https://codegolf.stackexchange.com/questions/tagged/code-golf?tab=votes&page={page_num}&pagesize={page_size}"
        topic50 = get_topic_lst(index_page)

        all_urls += topic50
        if len(topic50) != 50:
            print(f"so far collected {len(all_urls)} urls")
        print("first post is :", topic50[0])

    # save list for future purpose
    with open('codegolf_questions.txt', 'w') as f:
        for item in all_urls:
            f.write("%s\n" % item)

    # start scrapping
    data = {}
    for question in all_urls:
        print(f"working on {question}:")
        t = random.randint(100, 300)/300
        print(f"sleeping for {t} sec...(between each question)")
        time.sleep(t)

        key, plgolfs = get_page(question)
        data[key] = plgolfs
    # solutions saved to json
    with open('golfs.json', 'w') as f:
        json.dump(data, f, indent=4)
