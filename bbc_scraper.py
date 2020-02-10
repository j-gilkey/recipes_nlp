import requests
from bs4 import BeautifulSoup


def get_cuisine_type_links():
    page = requests.get("https://www.bbc.co.uk/food/cuisines")
    soup = BeautifulSoup(page.content, 'html.parser')

    list = soup.find_all(class_="promo promo__cuisine")
    link_list = []

    for item in list:
        link  = 'https://www.bbc.co.uk' + item['href'] + '/a-z'
        link_list.append(link)
        print(link)

    return link_list

    print('hello')




def get_alphabet_links():
    cuisine_links = get_cuisine_type_links()

    second_link = cuisine_links[1]


    link_list = []

    for link in cuisine_links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')

        test = soup.find_all(class_="tab-bar__tab gel-pica")

        if test:

            list = soup.find_all(class_="az-keyboard__link gel-pica-bold")
            link_list.append(link)

            for item in list:
                letter_link  = 'https://www.bbc.co.uk' + item['href']
                link_list.append(letter_link)

    #print(link_list)

    return link_list

def get_recipe_page_links():

    alphabet_links = get_alphabet_links()

    link_list = []

    for link in alphabet_links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')

        list = soup.find_all(class_="gel-layout__item gel-1/2 gel-1/3@m gel-1/4@xl")

        for item in list:
            #print(item.a['href'])
            recipe_link  = 'https://www.bbc.co.uk' + item.a['href']
            link_list.append(recipe_link)

    return(link_list)
        #must move the indent out one for the final run!!

def get_ingredients():

    recipe_links = get_recipe_page_links()

    recipe_list = []

    for link in recipe_links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')

        list = soup.find_all(class_="recipe-ingredients__list-item")

        ingredients = ''

        for item in list:
            ingredient = item.get_text()
            #print(ingredient)

            ingredients = ingredients + ingredient + ' '

        recipe_list.append(ingredients)

    return recipe_list


get_ingredients()




#get_recipe_page_links()
