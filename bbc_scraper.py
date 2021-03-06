import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_cuisine_type_links():
    #get a page link for each cuisine classification on bbc.com
    page = requests.get("https://www.bbc.co.uk/food/cuisines")
    #automatically goes to the BBC cuisine homepage
    soup = BeautifulSoup(page.content, 'html.parser')

    list = soup.find_all(class_="promo promo__cuisine")
    link_list = []
    for item in list:
        cuisine = item['href'].replace('/food/cuisines/','')
        link  = 'https://www.bbc.co.uk' + item['href'] + '/a-z'
        link_list.append([link, cuisine])
        #print(link)

    return link_list


def get_alphabet_links(cuisine_link):
    #gets a link for each cooresponding alphabetized page from a given cuisines link

    link_list = []
    page = requests.get(cuisine_link)
    soup = BeautifulSoup(page.content, 'html.parser')
    test = soup.find_all(class_="tab-bar__tab gel-pica")

    if test:
        list = soup.find_all(class_="az-keyboard__link gel-pica-bold")
        link_list.append(cuisine_link)

        for item in list:
            letter_link  = 'https://www.bbc.co.uk' + item['href']
            link_list.append(letter_link)

    return link_list

def get_recipe_page_links(alphabet_link):
    #get all recipe links on a given alpabetized page

    link_list = []
    page = requests.get(alphabet_link)
    soup = BeautifulSoup(page.content, 'html.parser')
    list = soup.find_all(class_="gel-layout__item gel-1/2 gel-1/3@m gel-1/4@xl")

    for item in list:
        recipe_link  = 'https://www.bbc.co.uk' + item.a['href']
        link_list.append(recipe_link)

    return(link_list)
        #must move the indent out one for the final run!!

def get_ingredients(recipe_link):
    #get the actual recipe ingredients for a given recipe

    page = requests.get(recipe_link)
    soup = BeautifulSoup(page.content, 'html.parser')
    list = soup.find_all(class_="recipe-ingredients__list-item")
    ingredients = ''

    for item in list:
        ingredient = item.get_text()
        ingredients = ingredients + ingredient + ' '

    return ingredients

def bbc_scrape_wrapper():
    #wrapper function that calls the above functions and iterates through them
    #due to the slow speed of the scrape it is recommended to run cuisine pages one at a time
    cuisine_links = [['https://www.bbc.co.uk/food/cuisines/pakistani/a-z', 'pakistani']]

    '''cuisine_links = [['https://www.bbc.co.uk/food/cuisines/african/a-z', 'african'],
                    ['https://www.bbc.co.uk/food/cuisines/american/a-z', 'american'],
                    ['https://www.bbc.co.uk/food/cuisines/british/a-z', 'british'],
                    ['https://www.bbc.co.uk/food/cuisines/chinese/a-z', 'chinese'],
                    ['https://www.bbc.co.uk/food/cuisines/french/a-z', 'french'],
                    ['https://www.bbc.co.uk/food/cuisines/indian/a-z', 'indian'],
                    ['https://www.bbc.co.uk/food/cuisines/irish/a-z', 'irish'],
                    ['https://www.bbc.co.uk/food/cuisines/italian/a-z', 'italian'],
                    ['https://www.bbc.co.uk/food/cuisines/japanese/a-z', 'japanese'],
                    ['https://www.bbc.co.uk/food/cuisines/korean/a-z', 'korean'],
                    ['https://www.bbc.co.uk/food/cuisines/mexican/a-z', 'mexican'],
                    ['https://www.bbc.co.uk/food/cuisines/nordic/a-z', 'nordic'],
                    ['https://www.bbc.co.uk/food/cuisines/north_african/a-z', 'north_african'],
                    ['https://www.bbc.co.uk/food/cuisines/pakistani/a-z', 'pakistani'],
                    ['https://www.bbc.co.uk/food/cuisines/thai_and_south-east_asian/a-z', 'thai_and_south-east_asian']]'''

    final_list = []

    i=0

    for cuisine_link in cuisine_links:
        alphabet_links = get_alphabet_links(cuisine_link[0])
        cuisine_type = cuisine_link[1]

        for alphabet_link in alphabet_links:
            recipe_links = get_recipe_page_links(alphabet_link)

            for recipe_link in recipe_links:
                ingredients = get_ingredients(recipe_link)
                final_list.append((ingredients, cuisine_type))
                #print(final_list)

    return final_list
