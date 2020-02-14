
import pandas as pd

def create_overall_dataframe():
    #takes in each individual cuisine

    amer = pd.read_csv('cuisine_csv/amer.csv' )
    british = pd.read_csv('cuisine_csv/british.csv' )
    chinese = pd.read_csv('cuisine_csv/chinese.csv' )
    french = pd.read_csv('cuisine_csv/french.csv' )
    indian = pd.read_csv('cuisine_csv/indian.csv' )
    irish = pd.read_csv('cuisine_csv/irish.csv' )
    italian = pd.read_csv('cuisine_csv/italian.csv' )
    japanese = pd.read_csv('cuisine_csv/japanese.csv' )
    mexican = pd.read_csv('cuisine_csv/mexican.csv' )
    nordic = pd.read_csv('cuisine_csv/nordic.csv' )
    north_african = pd.read_csv('cuisine_csv/north_african.csv' )
    pakistani = pd.read_csv('cuisine_csv/pakistani.csv' )

    cuisines = [amer, british, chinese, french, indian, irish, italian, japanese, mexican, nordic, north_african, pakistani]

    overall_df = pd.concat(cuisines)

    cuisine_map = {'american': 6, 'british' : 7, 'chinese' : 8, 'french' : 5, 'indian' : 9, 'irish' : 10, 'italian' : 3,
                'japanese' : 13, 'mexican' : 2, 'nordic' : 11, 'north_african' : 4, 'pakistani' : 12}

    #print(overall_df.head)

    overall_df['type'] = overall_df['type'].replace(cuisine_map)

    #print(overall_df.head)

    overall_df.to_csv('overall_bbc.csv', index=False)

create_overall_dataframe()
