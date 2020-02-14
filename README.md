# recipes_nlp

![Italy](/vis/Italy_Wordcloud.png)

![India](/vis/India_Wordcloud.png)


Initial goal of this project was to use Natural Language Processing in order to classify food recipes by cuisine type. However, we have done a little more than that. 

Just by looking at the ingredient list of a certain dish, can a machine learning model accurately predict which cuisine type this dish belongs to? How do world cuisines differ? Are there more ways to classify food recipes?

We started our project by scraping two popular websites, BBC Food and Allrecipes, which resulted in over 2000 observations in 13 cuisine types: American, British, Chinese, Japanese, Indian, Nordic, Pakistani, African, French, Italian, Mexican, Irish. Intuitively, we understood that Asian cuisines are different from western food, but how distinct are seemingly similar classes, such as Chinese and Japanese? 

# EDA and Initial modeling

![Count](/vis/recipe_count_cuisine.png)

![Scatterplot](/vis/Scatterplot)

Our EDA showed that there is some class imbalance in the dataset, for instance, there were significantly fewer Nordic and Pakistani recipes compared to the rest of the groups. Also, dimensionality reduction and creating a scatter plot showed us that American cuisine imitates all others too much, which may confuse our model. We thought that running our model with those classes and then without them will result in improvement in both accuracy and recall.

Classification was done using three different models: Naive Bayes, Logistic Regression, and SVM with Stochastic Gradient Descent. SVM model both improved the most and showed the best numbers: 72% accuracy and good recall and precision in most categories. 

# Unsupervised Clustering 

However, no other feature engineering improved our model any further. So we decided to use unsupervised clustering to look at the natural groupings of our data. Turns out, that if we set the number of clusters to 2, there is a great separation between desserts and savory food! And when the number of groups is equal to 6, there is a separation between the following clusters: Indian/Mexican, Italian, French, Desserts, Japanese/Chinese, Drinks. 

# Markov chain and recipe generator 

Finally, we wanted to check if a neural net (specifically, a Markov chain) can learn how to create a recipe and give us something meaningful. Turns out, the best it can do is the following:
'2 loins of lamb or mutton neck fillet, diced, 3 tbsp tomato purée, 1 tbsp dill seeds, 1 tsp sea salt, plus extra leaves to serve, salt and pepper'
'2 tbsp crème fraîche, 300g/10½ oz fettucine or tagliatelle, cooked according to packet instructions'
'200g/7oz white sweet potatoes, 12-10 inch flour tortillas, 9 ounces shredded Cheddar cheese'
'2 Japanese aubergine cut into very small florets, 1 garlic clove roughly chopped to garnish’

We believe that in order to create better recipes a neural net should train on a more homogeneous dataset, say, on desserts only.
 
