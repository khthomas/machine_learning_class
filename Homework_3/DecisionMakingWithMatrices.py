# Decision making with Matrices
# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.

import numpy as np
import pickle
from scipy.stats import rankdata

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.
people = {'Jane': {'willingness to travel': 1,
                  'desire for new experience':0,
                  'cost':0,
                  'indian food':1,
                  'Mexican food':1,
                  'hipster points':3,
                  'vegetarian': 3,
                  },

          }

names  = ['Kyle', 'Thomas', 'George', 'Max', 'Mary', 'Martha', 'Tacitus', 'Emily', 'Ruth', 'Jane']
cats = ['Close Distance', 'Desire for New Experience', 'Cost', 'American', 'Asian', 'Italian', 'Indian', 'Mexican', 'Hipster Points', 'Vegetarian']

def make_people_dict(names, categories):
    people = {}

    for n in names:
        random_prefs = np.random.randint(0, 11, len(categories))
        dict_input = dict(zip(categories, random_prefs))
        people[n] = dict_input
    
    return people

people = make_people_dict(names, cats)



# Transform the user data into a matrix(M_people). Keep track of column and row ids.


# Next you collected data from an internet website. You got the following information.

# resturants  = {'flacos':{'distance' :
#                         'novelty' :
#                         'cost':
#                         'average rating':
#                         'cuisine':
#                         'vegetarian'
#                         }

# }

rests = ['Big Johns Pizza', 'Dames Chicken and Waffels', 'Pho King', 'Sandwhich Stop', 'Valu Buffet', 'Burger Fi', 'Fish Market', 'Curry Stop']

resturants = make_people_dict(rests, cats)

##############################################
### QUESTIONS 
##############################################


# Transform the restaurant data into a matrix(M_resturants) use the same column index.
M_resturants = np.array([list(resturants[r].values()) for r in rests])
M_people = np.array([list(people[p].values()) for p in people])

# if using pickle
with open('/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/Homework_3/M_people', 'rb') as f:
    M_people = pickle.load(f)

# Home laptop
with open('/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/Homework_3/M_people', 'rb') as f:
    M_people = pickle.load(f)

with open('/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/Homework_3/M_rests', 'rb') as f:
    M_resturants = pickle.load(f)

with open('/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/Homework_3/M_rests', 'rb') as f:
    M_resturants = pickle.load(f)

# The most imporant idea in this project is the idea of a linear combination.
# Informally describe what a linear combination is and how it will relate to our resturant matrix.
# A linear combination is defined as "is an expression constructed from a set of terms by multiplying each term by a 
# constant and adding the results" (or perhaps more intuitively scaling and adding the basis vectors) [Wikipedia]. 
# it is a linear combination becuase if you hold one of the two scalers and let the other change freely your result with be a 
# line. Sacling both of them lets you reach any point on the plane (For example, the span of most 2D vectors is all of 2D space).
# In relation to our resturant example we will find the linear combination that results in a maximum value across all resturants 
# and across all preferences. 

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.
# I will calculate the top resturant for Kyle.
# The output vectors, called scores below, show my ranking of each resuturant based on my preference and the resturants features in realtion
# to my preferences. The resturant with the highest value is my top choice, the resturant with the lowest value is my last choice.
# Respectively these are: Curry Stop (top), Fish Market (last choice)

kyle = M_people[0]
type(kyle)

np.matmul(kyle, M_resturants.T)

def get_scores(person, rests_matrix, rest_names_list):
    r_scores = np.matmul(person, rests_matrix.T)
    output = dict(zip(rest_names_list, r_scores))
    sorted_output = sorted(output.items(), key = lambda x: x[1], reverse=True)
    return r_scores, output, sorted_output

scores, kyle_favs, sorted_kyle = get_scores(kyle, M_resturants, rests)
kyle_top_choce = sorted_kyle[0]
kyle_last_choice = sorted_kyle[-1]

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
M_usr_x_rest = np.matmul(M_people, M_resturants.T)

# This new martix represents each score by person an resturant
# The rows are people, the columns are resturants
# Assuming that indexing starts at 0:
# So entry a_01 would be person = Kyle, resturant = ig Johns Pizza, and the value is the result the matricies multiple together. 
# It represents the score of the perons preferences at that resturaunt.
# i = people, k = resturanuts, a = score


# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
sum_rests = M_usr_x_rest.sum(axis=0)
top_rest_index = list(sum_rests).index(max(sum_rests))
top_rest = rests[top_rest_index]
top_rest
# the analysis the best resturant was index 4 or Valu Buffet.


# each summation represents the cumulative score of resturants for all peoples weights. According to the inputs when I ran

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.
# after talking to John what I need to do here is rank each persons perfernce from 1 to the number of resturants.
# do it for each person, and then find the best value. Its a similar idea but a different scoring method.
# M_usr_x_rest_rank = np.amax(M_usr_x_rest, axis=1) # this finds the max value per person
M_usr_x_rest_rank = np.array( [rankdata(r) for r in M_usr_x_rest ] ) 
sum_rests_rank = M_usr_x_rest_rank.sum(axis=0)
top_rest_index_rank = list(sum_rests_rank).index(max(sum_rests_rank))
top_rest_rank = rests[top_rest_index_rank]
top_rest_rank

# result is still Valu Buffet, but there are also ties in some of the rankings



# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
# In my instance there is not a difference between the two. There is the problem of ties.

# How should you preprocess your data to remove this problem.
# I would not allow ties and perhaps limit the cumulative response that people can give.

# Find user profiles that are problematic, explain why?
# To do

# Think of two metrics to compute the disatistifaction with the group.
# Percent of people where the choice is in the bottom half of resturants
# Delta between chosen resturant and top resturant
# 

# Should you split in two groups today?

# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
# Intuitively, you remove the price component, will need to recalculate the rests matrix.
cost_index = cats.index('Cost')
#drop column == cost index from M_resturants
M_resturants_no_cost = np.delete(M_resturants, cost_index, axis=1)
M_people_no_cost = np.delete(M_people, cost_index, axis=1)


top_cost_no_object = np.matmul(M_people_no_cost, M_resturants_no_cost.T)
sum_rests_no_cost = top_cost_no_object.sum(axis=0)
top_rest_index_no_cost = list(sum_rests_no_cost).index(max(sum_rests_no_cost))
top_rest_no_cost = rests[top_rest_index_no_cost]
top_rest_no_cost

# Now the best resturant, when I ran this analysis the first time, is Big Johns Pizza




# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?
# you should be able to solve for it, I think there is a numpy built in.

# First simulate the results, I am going to make is square for this example
names_square = ["Abe", "Beth", "Carlos", "Danni"]
cats_square = ["Yummy", "Cheap", "Close", "Drinks"]
rests_square = ["Alpha", "Bravo", "Charlie", "Delta"]

people_square = make_people_dict(names_square, cats_square)
rests_sqr = make_people_dict(rests_square, cats_square)

M_resturants_square = np.array([list(rests_sqr[r].values()) for r in rests_square])
M_people_square = np.array([list(people_square[p].values()) for p in names_square])

scores_square = np.matmul(M_people_square, M_resturants_square)

# Can I get to the wieghts using: scores_square and M_resturants_square?
solved_for_weights = np.linalg.solve(M_resturants_square, scores_square)
solved_for_weights == M_people_square # FALSE

inv_scores = np.linalg.inv(scores_square)
using_inverse_for_weights = np.matmul(inv_scores, M_resturants_square)
using_inverse_for_weights == scores_square

inv_rests_square = np.linalg.inv(M_resturants_square)
using_inverse_rests = np.matmul(inv_rests_square, scores_square)


# Now I need to solve for the weights between M_resturants and random_results
# cannot use np.linalg.solve because my matrix are not square, nor should I assume they will be square
new_people_weights = np.linalg.solve(M_resturants_square, random_results)

# PICKLING OBJECTS SO I GET THE SAME RESULTS.
with open('/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/Homework_3/M_people', 'wb') as f:
    pickle.dump(M_people, f)

with open('/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/Homework_3/M_rests', 'wb') as f:
    pickle.dump(M_resturants, f)

