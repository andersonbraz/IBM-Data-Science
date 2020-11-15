# Capstone Project - Final assignment of the Applied Data Science Capstone Course by IBM on Coursera

## Introduction

Miami, officially the City of Miami, is a metropolis located in southeastern Florida in the United States. It is the third most populous metropolis on the East coast of the United States, and it is the seventh largest in the country. The city is an economic powerhouse, serving as the financial and business capital of Latin America. The city has the third tallest skyline in the U.S. with over 300 high-rises, 55 of which exceed 491 ft (150 m).

Miami is a major center and leader in finance, commerce, culture, arts, and international trade. The metro area is by far the largest urban economy in Florida and the 12th largest in the United States, with a GDP of $344.9 billion as of 2017. In 2020, Miami was classified as a Beta + level global city by the GaWC. In 2019, Miami ranked seventh in the United States and 31st among global cities in terms of business activity, human capital, information exchange, cultural experience, and political engagement. According to a 2018 UBS study of 77 world cities, the city was ranked as the third-richest in the United States and the eighth-richest in the world in terms of purchasing power. Miami is nicknamed the "Capital of Latin America" and is the largest city with a Cuban-American plurality.

This final project explores the best locations for Brazilian restaurants throughout the city of Miami. As Miami is considered the "Capital of Latin America", it has a long tradition of receive many Latin visitors and many of them Brazilians. That's why potentially the owner of the new Brazilian restaurant can have great success and consistent profit. However, as with any business, opening a new restaurant requires serious considerations and is more complicated than it seems from the first glance. In particular, the location of the restaurant is one of the most important factors that will affect whether it will have success or a failure. So our project will attempt to answer the questions “Where should the investor open a Brazilian Restaurant?” and “Where should I go If I want great Brazilian food?”

## Data

In order to answer the above questions, data on Miami City neighborhoods, boroughs to include boundaries, latitude, longitude, restaurants, and restaurant ratings and tips are required.

Miami City data containing the neighborhoods and boroughs, latitudes, and longitudes will be obtained from the data source: [miami_dataset.csv](https://raw.githubusercontent.com/andersonbraz/ibm-data-science/master/notes/module-09/week-02/coursera_capstone/miami_dataset.csv)

All data related to locations and quality of Brazilian restaurants will be obtained via the FourSquare API utilized via the Request library in Python.


## Methodology

• Data will be collected from [miami_dataset.csv](https://raw.githubusercontent.com/andersonbraz/ibm-data-science/master/notes/module-09/week-02/coursera_capstone/miami_dataset.csv) and cleaned and processed into a dataframe.

• FourSquare be used to locate all venues and then filtered by Brazilian restaurants. Ratings, tips, and likes by users will be counted and added to the dataframe.

• Data will be sorted based on rankings.

• Finally, the data will be visually assessed using graphing from Python libraries.

## Problem Statement

What is/are the best location(s) for Brazilian cuisine in Miami City? In what Neighborhood and/or borough should the investor open a Brazilian restaurant to have the best chance of being successful? Where would I go in Miami City to have the best Brazilian food?