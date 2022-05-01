#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flight data visualizations using US Bureau of Transportation Statistics data
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sb
import numpy as np
from synthdid.model import SynthDID

def readToFrame():
        """
        Reads flight data CSVs into dataframe
        """
        international_path = 'data/International'
        international_files = glob.iglob(international_path + "/*.csv")

        # Create empty dataframe
        international_df = pd.DataFrame()
        # Consolidate data into one frame
        for f in international_files:
                df = pd.read_csv(f)
                # Split dataframe into multiple frames, by month
                splitByMonth = [y for x, y in df.groupby('MONTH', as_index=False)]
                for frame in splitByMonth:
                        # Sum the number of passengers, distanced travelled, etc.
                        numPassengers = frame['PASSENGERS'].sum()
                        distanceTravelled = frame['DISTANCE'].sum()
                        numUniqueAirlines = frame['UNIQUE_CARRIER_NAME'].nunique()
                        numUniqueDestinations = frame['DEST'].nunique()
                        Year = frame['YEAR'].iloc[0]
                        month = frame['MONTH'].iloc[0]

                        # Add the information into a dictionary to store
                        dict = {'Number of Passengers': numPassengers, 'Total distance':distanceTravelled, 'Number of Carriers':numUniqueAirlines,
                                'Number of Destinations':numUniqueDestinations, 'Year':Year,'Month':month}
                        # Append to final dataframe
                        international_df = international_df.append(dict,ignore_index=True)

        international_df = international_df.sort_values('Year')
        international_df['Year'] = international_df['Year'].astype(int)
        return international_df

def plotAirports(international_df):
        """
        Plots airports available in data on world map 
        """
        # Read in airport data
        airports = pd.read_csv("data/airports.csv", delimiter=',')

        # Plot airports on world map
        fig, ax = plt.subplots(facecolor='#FCF6F5FF')
        fig.set_size_inches(14, 7)
        ax.scatter(airports['LONGITUDE'], airports['LATITUDE'], s=1, alpha=1, edgecolors='#891e5b')
        ax.axis('off')

        plt.show()
        plt.savefig('airports.png')

def barPlot(international_df):
        """
        Plots flight data on bar chart
        """
        sb.set(rc = {'figure.figsize':(18,8)})
        sb.set_style("white")

        # Plot yearly number of carriers
        plot = sb.barplot(data=international_df, x="Year", y="Number of Carriers",palette="rocket",ci=None)
        sb.despine()
        plt.savefig('carriers.png')
        plt.clf() # clear figure

def lineGraph(international_df):
        """
        Plots flight data as line graph
        """
        # Convert month column from numbers to names
        international_df['Month Name'] = pd.to_datetime(international_df['Month'], format='%m').dt.month_name().str.slice(stop=3)

        # Plot number of passengers
        plot = sb.relplot(data=international_df, x="Year", y="Number of Passengers", hue="Month Name", kind="line",palette="magma")
        plot.set(xlim=(1991, 2021))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x))) # Make x ticks int instead of float
        plt.savefig('line-chart.png')
        plt.clf() # clear figure

def heatMap(international_df):
        """
        Plots flight data as heatmap
        """
        # Remove rows with missing data
        international_df.dropna(inplace=True)
        # Remove unnecessary columns
        international_df = international_df.filter(['Year', 'Month','Number of Passengers'])

        international_df['Year'] = pd.to_numeric(international_df.Year, errors='coerce')
        # Group by month and year, get the average
        international_df = international_df.groupby(['Month', 'Year']).mean()
        international_df = international_df.unstack(level=0)

        # Plot heatmap
        ax = sb.heatmap(international_df)

        ax.xaxis.tick_top() # put ye labels at top of chart
        xticks_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12), labels=xticks_labels)
        # Remove Axis Labels
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('heatmap.png')

def syntheticControl(international_files):
        """
        Applies the synthetic control statistical method to infer
        causality between event and drops in passenger numbers
        """
        # Synthetic method requires different build of dataframe
        international_dfs=[]
        for f in international_files:
                df = pd.read_csv(f)
                # Split dataframe into multiple frames, each with one origin
                splitByOrigin = [y for x, y in df.groupby('ORIGIN', as_index=False)]
                for frame in splitByOrigin:
                        # Sum the passengers for each origin and year
                        numPassengers = frame['PASSENGERS'].sum()
                        year = frame['YEAR'].iloc[0]
                        origin = frame['ORIGIN'].iloc[0]

                        # Add the information into a dictionary to store
                        dict = {'Number of Passengers': numPassengers, 'Year':year,'Origin':origin}

                        # Add information to new dataframe and append to list
                        temp_df = pd.DataFrame([dict])
                        international_dfs.append(temp_df)

        # Merge all frames stored in list
        international_df = pd.concat(international_dfs,ignore_index=True) 
        international_df = international_df.sort_values(by=['Origin','Year'])

        # Prepare the frame for the synthdid package
        pivotY = international_df.pivot(columns='Origin',index='Year',values='Number of Passengers')
        dropNull = pivotY.dropna(axis=1) # drop all null values
        s = dropNull.sum() # get the sums
        international_df = dropNull[s.sort_values(ascending=False).index[:50]] # show only the top 50 sums
        international_df = international_df.apply(pd.to_numeric) 

        # Perform synthetic control
        PRE_TERM = [1990, 2007]
        POST_TERM = [2008, 2011]
        TREATMENT = ["JFK"]

        sdid = SynthDID(international_df, PRE_TERM, POST_TERM, TREATMENT)
        sdid.fit(zeta_type="base")
        sdid.plot(model="sc")


def consumerConfidence(international_files):
        """
        Plots changes in confidence indices alongside passenger numbers
        """
        international_df = pd.DataFrame()
        for f in international_files:
                df = pd.read_csv(f)
                # Sum the number of passengers, distanced travelled, etc.
                numPassengers = df['PASSENGERS'].sum()
                distanceTravelled = df['DISTANCE'].sum()
                numUniqueAirlines = df['UNIQUE_CARRIER_NAME'].nunique()
                numUniqueDestinations = df['DEST'].nunique()
                year = int(df['YEAR'].iloc[0])
                origin = str(df['ORIGIN'].iloc[0])

                # Add the information into a dictionary to store
                dict = {'Passengers': numPassengers, 'Year':year, 'Distance':distanceTravelled,'Num Airlines':numUniqueAirlines,'Num Destinations':numUniqueDestinations,'Origin':origin}
                # Append to final dataframe
                international_df = international_df.append(dict,ignore_index=True)

        international_df = international_df.sort_values('Year')
        
        # Read in indices
        consumer_confidence = pd.read_csv('data/conf_indices/consumer_confidence.csv')
        business_confidence = pd.read_csv('data/conf_indices/business_confidence.csv')

        # Convert year and month cols to date time
        consumer_confidence['Year'] = pd.to_datetime(consumer_confidence['TIME']).dt.year
        consumer_confidence['Month'] = pd.to_datetime(consumer_confidence['TIME']).dt.month
        business_confidence['Year'] = pd.to_datetime(business_confidence['TIME']).dt.year
        business_confidence['Month'] = pd.to_datetime(business_confidence['TIME']).dt.month

        business_confidence = business_confidence.groupby('Year')['Value'].sum()
        consumer_confidence = consumer_confidence.groupby('Year')['Value'].sum()
        
        consumer_confidence=consumer_confidence.to_frame()
        business_confidence=business_confidence.to_frame()

        # Plot number of passengers
        fig,ax=plt.subplots()
        l1 = ax.plot(international_df['Year'], international_df['Passengers'],label="Passengers")
        ax.set_ylabel("Number of Passengers",fontsize=14)
        # Plot consumer index#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flight data visualizations using US Bureau of Transportation Statistics data
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sb
import numpy as np
from synthdid.model import SynthDID

def readToFrame():
        """
        Reads flight data CSVs into dataframe
        """
        international_path = 'data/International'
        international_files = glob.iglob(international_path + "/*.csv")

        # Create empty list
        international_dfs=[]
        for f in international_files:
                df = pd.read_csv(f)
                # Split dataframe into multiple frames, by month
                splitByMonth = [y for x, y in df.groupby('MONTH', as_index=False)]
                for frame in splitByMonth:
                        # Sum the number of passengers, distanced travelled, etc.
                        numPassengers = frame['PASSENGERS'].sum()
                        distanceTravelled = frame['DISTANCE'].sum()
                        numUniqueAirlines = frame['UNIQUE_CARRIER_NAME'].nunique()
                        numUniqueDestinations = frame['DEST'].nunique()
                        Year = frame['YEAR'].iloc[0]
                        month = frame['MONTH'].iloc[0]

                        # Add the information into a dictionary to store
                        dict = {'Number of Passengers': numPassengers, 'Total distance':distanceTravelled, 'Number of Carriers':numUniqueAirlines,
                                'Number of Destinations':numUniqueDestinations, 'Year':Year,'Month':month}

                        # Add information to dataframe and append to list
                        temp_df = pd.DataFrame([dict])
                        international_dfs.append(temp_df)

        # Merge all frames stored in list
        international_df = pd.concat(international_dfs,ignore_index=True)     
        international_df = international_df.sort_values('Year')
        international_df['Year'] = international_df['Year'].astype(int)
        return international_df

def plotAirports(international_df):
        """
        Plots airports available in data on world map 
        """
        # Read in airport data
        airports = pd.read_csv("data/airports.csv", delimiter=',')

        # Plot airports on world map
        fig, ax = plt.subplots(facecolor='#FCF6F5FF')
        fig.set_size_inches(14, 7)
        ax.scatter(airports['LONGITUDE'], airports['LATITUDE'], s=1, alpha=1, edgecolors='#891e5b')
        ax.axis('off')

        plt.show()
        plt.savefig('airports.png')


def barPlot(international_df):
        """
        Plots flight data on bar chart
        """
        sb.set(rc = {'figure.figsize':(18,8)})
        sb.set_style("white")

        # Plot yearly number of carriers
        plot = sb.barplot(data=international_df, x="Year", y="Number of Carriers",palette="rocket",ci=None)
        sb.despine()
        plt.savefig('carriers.png')
        plt.clf() # clear figure

def lineGraph(international_df):
        """
        Plots flight data as line graph
        """
        # Convert month column from numbers to names
        international_df['Month Name'] = pd.to_datetime(international_df['Month'], format='%m').dt.month_name().str.slice(stop=3)

        # Plot number of passengers
        plot = sb.relplot(data=international_df, x="Year", y="Number of Passengers", hue="Month Name", kind="line",palette="magma")
        plot.set(xlim=(1991, 2021))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x))) # Make x ticks int instead of float
        plt.savefig('line-chart.png')
        plt.clf() # clear figure

def heatMap(international_df):
        """
        Plots flight data as heatmap
        """
        # Remove rows with missing data
        international_df.dropna(inplace=True)
        # Remove unnecessary columns
        international_df = international_df.filter(['Year', 'Month','Number of Passengers'])

        international_df['Year'] = pd.to_numeric(international_df.Year, errors='coerce')
        # Group by month and year, get the average
        international_df = international_df.groupby(['Month', 'Year']).mean()
        international_df = international_df.unstack(level=0)

        # Plot heatmap
        ax = sb.heatmap(international_df)

        ax.xaxis.tick_top() # put ye labels at top of chart
        xticks_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12), labels=xticks_labels)
        # Remove Axis Labels
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('heatmap.png')

def syntheticControl(international_files):
        """
        Applies the synthetic control statistical method to infer
        causality between event and drops in passenger numbers
        """
        # Synthetic method requires different build of dataframe
        international_dfs=[]
        for f in international_files:
                df = pd.read_csv(f)
                # Split dataframe into multiple frames, each with one origin
                splitByOrigin = [y for x, y in df.groupby('ORIGIN', as_index=False)]
                for frame in splitByOrigin:
                        # Sum the passengers for each origin and year
                        numPassengers = frame['PASSENGERS'].sum()
                        year = frame['YEAR'].iloc[0]
                        origin = frame['ORIGIN'].iloc[0]

                        # Add the information into a dictionary to store
                        dict = {'Number of Passengers': numPassengers, 'Year':year,'Origin':origin}

                        # Add information to new dataframe and append to list
                        temp_df = pd.DataFrame([dict])
                        international_dfs.append(temp_df)

        # Merge all frames stored in list
        international_df = pd.concat(international_dfs,ignore_index=True) 
        international_df = international_df.sort_values(by=['Origin','Year'])

        # Prepare the frame for the synthdid package
        pivotY = international_df.pivot(columns='Origin',index='Year',values='Number of Passengers')
        dropNull = pivotY.dropna(axis=1) # drop all null values
        s = dropNull.sum() # get the sums
        international_df = dropNull[s.sort_values(ascending=False).index[:50]] # show only the top 50 sums
        international_df = international_df.apply(pd.to_numeric) 

        # Perform synthetic control
        PRE_TERM = [1990, 2007]
        POST_TERM = [2008, 2011]
        TREATMENT = ["JFK"]

        sdid = SynthDID(international_df, PRE_TERM, POST_TERM, TREATMENT)
        sdid.fit(zeta_type="base")
        sdid.plot(model="sc")


def consumerConfidence(international_files):
        """
        Plots changes in confidence indices alongside passenger numbers
        """
        international_df = pd.DataFrame()
        for f in international_files:
                df = pd.read_csv(f)
                # Sum the number of passengers, distanced travelled, etc.
                numPassengers = df['PASSENGERS'].sum()
                distanceTravelled = df['DISTANCE'].sum()
                numUniqueAirlines = df['UNIQUE_CARRIER_NAME'].nunique()
                numUniqueDestinations = df['DEST'].nunique()
                year = int(df['YEAR'].iloc[0])
                origin = str(df['ORIGIN'].iloc[0])

                # Add the information into a dictionary to store
                dict = {'Passengers': numPassengers, 'Year':year, 'Distance':distanceTravelled,'Num Airlines':numUniqueAirlines,'Num Destinations':numUniqueDestinations,'Origin':origin}
                # Append to final dataframe
                international_df = international_df.append(dict,ignore_index=True)

        international_df = international_df.sort_values('Year')
        
        # Read in indices
        consumer_confidence = pd.read_csv('data/conf_indices/consumer_confidence.csv')
        business_confidence = pd.read_csv('data/conf_indices/business_confidence.csv')

        # Convert year and month cols to date time
        consumer_confidence['Year'] = pd.to_datetime(consumer_confidence['TIME']).dt.year
        consumer_confidence['Month'] = pd.to_datetime(consumer_confidence['TIME']).dt.month
        business_confidence['Year'] = pd.to_datetime(business_confidence['TIME']).dt.year
        business_confidence['Month'] = pd.to_datetime(business_confidence['TIME']).dt.month

        business_confidence = business_confidence.groupby('Year')['Value'].sum()
        consumer_confidence = consumer_confidence.groupby('Year')['Value'].sum()
        
        consumer_confidence=consumer_confidence.to_frame()
        business_confidence=business_confidence.to_frame()

        # Plot number of passengers
        fig,ax=plt.subplots()
        l1 = ax.plot(international_df['Year'], international_df['Passengers'],label="Passengers")
        ax.set_ylabel("Number of Passengers",fontsize=14)
        # Plot consumer index
        ax2=ax.twinx()
        l2 = ax2.plot(consumer_confidence.index, consumer_confidence['Value'],color="green",label='Consumer Confidence')
        ax2.set_ylabel("Confidence Index",fontsize=14)
        # Plot business index
        l3 = ax2.plot(business_confidence.index, business_confidence['Value'],color="pink",label='Business Confidence')
        ax2.legend( handles=l1+l2+l3 )
        plt.show()

def main():
        international_path = 'data/International'
        international_files = glob.iglob(international_path + "/*.csv")
        international_df = readToFrame()
        plotAirports(international_df)
        barPlot(international_df)
        lineGraph(international_df)
        heatMap(international_df)
        syntheticControl(international_files)
        consumerConfidence(international_files)
        

if __name__ == "__main__":
    main()
        ax2=ax.twinx()
        l2 = ax2.plot(consumer_confidence.index, consumer_confidence['Value'],color="green",label='Consumer Confidence')
        ax2.set_ylabel("Confidence Index",fontsize=14)
        # Plot business index
        l3 = ax2.plot(business_confidence.index, business_confidence['Value'],color="pink",label='Business Confidence')
        ax2.legend( handles=l1+l2+l3 )

        plt.show()

def main():
        international_path = 'data/International'
        international_files = glob.iglob(international_path + "/*.csv")
        international_df = readToFrame()
        plotAirports(international_df)
        barPlot(international_df)
        lineGraph(international_df)
        heatMap(international_df)
        syntheticControl(international_files)
        consumerConfidence(international_files)
        

if __name__ == "__main__":
    main()
